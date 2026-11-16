//! The headless runner (`threert-render live`): the frame loop at a fixed rate or in lockstep, the commands on stdin
//! and their parsing.

use super::*;

/// Headless runner: renders at a fixed frame rate and saves snapshots on request (tools/bench-live.ts).
pub fn run(presets: &Path, opts: LiveOpts) -> Result<()> {
    let (w, h) = (opts.width, opts.height);
    let (cmd_tx, cmd_rx) = mpsc::sync_channel::<Cmd>(CMD_AHEAD);
    spawn_stdin_reader(cmd_tx);
    let (mut game, mut core) = crate::script::GameSession::start(&opts.bundle, &opts.room, &opts.assets, gpu::Gpu::new()?, presets, w, h)?;
    // The host channel: this runner is the app. How the game was opened, once after setup; what the game posts goes
    // out on stdout (`emit_posts`), and whoever drives the runner answers with `{"cmd":"host"}`.
    if let Err(e) = game.host_messages(&mut core, &[crate::host::launch_message(opts.launch.as_ref())]) {
        eprintln!("live: launch: {e:#}");
    }
    emit_posts(&mut game);
    // Frame-script cost (script + commit), reported every 2 s.
    let (mut fs_sum, mut fs_max, mut fs_n) = (0.0f32, 0.0f32, 0u32);
    core.bounces = opts.bounces;
    core.set_display_scale(opts.display_scale);
    if let Some(s) = opts.scale {
        core.set_scale(s);
    }
    eprintln!("live: load {}", serde_json::Value::Object(core.load_timings().iter().map(|(k, v)| (k.clone(), (*v).into())).collect()));
    eprintln!("live: {}x{} @{} fps on {} ({})", w, h, opts.fps, core.g.adapter_info.name, if core.g.hwrt { "HW RT" } else { "BVH" });
    let staging = core.g.device.create_buffer(&wgpu::BufferDescriptor { label: Some("snapshot"), size: (w * h * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    // known-future `--stream`: frames out through a ring of three read-back buffers, one frame behind the GPU, and a
    // writer thread (the reader of the FIFO paces it); its byte buffers come back for reuse.
    let mut stream = opts.stream.clone().map(|p| Stream::new(&core.g.device, p, (w * h * 4) as u64));
    let mut emits: Vec<(u32, u32)> = vec![]; // core.frame ranges to stream
    let mut lock_spp = 1u32;
    let frame_dt = Duration::from_secs_f64(1.0 / opts.fps as f64);
    let mut next = Instant::now();
    let mut snapshots: Vec<(PathBuf, u32, u32)> = vec![]; // path, min history, first eligible frame
    let mut syncs: Vec<(String, u32)> = vec![]; // id, frame after which to answer
    let mut spp = 1u32;
    let mut last = Instant::now();
    let start = Instant::now();
    let mut log = opts.frame_log.as_ref().map(|p| std::fs::File::create(p).map(std::io::BufWriter::new)).transpose()?;
    let mut last_frame = Instant::now();
    let mut prev_start = Instant::now();
    let mut lockstep: Option<f32> = None;
    let mut budget = 0u32; // lockstep: frames the commands so far asked for and not yet rendered
    let mut held: std::collections::VecDeque<Cmd> = Default::default(); // lockstep: commands after a step not yet rendered
    loop {
        let t_frame = Instant::now();
        // In lockstep the commands run in order with the frames: a step's frames render before the commands after it
        // (known-future: a film driver sends frame k+1's view right behind frame k's step; held here until k is drawn).
        let mut cmds: std::collections::VecDeque<Cmd> = std::mem::take(&mut held);
        if lockstep.is_some() && budget == 0 && cmds.is_empty() {
            if let Some(st) = &mut stream {
                st.drain(&core.g.device, 0)?;
            }
            match cmd_rx.recv() {
                Ok(c) => cmds.push_back(c),
                Err(_) => return Ok(()),
            }
        }
        let room = CMD_AHEAD.saturating_sub(cmds.len());
        cmds.extend(std::iter::from_fn(|| cmd_rx.try_recv().ok()).take(room));
        while let Some(c) = if lockstep.is_some() && budget > 0 { None } else { cmds.pop_front() } {
            match c {
                Cmd::Camera(c) => core.set_camera(c),
                Cmd::Lockstep(dt) => {
                    lockstep = dt;
                    core.fixed_dt = dt;
                    budget = 0;
                }
                Cmd::Step { cam, frames, sync, snapshot, emit } => {
                    if let Some(c) = cam {
                        core.set_camera(c);
                    }
                    let n = frames.max(1);
                    if lockstep.is_some() {
                        budget += n;
                    }
                    let last = core.frame + n - 1;
                    if emit {
                        emits.push((core.frame + budget - n, core.frame + budget - 1));
                    }
                    if let Some(p) = snapshot {
                        snapshots.push((p, 0, last));
                    }
                    if let Some(id) = sync {
                        syncs.push((id, last));
                    }
                }
                Cmd::Snapshot(path, after, frames) => snapshots.push((path, after.unwrap_or(0), core.frame + frames.unwrap_or(1).max(1) - 1)),
                Cmd::Sync(id, frames) => syncs.push((id, core.frame + frames.max(1) - 1)),
                Cmd::Reset => core.reset_history(),
                Cmd::Time(t) => core.set_time(t),
                Cmd::Mode(m) => core.view_mode = m,
                Cmd::Bounces(b) => core.bounces = b,
                Cmd::Scale(s) => core.set_scale(s),
                Cmd::Event(kind, payload) => {
                    if let Err(e) = game.event(&mut core, &kind, payload) {
                        eprintln!("live: event {kind} failed: {e:#}");
                    }
                }
                Cmd::Click(u, v) => {
                    if let Err(e) = game.click(&mut core, u, v) {
                        eprintln!("live: click failed: {e:#}");
                    }
                }
                Cmd::Host(messages) => {
                    if let Err(e) = game.host_messages(&mut core, &messages) {
                        eprintln!("live: host messages failed: {e:#}");
                    }
                }
                Cmd::View { cam, dof, ev100 } => core.set_view(cam, dof, ev100),
                Cmd::Texture { key, img } => {
                    let t = Instant::now();
                    let Some((img, chains)) = img.recv().ok().flatten() else {
                        eprintln!("live: texture {key}: the image could not be read");
                        continue;
                    };
                    let done = if chains.is_empty() { core.g.replace_texture(&key, &img) } else { Ok(core.g.write_texture_chains(&chains)) };
                    match done {
                        Ok(n) => {
                            if std::env::var("THREERT_KF_VERBOSE").is_ok() {
                                eprintln!("live: texture {key} x{n} in {:.2} ms", t.elapsed().as_secs_f64() * 1e3)
                            }
                        }
                        Err(e) => eprintln!("live: texture {key}: {e:#}"),
                    }
                }
                Cmd::Emissive { matched, rgb } => {
                    if core.set_emissive(matched, rgb) == 0 {
                        eprintln!("live: emissive {matched:?}: no material emits that");
                    }
                }
                Cmd::SppCap(n) => lock_spp = n.max(1),
                Cmd::Status(_) => {}
                Cmd::Quit => {
                    if let Some(st) = stream.take() {
                        st.finish(&core.g.device)?;
                    }
                    return Ok(());
                }
            }
        }
        held = cmds;
        // The frame script runs on wall-clock time (frames slower than the target rate still animate at real speed).
        if lockstep.is_some() && budget == 0 {
            emit_posts(&mut game);
            continue; // commands without frames (events, switches): wait for a step
        }
        let dt = if freeze_time() { 0.0 } else { lockstep.unwrap_or_else(|| last.elapsed().as_secs_f32().min(0.1)) };
        last = Instant::now();
        core.advance_time(dt);
        if let Err(e) = game.frame(&mut core, dt) {
            eprintln!("live: frame script: {e:#}");
        }
        emit_posts(&mut game);
        if game.host.has_frame() {
            (fs_sum, fs_max, fs_n) = (fs_sum + game.frame_ms, fs_max.max(game.frame_ms), fs_n + 1);
            if fs_n == opts.fps * 2 {
                eprintln!("live: frame script {:.3} ms avg, {:.3} ms max (script + commit)", fs_sum / fs_n as f32, fs_max);
                (fs_sum, fs_max, fs_n) = (0.0, 0.0, 0);
            }
        }
        // Snapshot metadata is this frame's (history / spp at render time), so copy the output whenever one is pending.
        let pending = !snapshots.is_empty();
        let fno0 = core.frame;
        let emit_this = stream.is_some() && emits.iter().any(|&(a, b)| fno0 >= a && fno0 <= b);
        emits.retain(|&(_, b)| b > fno0);
        let ring = stream.as_ref().filter(|_| emit_this).map(|st| &st.ring[st.next]);
        core.render(spp, |e, out| {
            if pending {
                e.copy_buffer_to_buffer(out, 0, &staging, 0, (w * h * 4) as u64);
            }
            if let Some(r) = ring {
                e.copy_buffer_to_buffer(out, 0, r, 0, (w * h * 4) as u64);
            }
        });
        if emit_this {
            if let Some(st) = &mut stream {
                st.queued(&core.g.device)?;
            }
        }
        game.click_shown();
        let (hist, fno) = (core.history_len as u32, core.frame);
        syncs.retain(|(id, from)| {
            if fno > *from {
                eprintln!("live: sync {id}");
            }
            fno <= *from
        });
        if snapshots.iter().any(|(_, after, from)| fno > *from && hist >= *after) {
            staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            core.g.device.poll(wgpu::PollType::wait_indefinitely())?;
            let px = staging.slice(..).get_mapped_range()?.to_vec();
            staging.unmap();
            let mut hdr: Option<Vec<f32>> = None;
            snapshots.retain(|(path, after, from)| {
                if fno > *from && hist >= *after {
                    if path.extension().is_some_and(|e| e == "signals") {
                        // The signals apart (`read_signals`), as `render --signals` writes them.
                        if let Ok([d, s, e]) = core.read_signals() {
                            let stem = path.with_extension("");
                            for (name, v) in [("diffuse", d), ("specular", s), ("emission", e)] {
                                let _ = std::fs::write(format!("{}-{name}.raw", stem.display()), bytemuck::cast_slice(&v));
                            }
                        }
                    } else if path.extension().is_some_and(|e| e == "raw") {
                        // Linear radiance (RGB f32, output size), as `serve` live jobs write it.
                        if hdr.is_none() {
                            hdr = core.read_hdr().ok();
                        }
                        let _ = std::fs::write(path, bytemuck::cast_slice(hdr.as_deref().unwrap_or(&[])));
                    } else {
                        save_bgra(path, &px, w, h);
                    }
                    eprintln!("live: snapshot {} at history {hist} ({spp} spp/frame, GPU {:.2} ms, scale {:.2})", path.display(), core.gpu_ms, core.render_scale);
                    false
                } else {
                    true
                }
            });
        }
        if let Some(l) = &mut log {
            use std::io::Write;
            let line = serde_json::json!({
                "t": start.elapsed().as_secs_f64() * 1e3, "interval": last_frame.elapsed().as_secs_f64() * 1e3, "start": t_frame.duration_since(prev_start).as_secs_f64() * 1e3,
                "cpu": t_frame.elapsed().as_secs_f64() * 1e3, "script": game.frame_ms, "js": game.js_ms, "commit": game.commit_ms, "skin": core.cpu_skin_ms, "upload": core.cpu_upload_ms, "blas": core.gpu_blas_ms, "wait": core.cpu_wait_ms, "gpu": core.gpu_ms, "trace": core.gpu_trace_ms, "edit": core.edit_ms, "instanced": core.instanced_ms, "edit_split": [core.edit_split.tris, core.edit_split.upload, core.edit_split.blas, core.edit_split.added, core.edit_split.deformed, core.edit_split.objects], "rebuild": core.rebuild_ms, "fallback": core.fallback, "fallbacks": core.fallbacks, "room": core.room_stats(), "top": core.top_stats(), "voxel": std::mem::take(&mut game.host.voxels.borrow_mut().ticks), "physics": game.host.physics.borrow_mut().as_mut().map(|p| { let s = p.take_stats(); serde_json::json!({ "ticks": s.ticks, "step": s.step_ms, "sync": s.sync_ms, "bodies": s.bodies, "moved": s.moved, "kicks": s.kicks }) }), "scale": core.render_scale, "history": core.history_len, "guard": core.guard_count(),
            });
            writeln!(l, "{line}")?;
        }
        // (Summed over the frame's commits, a game event's and the frame script's: reset once logged.)
        (core.edit_ms, core.instanced_ms, core.edit_split, core.rebuild_ms, core.fallback) = (0.0, 0.0, EditSplit::default(), 0.0, None);
        prev_start = t_frame;
        last_frame = Instant::now();
        core.govern_resolution(opts.fps);
        spp = if core.render_scale < 1.0 { 1 } else if lockstep.is_some() { lock_spp } else { next_spp(spp, core.gpu_ms, opts.fps) };
        let _ = core.g.device.poll(wgpu::PollType::Poll);
        if lockstep.is_some() {
            budget -= 1;
            continue; // no pacing: frames as fast as the GPU renders them
        }
        // Pace to the frame rate; when the GPU is the bottleneck the two-frames-in-flight limit paces instead (sleeping
        // after a frame that did not wait would alternate long and short frames).
        next += frame_dt;
        let now = Instant::now();
        if next > now && core.gpu_ms < frame_dt.as_secs_f32() * 1000.0 * 0.95 {
            sleep_until(next);
        } else {
            next = now;
        }
    }
}

/// What the game posted on the host channel since the last call, as one line on stdout: `{"host":[{"t":...}, ...]}`
/// (nothing when it posted nothing).
fn emit_posts(game: &mut crate::script::GameSession) {
    use std::io::Write;
    let posts = game.take_posts();
    if !posts.is_empty() {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{}", serde_json::json!({ "host": posts }));
        let _ = out.flush();
    }
}

/// Sleeps until `t` without trusting the timer's precision: macOS gives a process that is not a foreground app timer
/// leeway of half the interval (a 16 ms sleep returned after 24, measured from a terminal-launched viewport and from
/// plain Python alike). Pacing sleeps of the headless viewport overslept by ~5 ms once the GPU fit the frame: Bistro's
/// camera path read frame interval p95 21.9 ms at GPU p95 16.0. Sleep 60 % of what is left, again, and yield through
/// the last 1.5 ms.
fn sleep_until(t: Instant) {
    loop {
        let now = Instant::now();
        if now >= t {
            return;
        }
        let rem = t - now;
        if rem > Duration::from_micros(1500) {
            std::thread::sleep(rem.mul_f64(0.6));
        } else {
            std::thread::yield_now();
        }
    }
}

pub fn save_bgra(path: &Path, px: &[u8], w: u32, h: u32) {
    let rgba: Vec<u8> = px.chunks(4).flat_map(|p| [p[2], p[1], p[0], 255]).collect();
    if let Err(e) = image::save_buffer(path, &rgba, w, h, image::ColorType::Rgba8) {
        eprintln!("live: snapshot failed: {e}");
    }
}

/// Commands read ahead of the frames at most (the reader then waits): a film driver sends about four a frame.
const CMD_AHEAD: usize = 96;

fn spawn_stdin_reader(cmd_tx: mpsc::SyncSender<Cmd>) {
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            let Some(cmd) = parse_cmd(&line) else { continue };
            if cmd_tx.send(cmd).is_err() {
                break;
            }
        }
        let _ = cmd_tx.send(Cmd::Quit);
    });
}

pub fn parse_cmd(line: &str) -> Option<Cmd> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    let vec = |k: &str| v[k].as_array().map(|a| Vec3::new(a[0].as_f64().unwrap_or(0.0) as f32, a[1].as_f64().unwrap_or(0.0) as f32, a[2].as_f64().unwrap_or(0.0) as f32));
    Some(match v["cmd"].as_str()? {
        "camera" if v.get("sync").is_some() || v.get("snapshot").is_some() || v.get("frames").is_some() => Cmd::Step {
            cam: Some(Camera::look_at(vec("pos")?, vec("target")?, v["tan"].as_f64().unwrap_or(0.0) as f32)),
            frames: v["frames"].as_u64().unwrap_or(1) as u32,
            sync: v.get("sync").map(|s| s.as_str().map(str::to_string).unwrap_or_else(|| s.to_string())),
            snapshot: v["snapshot"].as_str().map(PathBuf::from),
            emit: v["emit"].as_bool().unwrap_or(false),
        },
        "view" => {
            let (fwd, up) = (vec("fwd")?.normalize(), vec("up")?.normalize());
            let right = fwd.cross(up).normalize();
            let dof = v["dof"].as_array().filter(|a| a.len() == 3).map(|a| crate::desc::Dof {
                focus_distance: a[0].as_f64().unwrap_or(0.0) as f32,
                f_stop: a[1].as_f64().unwrap_or(0.0) as f32,
                sensor_width: a[2].as_f64().unwrap_or(0.0) as f32,
            });
            Cmd::View { cam: Camera { pos: vec("pos")?, fwd, up: right.cross(fwd), right, tan_half_fov: v["tan"].as_f64().unwrap_or(0.0) as f32 }, dof, ev100: v["ev100"].as_f64().map(|e| e as f32) }
        }
        "emissive" => {
            let f3 = |k: &str| v[k].as_array().filter(|a| a.len() == 3).map(|a| [a[0].as_f64().unwrap_or(0.0) as f32, a[1].as_f64().unwrap_or(0.0) as f32, a[2].as_f64().unwrap_or(0.0) as f32]);
            Cmd::Emissive { matched: f3("match")?, rgb: f3("rgb")? }
        }
        "texture" => {
            // Decoded on rayon's pool as the command arrives, ahead of the frames and in parallel (a film's screen is a
            // 4-megapixel PNG a frame): raw RGBA8 (`.rgba`, with w and h) or any image file. The frame that uses it
            // waits for it in order.
            let path = PathBuf::from(v["path"].as_str()?);
            let (w, h) = (v["w"].as_u64().unwrap_or(0) as u32, v["h"].as_u64().unwrap_or(0) as u32);
            let mask = v["mask"].as_str().map(PathBuf::from);
            let key = v["key"].as_str()?.to_string();
            let (tx, rx) = mpsc::channel();
            let key2 = key.clone();
            rayon::spawn(move || {
                let mut img = if path.extension().is_some_and(|e| e == "rgba") {
                    std::fs::read(&path).ok().and_then(|b| image::RgbaImage::from_raw(w, h, b))
                } else {
                    image::open(&path).ok().map(|i| i.to_rgba8())
                };
                // known-future: a mask multiplied in (his screen: image x mask), red channel, same size
                if let (Some(im), Some(m)) = (img.as_mut(), mask.as_ref().and_then(|m| kf_mask(m))) {
                    if m.dimensions() == im.dimensions() {
                        for (p, q) in im.pixels_mut().zip(m.pixels()) {
                            let k = q[0] as u32;
                            for c in 0..3 {
                                p[c] = ((p[c] as u32 * k + 127) / 255) as u8;
                            }
                        }
                    }
                }
                // the padded rects and mip chains here too, off the render thread (pages known: the registry)
                let _ = tx.send(img.map(|im| {
                    let chains = crate::texpack::chains_for(&key2, &im);
                    (im, chains)
                }));
            });
            Cmd::Texture { key, img: rx }
        }
        "camera" => Cmd::Camera(Camera::look_at(vec("pos")?, vec("target")?, v["tan"].as_f64().unwrap_or(0.0) as f32)),
        "frames" => Cmd::Step {
            cam: None,
            frames: v["n"].as_u64().unwrap_or(1) as u32,
            sync: v.get("sync").map(|s| s.as_str().map(str::to_string).unwrap_or_else(|| s.to_string())),
            snapshot: v["snapshot"].as_str().map(PathBuf::from),
            emit: v["emit"].as_bool().unwrap_or(false),
        },
        "lockstep" => Cmd::Lockstep(if v["on"].as_bool().unwrap_or(true) { Some(v["dt"].as_f64().unwrap_or(1.0 / 60.0) as f32) } else { None }),
        "event" => Cmd::Event(v["type"].as_str()?.to_string(), if v["payload"].is_null() { serde_json::json!({}) } else { v["payload"].clone() }),
        "click" => Cmd::Click(v["u"].as_f64()? as f32, v["v"].as_f64()? as f32),
        "host" => Cmd::Host(v["messages"].as_array()?.clone()),
        "snapshot" => Cmd::Snapshot(PathBuf::from(v["path"].as_str().unwrap_or("snapshot.png")), v["after"].as_u64().map(|a| a as u32), v["frames"].as_u64().map(|a| a as u32)),
        "reset" => Cmd::Reset,
        "time" => Cmd::Time(v["t"].as_f64()? as f32),
        "mode" => Cmd::Mode(v["mode"].as_u64()? as u32),
        "bounces" => Cmd::Bounces(v["bounces"].as_u64()?.clamp(1, 16) as u32),
        "spp" => Cmd::SppCap(v["max"].as_u64()?.clamp(1, 64) as u32),
        "scale" => Cmd::Scale(v["scale"].as_f64()? as f32),
        "sync" => Cmd::Sync(v["id"].as_str().map(str::to_string).unwrap_or_else(|| v["id"].to_string()), v["frames"].as_u64().unwrap_or(1) as u32),
        "quit" => Cmd::Quit,
        _ => return None,
    })
}

pub fn parse_opts(args: &[String]) -> Result<LiveOpts> {
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let num = |name: &str, d: u32| -> Result<u32> { flag(name).map(|v| v.parse().map_err(|_| anyhow!("{name} must be a number"))).transpose().map(|v| v.unwrap_or(d)) };
    Ok(LiveOpts {
        width: num("--width", 1280)?,
        height: num("--height", 720)?,
        fps: num("--fps", 60)?,
        bounces: num("--bounces", 4)?,
        // (Without `--scale`: full resolution, or dynamic resolution on a display's physical pixels.)
        scale: parse_scale(flag("--scale").as_deref(), if flag("--display-scale").is_some() { None } else { Some(1.0) })?,
        bundle: PathBuf::from(args.get(2).filter(|a| !a.starts_with("--")).ok_or_else(|| anyhow!("game bundle required"))?),
        assets: PathBuf::from(flag("--assets").unwrap_or_else(|| ".".into())),
        room: flag("--room").unwrap_or_else(|| "local".into()),
        frame_log: flag("--frame-log").map(PathBuf::from),
        display_scale: flag("--display-scale").map(|v| v.parse::<f32>()).transpose()?.unwrap_or(1.0),
        launch: parse_launch(flag("--launch").as_deref())?,
        stream: flag("--stream").map(PathBuf::from),
    })
}

/// `--launch '{"room":"K7Q4MZ","name":"Ada"}'`: the `params` of the host channel's `launch` message.
pub fn parse_launch(v: Option<&str>) -> Result<Option<serde_json::Value>> {
    v.map(|t| serde_json::from_str::<serde_json::Value>(t).ok().filter(|p| p.is_object()).ok_or_else(|| anyhow!("--launch must be a JSON object, as in --launch '{{\"room\":\"K7Q4MZ\"}}'"))).transpose()
}

/// known-future `--stream`: frames read back through a ring of buffers one frame behind the GPU and written to a FIFO
/// by a thread of their own, in order. The byte buffers go back and forth, so no frame allocates.
struct Stream {
    ring: Vec<wgpu::Buffer>,
    next: usize,
    pend: std::collections::VecDeque<(usize, Arc<std::sync::atomic::AtomicBool>)>,
    tx: Option<mpsc::SyncSender<Vec<u8>>>,
    back: mpsc::Receiver<Vec<u8>>,
    spare: Vec<Vec<u8>>,
    writer: Option<std::thread::JoinHandle<std::io::Result<()>>>,
    pub(super) read_ms: f64,
}

impl Stream {
    fn new(d: &wgpu::Device, path: PathBuf, size: u64) -> Stream {
        let ring = (0..3).map(|_| d.create_buffer(&wgpu::BufferDescriptor { label: Some("stream"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })).collect();
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(2);
        let (btx, back) = mpsc::channel::<Vec<u8>>();
        let writer = std::thread::spawn(move || -> std::io::Result<()> {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(&path)?;
            for b in rx {
                f.write_all(&b)?;
                let _ = btx.send(b);
            }
            f.flush()
        });
        Stream { ring, next: 0, pend: Default::default(), tx: Some(tx), back, spare: vec![], writer: Some(writer), read_ms: 0.0 }
    }

    /// The frame just submitted copied its output into `ring[next]`: map it, and read what is older than one frame.
    fn queued(&mut self, d: &wgpu::Device) -> Result<()> {
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = done.clone();
        self.ring[self.next].slice(..).map_async(wgpu::MapMode::Read, move |r| {
            if r.is_ok() {
                flag.store(true, Ordering::Release);
            }
        });
        self.pend.push_back((self.next, done));
        self.next = (self.next + 1) % self.ring.len();
        self.drain(d, 1)
    }

    /// Reads (waiting for them) all but `keep` of the frames in flight, oldest first, and hands them to the writer.
    fn drain(&mut self, d: &wgpu::Device, keep: usize) -> Result<()> {
        while self.pend.len() > keep {
            let (i, done) = self.pend.pop_front().unwrap();
            let t = Instant::now();
            while !done.load(Ordering::Acquire) {
                d.poll(wgpu::PollType::Poll)?;
                if !done.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_micros(100));
                }
            }
            self.spare.extend(self.back.try_iter());
            let mut buf = self.spare.pop().unwrap_or_default();
            buf.clear();
            buf.extend_from_slice(&self.ring[i].slice(..).get_mapped_range()?);
            self.ring[i].unmap();
            self.read_ms += t.elapsed().as_secs_f64() * 1e3;
            if let Some(tx) = &self.tx {
                tx.send(buf).map_err(|_| anyhow!("stream: the writer stopped (reader gone?)"))?;
            }
        }
        Ok(())
    }

    fn finish(mut self, d: &wgpu::Device) -> Result<()> {
        self.drain(d, 0)?;
        drop(self.tx.take());
        if let Some(w) = self.writer.take() {
            w.join().map_err(|_| anyhow!("stream writer panicked"))??;
        }
        eprintln!("live: stream closed (read-back {:.1} ms in all)", self.read_ms);
        Ok(())
    }
}

/// known-future: a texture mask, decoded once (the screen's corner and notch mask).
fn kf_mask(p: &std::path::Path) -> Option<Arc<image::RgbaImage>> {
    static MASKS: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<PathBuf, Arc<image::RgbaImage>>>> = std::sync::OnceLock::new();
    let m = MASKS.get_or_init(Default::default);
    if let Some(i) = m.lock().unwrap().get(p) {
        return Some(i.clone());
    }
    let i = Arc::new(image::open(p).ok()?.to_rgba8());
    m.lock().unwrap().insert(p.to_path_buf(), i.clone());
    Some(i)
}
