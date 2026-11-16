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
    // known-future `--stream`: frames out through a ring of read-back buffers and a writer thread that writes them from
    // the mapped buffers (the reader of the FIFO paces it).
    let nv12 = (opts.stream.is_some() && opts.stream_nv12).then(|| Nv12::new(&core.g.device, &core.targets.out, w, h));
    let mut stream = opts.stream.clone().map(|p| Stream::new(&core.g.device, p, if nv12.is_some() { (w * h * 3 / 2) as u64 } else { (w * h * 4) as u64 }));
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
                Cmd::View { cam, dof, ev100, dof_half } => {
                    core.set_view(cam, dof, ev100);
                    if dof_half.is_some() {
                        core.kf_dof_half = dof_half;
                    }
                }
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
                Cmd::Sharp { matched } => {
                    let n = core.set_sharp(matched);
                    eprintln!("live: sharp {matched:?}: {n} material(s){}", if kf_sharp_env() { "" } else { " (THREERT_KF_SHARP is not 1: drawn as before)" });
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
                        st.finish()?;
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
        let slot = match (&mut stream, emit_this) {
            (Some(st), true) => Some(st.acquire()?),
            _ => None,
        };
        let ring = stream.as_ref().zip(slot).map(|(st, i)| &st.ring[i]);
        core.render(spp, |e, out| {
            if pending {
                e.copy_buffer_to_buffer(out, 0, &staging, 0, (w * h * 4) as u64);
            }
            if let Some(r) = ring {
                match &nv12 {
                    Some(c) => {
                        c.record(e);
                        e.copy_buffer_to_buffer(&c.buf, 0, r, 0, r.size());
                    }
                    None => e.copy_buffer_to_buffer(out, 0, r, 0, r.size()),
                }
            }
        });
        if let (Some(st), Some(i)) = (&mut stream, slot) {
            st.queued(i)?;
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
        // known-future: lockstep takes the samples asked for (`spp`) at any render scale
        spp = if lockstep.is_some() { lock_spp } else if core.render_scale < 1.0 { 1 } else { next_spp(spp, core.gpu_ms, opts.fps) };
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
            Cmd::View { cam: Camera { pos: vec("pos")?, fwd, up: right.cross(fwd), right, tan_half_fov: v["tan"].as_f64().unwrap_or(0.0) as f32 }, dof, ev100: v["ev100"].as_f64().map(|e| e as f32), dof_half: v["dof_half"].as_bool() }
        }
        "emissive" => {
            let f3 = |k: &str| v[k].as_array().filter(|a| a.len() == 3).map(|a| [a[0].as_f64().unwrap_or(0.0) as f32, a[1].as_f64().unwrap_or(0.0) as f32, a[2].as_f64().unwrap_or(0.0) as f32]);
            Cmd::Emissive { matched: f3("match")?, rgb: f3("rgb")? }
        }
        "sharp" => {
            let a = v["match"].as_array().filter(|a| a.len() == 3)?;
            Cmd::Sharp { matched: [a[0].as_f64().unwrap_or(0.0) as f32, a[1].as_f64().unwrap_or(0.0) as f32, a[2].as_f64().unwrap_or(0.0) as f32] }
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
        stream_nv12: flag("--stream-format").is_some_and(|f| f == "nv12"),
    })
}

/// `--launch '{"room":"K7Q4MZ","name":"Ada"}'`: the `params` of the host channel's `launch` message.
pub fn parse_launch(v: Option<&str>) -> Result<Option<serde_json::Value>> {
    v.map(|t| serde_json::from_str::<serde_json::Value>(t).ok().filter(|p| p.is_object()).ok_or_else(|| anyhow!("--launch must be a JSON object, as in --launch '{{\"room\":\"K7Q4MZ\"}}'"))).transpose()
}

/// known-future `--stream`: frames read back through a ring of buffers and written to a FIFO by a thread of their own,
/// in order, straight from the mapped buffer (no copy on the render thread: at 4K the copy was ~9 ms a frame there).
/// The render thread takes a free buffer (`acquire`, waiting only when every one is still being written), the frame
/// copies its output into it, `queued` maps it and hands it to the writer, which waits for the mapping (polling the
/// device), writes, unmaps and gives the buffer back.
struct Stream {
    ring: Vec<wgpu::Buffer>,
    free: mpsc::Receiver<usize>,
    tx: Option<mpsc::SyncSender<(usize, Arc<std::sync::atomic::AtomicBool>)>>,
    writer: Option<std::thread::JoinHandle<std::io::Result<f64>>>,
    pub(super) wait_ms: f64,
}

/// Read-back buffers in the ring: frames in flight between the GPU and the FIFO's reader.
const STREAM_RING: usize = 4;

impl Stream {
    fn new(d: &wgpu::Device, path: PathBuf, size: u64) -> Stream {
        let ring: Vec<wgpu::Buffer> = (0..STREAM_RING).map(|_| d.create_buffer(&wgpu::BufferDescriptor { label: Some("stream"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })).collect();
        let (tx, rx) = mpsc::sync_channel::<(usize, Arc<std::sync::atomic::AtomicBool>)>(STREAM_RING);
        let (ftx, free) = mpsc::channel::<usize>();
        for i in 0..STREAM_RING {
            let _ = ftx.send(i);
        }
        let (dev, bufs) = (d.clone(), ring.clone());
        let writer = std::thread::spawn(move || -> std::io::Result<f64> {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(&path)?;
            let mut map_wait = 0.0;
            for (i, done) in rx {
                let t = Instant::now();
                while !done.load(Ordering::Acquire) {
                    let _ = dev.poll(wgpu::PollType::Poll);
                    if !done.load(Ordering::Acquire) {
                        std::thread::sleep(Duration::from_micros(100));
                    }
                }
                map_wait += t.elapsed().as_secs_f64() * 1e3;
                {
                    let view = bufs[i].slice(..).get_mapped_range().map_err(|e| std::io::Error::other(format!("stream map: {e:?}")))?;
                    f.write_all(&view)?;
                }
                bufs[i].unmap();
                let _ = ftx.send(i);
            }
            f.flush()?;
            Ok(map_wait)
        });
        Stream { ring, free, tx: Some(tx), writer: Some(writer), wait_ms: 0.0 }
    }

    /// A buffer free for this frame's output (waits while the writer has them all).
    fn acquire(&mut self) -> Result<usize> {
        let t = Instant::now();
        let i = self.free.recv().map_err(|_| anyhow!("stream: the writer stopped (reader gone?)"))?;
        self.wait_ms += t.elapsed().as_secs_f64() * 1e3;
        Ok(i)
    }

    /// The frame just submitted copied its output into `ring[i]`: map it and hand it to the writer.
    fn queued(&mut self, i: usize) -> Result<()> {
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = done.clone();
        self.ring[i].slice(..).map_async(wgpu::MapMode::Read, move |r| {
            if r.is_ok() {
                flag.store(true, Ordering::Release);
            }
        });
        if let Some(tx) = &self.tx {
            tx.send((i, done)).map_err(|_| anyhow!("stream: the writer stopped (reader gone?)"))?;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        drop(self.tx.take());
        let mut map_wait = 0.0;
        if let Some(w) = self.writer.take() {
            map_wait = w.join().map_err(|_| anyhow!("stream writer panicked"))??;
        }
        eprintln!("live: stream closed (render thread waited {:.1} ms for a free buffer; the writer {:.1} ms for mappings)", self.wait_ms, map_wait);
        Ok(())
    }
}

/// known-future `--stream-format nv12`: the frame's BGRA8 output converted to NV12 on the GPU, exactly as the
/// known-future compositor converts (compositor.py `to_nv12`: each 8-bit channel / 255 rounded to f16, BT.709 luma
/// 16 + 219 Y, chroma 128 + 224 (B - Y) / 1.8556 and 128 + 224 (R - Y) / 1.5748 averaged over 2 x 2, rounded half to
/// even), so the compositor hands the bytes to the encoder as they are. One invocation a 4 x 2 block of pixels.
struct Nv12 {
    pipe: wgpu::ComputePipeline,
    bg: wgpu::BindGroup,
    buf: wgpu::Buffer,
    groups: (u32, u32),
}

const NV12_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> dims: vec4<u32>;
fn h16(x: f32) -> f32 { return unpack2x16float(pack2x16float(vec2<f32>(x, 0.0))).x; }
fn ch(p: u32, sh: u32) -> f32 { return clamp(h16(f32((p >> sh) & 255u) / 255.0), 0.0, 1.0); }
fn byte(x: f32) -> u32 { return u32(clamp(round(x), 0.0, 255.0)); }
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = dims.x;
    let h = dims.y;
    if (gid.x * 4u >= w || gid.y * 2u >= h) { return; }
    let kr = 0.2126;
    let kb = 0.0722;
    let kg = 0.7152;
    var us: array<f32, 8>;
    var vs: array<f32, 8>;
    for (var j = 0u; j < 2u; j++) {
        let y = gid.y * 2u + j;
        var word = 0u;
        for (var i = 0u; i < 4u; i++) {
            let p = src[y * w + gid.x * 4u + i];
            let r = ch(p, 16u);
            let g = ch(p, 8u);
            let b = ch(p, 0u);
            let yl = kr * r + kg * g + kb * b;
            word |= byte(16.0 + 219.0 * yl) << (8u * i);
            us[j * 4u + i] = 128.0 + 224.0 * (b - yl) / 1.8556;
            vs[j * 4u + i] = 128.0 + 224.0 * (r - yl) / 1.5748;
        }
        dst[(y * w + gid.x * 4u) / 4u] = word;
    }
    var uv = 0u;
    for (var k = 0u; k < 2u; k++) {
        let a = k * 2u;
        let u = (us[a] + us[a + 1u] + us[4u + a] + us[4u + a + 1u]) / 4.0;
        let v = (vs[a] + vs[a + 1u] + vs[4u + a] + vs[4u + a + 1u]) / 4.0;
        uv |= (byte(u) | (byte(v) << 8u)) << (16u * k);
    }
    dst[(w * h + gid.y * w + gid.x * 4u) / 4u] = uv;
}
"#;

impl Nv12 {
    fn new(d: &wgpu::Device, out: &wgpu::Buffer, w: u32, h: u32) -> Nv12 {
        use wgpu::util::DeviceExt;
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("kf nv12"), source: wgpu::ShaderSource::Wgsl(NV12_WGSL.into()) });
        let pipe = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("kf nv12"), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let buf = d.create_buffer(&wgpu::BufferDescriptor { label: Some("kf nv12"), size: (w * h * 3 / 2) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let dims = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("kf nv12 dims"), contents: bytemuck::cast_slice(&[w, h, 0u32, 0u32]), usage: wgpu::BufferUsages::UNIFORM });
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("kf nv12"),
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
            ],
        });
        Nv12 { pipe, bg, buf, groups: ((w / 4).div_ceil(8), (h / 2).div_ceil(8)) }
    }

    fn record(&self, e: &mut wgpu::CommandEncoder) {
        let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("kf nv12"), timestamp_writes: None });
        pass.set_pipeline(&self.pipe);
        pass.set_bind_group(0, &self.bg, &[]);
        pass.dispatch_workgroups(self.groups.0, self.groups.1, 1);
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
