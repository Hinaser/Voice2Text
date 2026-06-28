//! Filesystem resolution shared across the app: where the models live and where
//! the GPU sidecar executables are.

use std::path::PathBuf;

/// Resolve the models directory: explicit override, else `$VOICE2TEXT_MODELS`,
/// else the repo `models/` (dev), else a `models/` next to the executable
/// (shipped).
pub fn models_dir(override_dir: &str) -> PathBuf {
    if !override_dir.trim().is_empty() {
        return PathBuf::from(override_dir.trim());
    }
    if let Ok(d) = std::env::var("VOICE2TEXT_MODELS") {
        return PathBuf::from(d);
    }
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("models");
    if dev.exists() {
        return dev;
    }
    exe_dir().map(|d| d.join("models")).unwrap_or(dev)
}

/// Resolve a sidecar executable by file name. In both the dev workspace build
/// and a shipped bundle the sidecars sit next to the main exe, so "beside the
/// exe" is the common case; `$VOICE2TEXT_SIDECAR_DIR` overrides it (e.g. tests).
pub fn sidecar_exe(file_name: &str) -> PathBuf {
    if let Ok(dir) = std::env::var("VOICE2TEXT_SIDECAR_DIR") {
        return PathBuf::from(dir).join(file_name);
    }
    if let Some(beside) = exe_dir().map(|d| d.join(file_name)) {
        if beside.exists() {
            return beside;
        }
    }
    // Last resort: the workspace's shared target dir (covers odd dev launches).
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..").join("target").join("release").join(file_name)
}

fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe().ok().and_then(|p| p.parent().map(PathBuf::from))
}

/// In release builds, spawn a child with `CREATE_NO_WINDOW` so a console-subsystem
/// sidecar launched from the (windowed) app doesn't flash its own console window.
/// In debug builds the app keeps its console and the sidecar inherits it, so logs
/// stay visible — leave the flag off.
pub fn hide_console(cmd: &mut std::process::Command) {
    #[cfg(not(debug_assertions))]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    // Referenced in release via the cfg block; silence the unused warning in debug.
    let _ = cmd;
}
