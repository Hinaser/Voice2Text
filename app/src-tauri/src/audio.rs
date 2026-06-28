//! Audio endpoint enumeration and resolution, shared by the capture pipeline
//! (which opens a chosen device) and the `list_audio_devices` command (which
//! populates the settings dropdowns). Only active endpoints are returned.

use wasapi::{Device, DeviceEnumerator, Direction, Role};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// `(id, friendly_name)` for one endpoint.
pub type DeviceInfo = (String, String);

/// List active render (system output) and capture (microphone) endpoints.
pub fn list_devices() -> Res<(Vec<DeviceInfo>, Vec<DeviceInfo>)> {
    // Try to enter MTA, but tolerate "already initialized" (RPC_E_CHANGED_MODE):
    // this is called from the Tauri command thread, which the webview has already
    // put into an STA. COM just needs to be initialized in *some* mode for the
    // (apartment-agnostic) device enumerator to work.
    let _ = wasapi::initialize_mta().ok();
    let en = DeviceEnumerator::new()?;
    Ok((collect(&en, &Direction::Render)?, collect(&en, &Direction::Capture)?))
}

fn collect(en: &DeviceEnumerator, direction: &Direction) -> Res<Vec<DeviceInfo>> {
    let coll = en.get_device_collection(direction)?;
    let mut out = Vec::new();
    for i in 0..coll.get_nbr_devices()? {
        if let Ok(d) = coll.get_device_at_index(i) {
            let id = d.get_id().unwrap_or_default();
            if id.is_empty() {
                continue;
            }
            let name = d.get_friendlyname().unwrap_or_else(|_| "(unknown device)".into());
            out.push((id, name));
        }
    }
    Ok(out)
}

/// Friendly names of the endpoints that will actually be captured, paired with a
/// display role, for the status bar. Mirrors how the capture threads resolve
/// devices, so the names shown match what's recorded. Best-effort: returns
/// whatever it can resolve (empty on enumerator failure).
pub fn capture_source_names(output_id: &str, input_id: &str, mic: bool) -> Vec<(&'static str, String)> {
    let mut out = Vec::new();
    let _ = wasapi::initialize_mta().ok();
    let en = match DeviceEnumerator::new() {
        Ok(e) => e,
        Err(_) => return out,
    };
    let name = |d: &Device| d.get_friendlyname().unwrap_or_else(|_| "default device".into());
    if let Ok(d) = resolve_device(&en, &Direction::Render, output_id) {
        out.push(("System audio", name(&d)));
    }
    if mic {
        if let Ok(d) = resolve_device(&en, &Direction::Capture, input_id) {
            out.push(("Microphone", name(&d)));
        }
    }
    out
}

/// Resolve the endpoint to capture from: the configured `id` if set and still
/// present, otherwise the default device for the Console role.
pub fn resolve_device(en: &DeviceEnumerator, direction: &Direction, id: &str) -> Res<Device> {
    let id = id.trim();
    if !id.is_empty() {
        if let Ok(d) = en.get_device(id) {
            return Ok(d);
        }
        // Configured device is gone (unplugged / changed) — fall back silently.
    }
    Ok(en.get_default_device_for_role(direction, &Role::Console)?)
}
