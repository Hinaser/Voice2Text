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
