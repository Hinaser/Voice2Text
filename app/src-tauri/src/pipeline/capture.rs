//! WASAPI audio capture. Each source runs on its own thread and pushes 16 kHz
//! mono f32 chunks into a shared tagged channel.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;

use wasapi::{DeviceEnumerator, Direction, Role, SampleType, StreamMode, WaveFormat};

use super::{Source, RATE};

/// Capture one source until `running` clears. System = default render endpoint
/// opened for loopback; Mic = default capture endpoint. WASAPI autoconverts to
/// 16 kHz mono f32 for us.
pub fn run(source: Source, running: &AtomicBool, tx: &Sender<(Source, Vec<f32>)>) -> Result<(), Box<dyn std::error::Error>> {
    wasapi::initialize_mta().ok()?;
    let enumerator = DeviceEnumerator::new()?;
    let device = match source {
        Source::System => enumerator.get_default_device_for_role(&Direction::Render, &Role::Console)?,
        Source::Mic => enumerator.get_default_device_for_role(&Direction::Capture, &Role::Console)?,
    };
    let mut client = device.get_iaudioclient()?;
    let fmt = WaveFormat::new(32, 32, &SampleType::Float, RATE, 1, None);
    let (_def, min_time) = client.get_device_period()?;
    let mode = StreamMode::EventsShared { autoconvert: true, buffer_duration_hns: min_time };
    client.initialize_client(&fmt, &Direction::Capture, &mode)?;
    let h_event = client.set_get_eventhandle()?;
    let capture = client.get_audiocaptureclient()?;

    let mut bytes: VecDeque<u8> = VecDeque::new();
    client.start_stream()?;
    while running.load(Ordering::SeqCst) {
        capture.read_from_device_to_deque(&mut bytes)?;
        if bytes.len() >= 4 {
            let mut out = Vec::with_capacity(bytes.len() / 4);
            while bytes.len() >= 4 {
                let b = [bytes.pop_front().unwrap(), bytes.pop_front().unwrap(), bytes.pop_front().unwrap(), bytes.pop_front().unwrap()];
                out.push(f32::from_le_bytes(b));
            }
            if tx.send((source, out)).is_err() {
                break;
            }
        }
        let _ = h_event.wait_for_event(200);
    }
    client.stop_stream()?;
    Ok(())
}
