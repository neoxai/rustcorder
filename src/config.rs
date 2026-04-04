//! Interactive device configuration wizard (`--config`).
//!
//! Runs before the TUI is initialised — uses plain stdin/stdout.  Detects the
//! Deity VO-7U, lets the user pick a playback device, records a 3-second test
//! clip, plays it back, then writes `chosen.devices.txt`.

use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::mpsc::TryRecvError;
use std::time::Instant;

use anyhow::Result;

use crate::audio::{self, AudioEvent};
use crate::device::{self, ChosenDevices, DeviceInfo};
use crate::wav::WavWriter;

const TEST_FILE: &str = "_config_test.wav";
const TEST_DURATION_SECS: f64 = 3.0;

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn run_config() -> Result<()> {
    println!();
    println!("=== rustcorder — Device Configuration ===");
    println!();

    // Capture is always the Deity VO-7U.
    let mic = device::find_approved_mic()?;
    println!("Microphone: {}", mic.description);
    println!();

    loop {
        let playback = pick_playback_device()?;

        println!();
        println!("Selected:");
        println!("  Capture : {}", mic.description);
        print_device_summary("  Playback", &playback);

        let ok = run_test(&mic.alsa_name, &playback.alsa_name)?;

        if ok {
            device::save_chosen_devices(&ChosenDevices { playback })?;
            println!();
            println!("Saved to chosen.devices.txt.");
            println!();
            break;
        }

        println!();
        println!("Let's try again.");
        println!();
    }

    Ok(())
}

// ── Device picker ─────────────────────────────────────────────────────────────

fn pick_playback_device() -> Result<DeviceInfo> {
    let devices = device::enumerate_playback_devices();

    println!("Playback devices (speakers / headphones):");
    println!();
    print_device_list(&devices);

    loop {
        print!("Select playback device [1-{}]: ", devices.len());
        io::stdout().flush()?;
        let mut line = String::new();
        io::stdin().read_line(&mut line)?;
        if let Ok(n) = line.trim().parse::<usize>() {
            if n >= 1 && n <= devices.len() {
                return Ok(devices.into_iter().nth(n - 1).unwrap());
            }
        }
        println!("Please enter a number between 1 and {}.", devices.len());
    }
}

// ── Display helpers ───────────────────────────────────────────────────────────

fn print_device_list(devices: &[DeviceInfo]) {
    for (i, d) in devices.iter().enumerate() {
        if d.bus_type.is_empty() {
            println!("  [{}]  {}   {}", i + 1, d.description, d.alsa_name);
        } else {
            println!(
                "  [{}]  {}   [{}]   {}",
                i + 1,
                d.description,
                d.bus_type,
                d.alsa_name
            );
        }
        if let Some(ref det) = d.detail {
            println!("       {}", det);
        }
        println!();
    }
}

fn print_device_summary(label: &str, d: &DeviceInfo) {
    if d.bus_type.is_empty() {
        println!("{}: {}   {}", label, d.description, d.alsa_name);
    } else {
        println!(
            "{}: {}   [{}]   {}",
            label, d.description, d.bus_type, d.alsa_name
        );
    }
    if let Some(ref det) = d.detail {
        println!("           {}", det);
    }
}

// ── Test recording + playback ─────────────────────────────────────────────────

fn run_test(capture_alsa: &str, playback_alsa: &str) -> Result<bool> {
    let path = PathBuf::from(TEST_FILE);

    // ── Record ────────────────────────────────────────────────────────────
    println!();
    print!(
        "Recording {}s test clip — speak into the microphone",
        TEST_DURATION_SECS as u32
    );
    io::stdout().flush()?;

    let cap = audio::start_capture(capture_alsa)?;
    let mut wav = WavWriter::new(&path)?;

    let start = Instant::now();
    let mut next_tick_secs = 1u64;

    loop {
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed >= TEST_DURATION_SECS {
            break;
        }

        if start.elapsed().as_secs() >= next_tick_secs {
            print!(" {}.", TEST_DURATION_SECS as u64 - next_tick_secs + 1);
            io::stdout().flush()?;
            next_tick_secs += 1;
        }

        match cap.rx.try_recv() {
            Ok(AudioEvent::Samples { data, .. }) => {
                wav.write_s24le(&data)?;
            }
            Ok(AudioEvent::DeviceGone) => {
                cap.stop();
                let _ = std::fs::remove_file(&path);
                anyhow::bail!("Capture device disconnected during test recording.");
            }
            Ok(AudioEvent::Error(e)) => {
                cap.stop();
                let _ = std::fs::remove_file(&path);
                anyhow::bail!("Capture error: {}", e);
            }
            Ok(AudioEvent::Xrun) => {}
            Err(TryRecvError::Empty) => {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            Err(TryRecvError::Disconnected) => break,
        }
    }

    cap.stop();
    wav.finalize()?;
    println!(" done.");

    // ── Play back ─────────────────────────────────────────────────────────
    println!("Playing back through: {}", playback_alsa);

    let pb = audio::start_playback(path.clone(), 0.0, playback_alsa)?;
    loop {
        match pb.rx.recv() {
            Ok(audio::PlaybackEvent::Done) => break,
            Ok(audio::PlaybackEvent::Error(e)) => {
                pb.stop();
                let _ = std::fs::remove_file(&path);
                anyhow::bail!("Playback error: {}", e);
            }
            Err(_) => break,
        }
    }
    pb.stop();

    let _ = std::fs::remove_file(&path);

    // ── Confirm ───────────────────────────────────────────────────────────
    print!("Did you hear your voice clearly? [Y/n]: ");
    io::stdout().flush()?;
    let mut answer = String::new();
    io::stdin().read_line(&mut answer)?;
    let answer = answer.trim().to_lowercase();
    Ok(answer.is_empty() || answer == "y" || answer == "yes")
}
