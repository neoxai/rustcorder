pub mod cpal_player;
pub mod pulse_player;
pub mod rodio_player;

use std::io::Write;
use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};

/// Common interface every player implementation must satisfy.
pub trait Player {
    /// Toggle between playing and paused, resuming from the same position.
    fn play_pause(&mut self);
    /// Stop playback immediately.
    fn stop(&mut self);
    /// True once the file has played to the end naturally.
    fn is_finished(&self) -> bool;
    /// True when currently paused (not playing, not stopped).
    fn is_paused(&self) -> bool;
}

/// Entry point called from main.  Dispatches to the correct player type.
pub fn run(player_type: u8, file: &Path) -> Result<()> {
    match player_type {
        1 => run_loop(&mut rodio_player::RodioPlayer::new(file)?),
        2 => run_loop(&mut cpal_player::CpalPlayer::new(file)?),
        3 => run_loop(&mut pulse_player::PulsePlayer::new(file)?),
        _ => anyhow::bail!("--type must be 1, 2, or 3"),
    }
}

/// Enable raw mode, run the spacebar input loop, then restore the terminal.
fn run_loop(player: &mut impl Player) -> Result<()> {
    enable_raw_mode()?;

    // In raw mode \n doesn't CR, so use \r\n explicitly.
    print!("Playing  |  SPACE = pause/resume   Q = quit\r\n");
    print!("[playing]\r\n");
    std::io::stdout().flush()?;

    let result = input_loop(player);

    // Always restore terminal even if the loop errored.
    let _ = disable_raw_mode();
    println!();
    result
}

fn input_loop(player: &mut impl Player) -> Result<()> {
    loop {
        if player.is_finished() {
            print!("\r[done]      \r\n");
            std::io::stdout().flush()?;
            break;
        }

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char(' ') => {
                        player.play_pause();
                        let label = if player.is_paused() { "[paused] " } else { "[playing]" };
                        print!("\r{label}   ");
                        std::io::stdout().flush()?;
                    }
                    KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
                        player.stop();
                        print!("\r[stopped]   \r\n");
                        std::io::stdout().flush()?;
                        break;
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

// ── Shared WAV utilities ──────────────────────────────────────────────────────

const WAV_HEADER_BYTES: usize = 44;
const BYTES_PER_SAMPLE: usize = 3; // 24-bit PCM

/// Read the raw audio bytes from a WAV file (skips the 44-byte header).
pub fn read_wav_bytes(path: &Path) -> Result<Vec<u8>> {
    let raw = std::fs::read(path)?;
    anyhow::ensure!(
        raw.len() > WAV_HEADER_BYTES,
        "file too small to be a valid WAV: {}",
        path.display()
    );
    Ok(raw[WAV_HEADER_BYTES..].to_vec())
}

/// Convert raw 24-bit LE WAV bytes to normalised f32 samples in [-1.0, 1.0].
pub fn wav_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(BYTES_PER_SAMPLE)
        .map(|c| {
            let raw = (c[0] as u32) | ((c[1] as u32) << 8) | ((c[2] as u32) << 16);
            // Sign-extend from bit 23 into a full i32.
            let s24 = (raw << 8) as i32 >> 8;
            s24 as f32 / 8_388_608.0
        })
        .collect()
}

/// Convert raw 24-bit LE WAV bytes to S32LE bytes (4 bytes per sample).
/// Matches the existing ALSA code: each 24-bit sample is left-shifted 8 bits.
pub fn wav_bytes_to_s32le(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity((bytes.len() / BYTES_PER_SAMPLE) * 4);
    for c in bytes.chunks_exact(BYTES_PER_SAMPLE) {
        // S32LE layout (little-endian): [0x00, b0, b1, b2]
        out.push(0x00);  // bits  0-7  (zero — shifted in)
        out.push(c[0]);  // bits  8-15
        out.push(c[1]);  // bits 16-23
        out.push(c[2]);  // bits 24-31 (sign bit preserved)
    }
    out
}
