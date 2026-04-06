//! Reconstruct a chapter's audio from its timeline index and clip WAV files.
//!
//! # Algorithm
//!
//! 1. **Parse** `Chapter_NN_timeline.txt` into ordered `(filename, start_secs)` entries.
//! 2. **Resolve** each clip's effective contribution using "last recorded wins":
//!    each clip contributes from its own start up to where the next clip takes
//!    over.  Clips completely superseded (zero contribution) or below the
//!    `DISCARD_SHORT_CLIPS` threshold are skipped.
//! 3. **Write** trimmed copies of each contributing region to `<book>/temp/`
//!    (useful for debugging; the directory is cleared before each export).
//! 4. **Crossfade** adjacent clips: the last `CROSSFADE_TIME` ms of clip A
//!    linearly ramps to silence while the first `CROSSFADE_TIME` ms of clip B
//!    ramps up from silence; the two windows are mixed (summed) so there is
//!    never a moment of absolute silence at a boundary.
//! 5. **Write** the final mixed audio to `<book>/out/Chapter_NN.wav`.

use std::fs;
use std::io::{Read as _, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::wav::WavWriter;

// ── Fixed WAV parameters (must match WavWriter / capture pipeline) ────────────

const SAMPLE_RATE: u32 = 48_000;
const BYTES_PER_SAMPLE: u64 = 3; // 24-bit mono
/// Byte offset of the `data` chunk size field written by WavWriter.
const WAV_DATA_SIZE_OFFSET: u64 = 40;
/// Byte offset where PCM sample data begins in files written by WavWriter.
const WAV_HEADER_SIZE: u64 = 44;

// ── Timeline parsing ──────────────────────────────────────────────────────────

struct TimelineEntry {
    filename: String,
    /// Absolute timeline position (seconds from chapter start) where this clip
    /// was recorded.
    start: f64,
}

/// Parse a `Chapter_NN_timeline.txt` file into a list of entries in file order
/// (= recording order, oldest first).
fn parse_timeline(path: &Path) -> Result<Vec<TimelineEntry>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading timeline {}", path.display()))?;

    let mut entries = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if !line.starts_with("CLIP ") {
            continue;
        }
        // Format: CLIP <filename> START=<f64>
        let rest = &line["CLIP ".len()..];
        if let Some(sep) = rest.rfind(" START=") {
            let filename = rest[..sep].trim().to_string();
            let start_str = &rest[sep + " START=".len()..];
            if let Ok(start) = start_str.trim().parse::<f64>() {
                entries.push(TimelineEntry { filename, start });
            }
        }
    }

    Ok(entries)
}

// ── WAV sample I/O ────────────────────────────────────────────────────────────

/// Read the PCM sample count from a WAV file written by `WavWriter`.
///
/// Reads the `data` chunk size field at the known fixed offset and divides by
/// the bytes-per-sample constant.  This avoids a full header parse because the
/// format is always the same (48 kHz, mono, 24-bit PCM).
fn wav_sample_count(path: &Path) -> Result<u64> {
    let mut f = fs::File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    f.seek(SeekFrom::Start(WAV_DATA_SIZE_OFFSET))?;
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    let data_bytes = u32::from_le_bytes(buf) as u64;
    Ok(data_bytes / BYTES_PER_SAMPLE)
}

/// Read `count` samples starting at sample index `offset` from a WAV file.
///
/// Returns i32 values sign-extended to the 24-bit signed range
/// `[-8_388_608, 8_388_607]` — matching the values produced by the capture
/// pipeline and accepted by `WavWriter::write_s24le`.
fn read_wav_samples(path: &Path, offset: u64, count: u64) -> Result<Vec<i32>> {
    let mut f = fs::File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;

    let byte_offset = WAV_HEADER_SIZE + offset * BYTES_PER_SAMPLE;
    f.seek(SeekFrom::Start(byte_offset))?;

    let byte_count = (count * BYTES_PER_SAMPLE) as usize;
    let mut buf = vec![0u8; byte_count];
    f.read_exact(&mut buf)
        .with_context(|| format!("reading {} samples from {}", count, path.display()))?;

    let samples = buf
        .chunks_exact(3)
        .map(|c| {
            // Reassemble little-endian 24-bit value and sign-extend to i32.
            // Using u32 arithmetic to avoid overflow in the shift.
            let raw = u32::from_le_bytes([c[0], c[1], c[2], 0]);
            ((raw << 8) as i32) >> 8
        })
        .collect();

    Ok(samples)
}

// ── Clip resolution ───────────────────────────────────────────────────────────

struct ResolvedClip {
    path: PathBuf,
    /// Sample index within the file at which the contributing region begins.
    /// Always 0 in the current model (each clip contributes from its own start).
    sample_offset: u64,
    /// Number of samples to read.
    sample_count: u64,
    /// Stem used when naming temp WAV files.
    label: String,
}

/// Build the ordered list of clip regions that make up the final chapter audio.
///
/// Applies "last recorded wins": each clip contributes from its own start up to
/// where the next clip (which was recorded later) takes over.  Clips with zero
/// effective contribution are dropped.  Any gaps between consecutive clips
/// (where one clip ends before the next begins) are closed by pushing clips
/// together — room-tone silence is intentional and will have been captured.
///
/// `discard_short` / `discard_secs` mirror the `DISCARD_SHORT_CLIPS` /
/// `DISCARD_DURATION` env vars: clips whose total WAV duration is below the
/// threshold are skipped entirely regardless of their effective contribution.
fn resolve_clips(
    entries: &[TimelineEntry],
    book_dir: &Path,
    discard_short: bool,
    discard_secs: f64,
) -> Result<Vec<ResolvedClip>> {
    let mut resolved = Vec::new();

    for (i, entry) in entries.iter().enumerate() {
        let path = book_dir.join(&entry.filename);

        let total_samples = wav_sample_count(&path)
            .with_context(|| format!("reading header of {}", entry.filename))?;
        let total_secs = total_samples as f64 / SAMPLE_RATE as f64;

        // Skip clips that fall below the discard threshold.
        if discard_short && total_secs < discard_secs {
            continue;
        }

        // Effective end of this clip's contribution: the later clip takes over
        // at its start time, so anything after that is superseded.
        let clip_end_secs = entry.start + total_secs;
        let effective_end = if i + 1 < entries.len() {
            clip_end_secs.min(entries[i + 1].start)
        } else {
            clip_end_secs
        };

        let contribution_secs = effective_end - entry.start;
        if contribution_secs <= 0.0 {
            continue; // completely superseded
        }

        let sample_count = (contribution_secs * SAMPLE_RATE as f64).round() as u64;
        if sample_count == 0 {
            continue;
        }

        let label = entry
            .filename
            .trim_end_matches(".wav")
            .to_string();

        resolved.push(ResolvedClip {
            path,
            sample_offset: 0,
            sample_count,
            label,
        });
    }

    Ok(resolved)
}

// ── Crossfade mixing ──────────────────────────────────────────────────────────

/// Concatenate clip sample buffers with a linear crossfade at each boundary.
///
/// At the transition between clip A and clip B:
/// - The last `cf` samples of A ramp from 1.0 → 0.0 (fade out).
/// - The first `cf` samples of B ramp from 0.0 → 1.0 (fade in).
/// - Both windows are mixed (summed) into `cf` output samples — the two fades
///   play simultaneously, so there is no silent point at any boundary.
///
/// The total output length is `sum(clip lengths) − (N−1) × cf`.
///
/// If a clip is shorter than `cf`, the crossfade window is reduced to the clip
/// length.  `DISCARD_SHORT_CLIPS` prevents this from occurring in normal use.
fn crossfade_concat(clips: &[Vec<i32>], cf_samples: usize) -> Vec<i32> {
    if clips.is_empty() {
        return Vec::new();
    }
    if clips.len() == 1 {
        return clips[0].clone();
    }

    let n = clips.len();

    // Actual crossfade window at each clip boundary (may be shorter than
    // cf_samples if one of the adjacent clips is very short).
    let cf_lens: Vec<usize> = (0..n - 1)
        .map(|i| cf_samples.min(clips[i].len()).min(clips[i + 1].len()))
        .collect();

    let mut result = Vec::new();

    for i in 0..n {
        let clip = &clips[i];

        // How many samples at the start were already consumed by the previous
        // boundary's fade-in mix (written during the previous iteration).
        let fade_in_consumed = if i == 0 { 0 } else { cf_lens[i - 1] };

        // How many samples at the end will be consumed by the next boundary's
        // fade-out mix (written at the end of this iteration).
        let fade_out_len = if i == n - 1 { 0 } else { cf_lens[i] };

        // Write the "body" — the region between the two fade windows.
        let body_start = fade_in_consumed;
        let body_end = clip.len().saturating_sub(fade_out_len);
        if body_start < body_end {
            result.extend_from_slice(&clip[body_start..body_end]);
        }

        // Write the crossfade mix at the end of this clip into the start of
        // the next clip.
        if i < n - 1 {
            let cf = cf_lens[i];
            let next = &clips[i + 1];
            let a_start = clip.len().saturating_sub(cf);

            for j in 0..cf {
                // t goes from 0.0 (fully A) to approaching 1.0 (fully B).
                let t = j as f64 / cf as f64;
                let a = clip[a_start + j] as f64;
                let b = next[j] as f64;
                let mixed = a * (1.0 - t) + b * t;
                result.push(mixed.round().clamp(-8_388_608.0, 8_388_607.0) as i32);
            }
        }
    }

    result
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Export a single chapter to a combined WAV file.
///
/// Reads `DISCARD_SHORT_CLIPS` and `DISCARD_DURATION` from the process
/// environment (already loaded from `.env` by `main`).
///
/// Writes per-clip trimmed WAVs to `<book_dir>/temp/` (cleared first) for
/// debugging, then the final mixed file to `<book_dir>/out/Chapter_NN.wav`.
///
/// Returns the path of the written output file.
pub fn export_chapter(book_dir: &Path, chapter: u32, crossfade_ms: f64) -> Result<PathBuf> {
    let timeline_path = book_dir.join(format!("Chapter_{:02}_timeline.txt", chapter));

    // DISCARD settings from environment.
    let discard_short = std::env::var("DISCARD_SHORT_CLIPS")
        .map(|v| v.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let discard_secs = std::env::var("DISCARD_DURATION")
        .ok()
        .and_then(|v| v.trim().trim_end_matches('s').parse::<f64>().ok())
        .unwrap_or(1.0)
        .max(0.0);

    // 1. Parse and resolve.
    let entries = parse_timeline(&timeline_path)?;
    let clips = resolve_clips(&entries, book_dir, discard_short, discard_secs)?;

    if clips.is_empty() {
        anyhow::bail!(
            "No clips to export for chapter {:02} (all superseded or discarded).",
            chapter
        );
    }

    // 2. Clear the temp directory.
    let temp_dir = book_dir.join("temp");
    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir)
            .with_context(|| format!("clearing temp dir {}", temp_dir.display()))?;
    }
    fs::create_dir_all(&temp_dir)?;

    let cf_samples = ((crossfade_ms / 1000.0) * SAMPLE_RATE as f64).round() as usize;

    // 3. Read each clip region and write a trimmed temp WAV.
    let mut all_samples: Vec<Vec<i32>> = Vec::with_capacity(clips.len());

    for (idx, clip) in clips.iter().enumerate() {
        let samples =
            read_wav_samples(&clip.path, clip.sample_offset, clip.sample_count)
                .with_context(|| format!("reading {}", clip.path.display()))?;

        // Write trimmed clip to temp for debugging.
        let temp_name = format!("clip_{:03}_{}.wav", idx + 1, clip.label);
        let temp_path = temp_dir.join(&temp_name);
        let mut w = WavWriter::new(&temp_path)
            .with_context(|| format!("creating temp file {}", temp_path.display()))?;
        w.write_s24le(&samples)?;
        w.finalize()?;

        all_samples.push(samples);
    }

    // 4. Crossfade and concatenate.
    let final_samples = crossfade_concat(&all_samples, cf_samples);

    // 5. Write output.
    let out_dir = book_dir.join("out");
    fs::create_dir_all(&out_dir)?;
    let out_path = out_dir.join(format!("Chapter_{:02}.wav", chapter));

    let mut writer = WavWriter::new(&out_path)
        .with_context(|| format!("creating output file {}", out_path.display()))?;
    writer.write_s24le(&final_samples)?;
    writer.finalize()?;

    Ok(out_path)
}
