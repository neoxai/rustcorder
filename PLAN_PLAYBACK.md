# Playback Experiment Plan

Add a `--playback` subcommand to test three different audio playback implementations,
so the best one can be chosen to replace the current ALSA-based playback.

## CLI
```
rustcorder --playback --type <1|2|3> --file <path/to/file.wav>
```
- Spacebar pauses/resumes at the same position
- Terminal only (no TUI)
- Prints `[playing]` / `[paused]` to stdout

---

## TODOs

### Setup
- [x] Add `src/playback/` module directory
- [x] Create `src/playback/mod.rs` with shared `Player` trait and spacebar input loop
- [x] Add new crate dependencies to `Cargo.toml`
  - `rodio = "0.19"`
  - `cpal = "0.15"`
  - `libpulse-binding = "2"`
  - `libpulse-simple-binding = "2"`
- [x] Wire up `--playback`, `--type`, `--file` flags in `src/main.rs`

### Type 1 — `rodio` (high-level sink, cpal → ALSA backend)
- [x] Create `src/playback/rodio_player.rs`
- [x] Load WAV file via `rodio::Decoder`
- [x] Use `rodio::Sink` for playback
- [x] Implement pause/resume via `Sink::pause()` / `Sink::play()` (position tracked internally)
- [x] Implement `Player` trait for `RodioPlayer`

### Type 2 — `cpal` (low-level callback with cursor)
- [x] Create `src/playback/cpal_player.rs`
- [x] Pre-load WAV samples into `Vec<f32>` using shared `wav_bytes_to_f32()`
- [x] Share buffer + cursor via `Arc<Mutex<>>` with audio callback
- [x] Audio callback advances cursor and writes samples; pause = stop advancing cursor
- [x] Resume = continue from saved cursor position
- [x] Implement `Player` trait for `CpalPlayer`

### Type 3 — `libpulse-simple-binding` (PulseAudio daemon stream)
- [x] Create `src/playback/pulse_player.rs`
- [x] Pre-load WAV, convert to S32LE bytes (`wav_bytes_to_s32le()`)
- [x] Background thread feeds chunks to `pa_simple` stream
- [x] Pause = set flag + `Simple::flush()` to drain PA buffer promptly
- [x] Resume = clear flag, continue writing from saved cursor
- [x] Implement `Player` trait for `PulsePlayer`

### Integration
- [x] Dispatch to correct player in `src/main.rs` based on `--type` value
- [x] Verify all three compile cleanly (`cargo build` — clean)
- [ ] Test spacebar pause/resume on each type with a real .wav file

---

## Implementation Notes
- `src/playback/mod.rs` contains `read_wav_bytes()`, `wav_bytes_to_f32()`, and
  `wav_bytes_to_s32le()` — shared utilities used by all three players
- `crossterm` (already a dep) handles raw terminal keypress input in the shared loop
- No changes to existing recording/app logic — fully additive
- System dep for Type 3: `sudo apt install libpulse-dev`
