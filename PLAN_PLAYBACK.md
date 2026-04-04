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
- [ ] Add `src/playback/` module directory
- [ ] Create `src/playback/mod.rs` with shared `Player` trait and spacebar input loop
- [ ] Add new crate dependencies to `Cargo.toml`
  - `rodio = "0.19"`
  - `cpal = "0.15"`
  - `libpulse-binding = "2.28"`
- [ ] Wire up `--playback`, `--type`, `--file` flags in `src/main.rs`

### Type 1 — `rodio` (high-level sink, cpal → ALSA backend)
- [ ] Create `src/playback/rodio_player.rs`
- [ ] Load WAV file via `rodio::Decoder`
- [ ] Use `rodio::Sink` for playback
- [ ] Implement pause/resume via `Sink::pause()` / `Sink::play()` (position tracked internally)
- [ ] Implement `Player` trait for `RodioPlayer`

### Type 2 — `cpal` (low-level callback with cursor)
- [ ] Create `src/playback/cpal_player.rs`
- [ ] Pre-load WAV samples into `Vec<i32>` using existing `src/wav.rs`
- [ ] Share buffer + cursor via `Arc<Mutex<>>` with audio callback
- [ ] Audio callback advances cursor and writes samples; pause = stop advancing cursor
- [ ] Resume = continue from saved cursor position
- [ ] Implement `Player` trait for `CpalPlayer`

### Type 3 — `libpulse-binding` (PulseAudio daemon stream)
- [ ] Create `src/playback/pulse_player.rs`
- [ ] Connect to PulseAudio daemon and create a playback stream
- [ ] Feed WAV samples into the stream
- [ ] Implement pause/resume via `Stream::cork()` / `Stream::uncork()`
- [ ] Implement `Player` trait for `PulsePlayer`

### Integration
- [ ] Dispatch to correct player in `src/main.rs` based on `--type` value
- [ ] Verify all three compile and play back the file cleanly
- [ ] Test spacebar pause/resume on each type

---

## Implementation Notes
- Use existing `src/wav.rs` for WAV header reading and sample conversion across all types
- `crossterm` (already a dep) handles raw terminal keypress input in the shared loop
- No changes to existing recording/app logic — this is fully additive
