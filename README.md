# rustcorder

A distraction-free terminal recorder for audiobook narration, built in Rust. Designed for USB broadcast microphones, lossless capture, and a punch-and-roll editing workflow.

---

## Requirements

- Ubuntu (tested on Ubuntu 22.04+)
- Rust toolchain (`cargo`)
- ALSA development headers: `sudo apt install libasound2-dev`
- A Deity VO-7U USB microphone (or run with `--unsafe` for any capture device)

---

## Build

```sh
cargo build --release
```

The binary is at `target/release/rustcorder`.

---

## Usage

```sh
# Normal mode — requires the Deity VO-7U
./rustcorder

# Unsafe mode — accepts any ALSA capture device (for testing)
./rustcorder --unsafe

# Device configuration wizard — select microphone and playback device, run a
# test recording, then save your choices to chosen.devices.txt
./rustcorder --config
```

Run from the directory where you want book folders created. Session state and the `.env` config file are read from the current working directory.

---

## Device Configuration (`--config`)

Run `./rustcorder --config` once to set up your preferred microphone and playback device. The wizard:

1. Lists every ALSA capture (microphone) device found on the machine.
2. Lists every ALSA playback device, plus the system default.
3. Records a 3-second test clip through the selected microphone.
4. Plays it back through the selected output device.
5. Saves the choices to `chosen.devices.txt` in the current directory.

When `--unsafe` is started and `chosen.devices.txt` is present, those devices are used automatically — no auto-detection required.

### `chosen.devices.txt` format

```
CAPTURE_DEVICE=hw:1,0
CAPTURE_NAME=Deity VO-7U (VO7U)
PLAYBACK_DEVICE=default
PLAYBACK_NAME=System default (PipeWire / PulseAudio)
```

---

## Configuration

Copy `.env` into the directory you run the binary from (it is already present in the repository root). Variables set in the shell environment take precedence over the file.

| Variable | Default | Description |
|---|---|---|
| `PUNCH_BACK_TIME` | `15` | Seconds to rewind before rollback playback during punch-and-roll. Minimum: 1. |

---

## Keyboard Reference

| Mode | Key | Action |
|---|---|---|
| Ready | `Space` / `Enter` | Begin signal check, then start recording |
| Ready | `E` | Edit book / chapter |
| Ready | `R` | Re-detect microphone |
| Ready | `Q` / `Esc` | Quit |
| Pre-check | `Esc` | Abort signal check |
| Recording | `Space` / `Enter` | Stop recording |
| Recording | `P` | **Punch and roll** — stop, rewind, play back |
| Rollback | `Space` / `Enter` | **Punch in** — mark this moment as the new start |
| Rollback | `Esc` | Cancel punch, return to post-recording prompt |
| Post-recording | `Y` / `Enter` | Continue chapter (increment part) |
| Post-recording | `N` | Chapter complete (advance chapter, reset part) |
| Mic error | `R` / `Enter` | Retry microphone detection |
| Any | `Ctrl-C` | Emergency stop — finalizes current file and exits |

---

## File Layout

```
$PWD/
├── .env                          # Local configuration
├── session.save.txt              # Persisted session state
└── BookName/
    ├── Chapter_01_part001.wav
    ├── Chapter_01_part002.wav
    ├── Chapter_01_timeline.txt   # Timeline index for Chapter 01
    ├── Chapter_02_part001.wav
    └── Chapter_02_timeline.txt
```

### Audio format

All recordings are written as:

- Container: WAV (PCM)
- Bit depth: 24-bit (S24\_LE)
- Sample rate: 48,000 Hz
- Channels: Mono

### `session.save.txt`

Persisted across launches. Human-readable, line-based:

```
BOOK=The_Hobbit
CHAPTER=02
PART=3
TIMELINE_POS=254.891
```

`TIMELINE_POS` is the absolute position in seconds from the start of the current chapter where the next clip will begin.

---

## Punch-and-Roll

Punch-and-roll is a professional narration technique for correcting mistakes mid-take without stopping to edit.

### How it works

1. While **Recording**, press **`P`**
2. The current clip is finalized immediately
3. Playback rewinds `PUNCH_BACK_TIME` seconds from the end and plays through your speakers
4. When you hear the point you want to re-record from, press **`Space`**
5. Playback stops; the elapsed time is used to compute the exact punch-in position
6. A signal check runs, then recording begins as a new part file, starting at the punch-in timestamp in the timeline

The old clip file is never modified. The timeline index records the punch-in point so that the new clip supersedes the old one from that moment forward.

### The timeline index

Each chapter has a `Chapter_NN_timeline.txt` file alongside the WAV files:

```
# rustcorder timeline — The_Hobbit / Chapter 01
CLIP Chapter_01_part001.wav START=0.000
CLIP Chapter_01_part002.wav START=127.342
```

**Reading rule:** each clip owns audio from its `START` until the `START` of the next clip (or the clip's own end, whichever is earlier). If a punch-in clip starts before the previous clip ends, the overlap belongs to the new clip.

**Example — one punch:**
```
CLIP Chapter_01_part001.wav START=0.000    # 142 s long
CLIP Chapter_01_part002.wav START=127.342  # takes over at 127.342 s
```
`part001` contributes audio from 0 → 127.342 s. The remainder of `part001` is discarded in assembly. `part002` contributes everything from its own beginning onward.

Entries are appended at the **start** of each recording, before any audio is written, so a crash mid-take still leaves a valid (if short) entry.

---

## Microphone Enforcement

In normal mode, rustcorder will only record from the **Deity VO-7U** (USB VID `0x19f7`, PID `0x003c`). It:

- Enumerates all ALSA capture devices via `/sys/class/sound`
- Resolves each device's USB vendor/product ID from sysfs
- Requires exactly one matching approved device — zero or two or more is a blocking error
- Continuously monitors for USB disconnection during recording; if the mic is unplugged, the current file is finalized cleanly and recording stops
- Never falls back to any other audio device

Use `--unsafe` to bypass this check (e.g. for testing with a built-in microphone). A yellow warning banner is displayed throughout the session when unsafe mode is active.

---

## Pre-Recording Signal Check

Before each recording (including after a punch-in), rustcorder captures 2 seconds of audio and checks whether any signal above −60 dBFS is detected. If the microphone is hardware-muted or silent, recording does not begin and a warning is shown. This check runs every time, with no bypass.

## Runtime Silence Watchdog

During recording, if no audio above −60 dBFS is detected for 15 consecutive seconds, a flashing warning banner is displayed. Recording continues; the warning clears when signal resumes.

---

## Architecture

| File | Responsibility |
|---|---|
| `src/main.rs` | Entry point — terminal setup, event loop, signal handling |
| `src/app.rs` | Application state machine and all business logic |
| `src/audio.rs` | ALSA capture thread, ALSA playback thread, RMS computation |
| `src/device.rs` | USB microphone discovery and enforcement via sysfs |
| `src/session.rs` | Session persistence, file naming, timeline index management |
| `src/wav.rs` | WAV file writer (fixed 48kHz/24-bit/mono format) |
| `src/render.rs` | Terminal UI rendering (ratatui) |

### State machine

```
Setup ──────────────────────────────────────────────────────────────────────────→ Ready
                                                                                     │
                                                                  ┌──────────────────┘
                                                                  │
                                                                  ▼
                                                               PreCheck ──── (no signal) ──→ Ready
                                                                  │
                                                            (signal found)
                                                                  │
                                                                  ▼
                                               ┌─────────────── Recording ──────────────────────────┐
                                               │  Space/Enter        │                              │
                                               │                     │ P                             │ mic unplugged
                                               ▼                     ▼                              ▼
                                        PostRecording        PunchRollback ──── Esc ──→ PostRecording
                                         │        │                  │
                                       Y/Enter    N              Space/Enter
                                         │        │                  │
                                         ▼        ▼                  ▼
                                        Ready   Ready           PunchReady ──→ PreCheck ──→ Recording
```

### Dependencies

| Crate | Purpose |
|---|---|
| `alsa` | Direct ALSA bindings for capture and playback |
| `ratatui` | Terminal UI rendering |
| `crossterm` | Cross-platform terminal control |
| `signal-hook` | SIGTERM handling for clean shutdown |
| `dotenvy` | `.env` file loading |
| `anyhow` | Error propagation |

---

## Playback Device

During punch-and-roll rollback, audio is played to the ALSA `default` PCM device. On Ubuntu with PipeWire or PulseAudio, this routes to the system's active output (typically your speakers or headphones). No output device configuration is available in this version; that is planned for a future release.
