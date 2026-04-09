# rustcorder

A distraction-free terminal recorder for audiobook narration, built in Rust. Designed for USB broadcast microphones, lossless capture, and a punch-and-roll editing workflow. Includes a browser-based EPUB reader that stays in sync with the recorder in real time.

---

## Requirements

- Ubuntu (tested on Ubuntu 22.04+)
- Rust toolchain (`cargo`)
- ALSA development headers: `sudo apt install libasound2-dev`
- PulseAudio development headers: `sudo apt install libpulse-dev`
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

# Device configuration wizard
./rustcorder --config

# Export a chapter to a single WAV file
./rustcorder --export --book <BookName> --chapter <N>

# Playback experiment (three backends)
./rustcorder --playback --type <1|2|3> --file <path/to/file.wav>
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
| `CROSSFADE_TIME` | `10ms` | Crossfade duration applied between clips during `--export`. |
| `BROWSER_PORT` | `7474` | Port for the local web server. Searches up to 10 ports if the default is in use. |
| `BROWSER_OPEN` | _(unset)_ | Set to `true` to auto-open the browser on startup via `xdg-open`. |

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
├── chosen.devices.txt            # Saved mic/playback device selection
└── BookName/
    ├── BookName.epub             # Optional — auto-discovered for the browser EPUB viewer
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

Entries are appended at the **start** of each recording, before any audio is written, so a crash mid-take still leaves a valid (if short) entry.

---

## Chapter Export (`--export`)

The `--export` command assembles a chapter's clips into a single finished WAV file:

```sh
./rustcorder --export --book MouseAndMotorcycle --chapter 1
```

- Reads `Chapter_01_timeline.txt` to determine which portions of each clip are canonical ("last recorded wins").
- Clips that were entirely superseded by a later punch-in are omitted.
- A linear crossfade (`CROSSFADE_TIME`, default 10 ms) is applied at every clip boundary to eliminate pops.
- Outputs a single `Chapter_01_export.wav` in the book directory.

---

## EPUB Reader

If a `.epub` file is present in the book directory, rustcorder serves it automatically through the browser viewer. No terminal pane — the browser is the only reading interface.

A local web server starts automatically on `localhost:7474` (configurable via `BROWSER_PORT`). Opening that URL in a browser shows a full epub.js rendition with:

- A status bar mirroring the TUI: mode, book, chapter, part, elapsed time, VU meter.
- The EPUB displayed at the last saved reading position (persisted in browser `localStorage` per book).
- Real-time recorder state sync via WebSocket (~10 Hz).
- Keyboard bindings that send recorder actions back to Rust (Space, P, Y, N, Esc).
- `←` / `→` scroll the EPUB directly in the browser; position is saved to `localStorage` automatically.

Set `BROWSER_OPEN=true` to have rustcorder automatically open the browser on startup.

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
| `src/main.rs` | Entry point — terminal setup, event loop, signal handling, CLI flags |
| `src/app.rs` | Application state machine and all business logic |
| `src/audio.rs` | ALSA capture thread, ALSA playback thread, RMS computation |
| `src/device.rs` | USB microphone discovery and enforcement via sysfs |
| `src/session.rs` | Session persistence, file naming, timeline index management |
| `src/wav.rs` | WAV file writer (fixed 48kHz/24-bit/mono format) |
| `src/render.rs` | Terminal UI rendering (ratatui) |
| `src/config.rs` | `--config` device wizard |
| `src/web/mod.rs` | axum HTTP/WebSocket server (browser viewer) |
| `src/web/state.rs` | `BrowserState` snapshot struct, channel types |
| `src/export/timeline.rs` | `--export` chapter assembly with crossfade |
| `src/playback/` | Experimental playback backends (rodio, cpal, PulseAudio) |
| `static/index.html` | epub.js browser viewer (served by the web server) |

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
| `axum` | HTTP and WebSocket server for the browser viewer |
| `tokio` | Async runtime for the web server thread |
| `serde` / `serde_json` | JSON serialization for WebSocket state messages |
| `zip` | ZIP extraction for serving EPUB file contents via the web server |
| `rodio` / `cpal` / `libpulse-*` | Experimental playback backends (`--playback`) |
