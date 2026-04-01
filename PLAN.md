# Rust Audiobook Recorder – PLAN Document

## Purpose

Build a **reliable, distraction-free audiobook narration recorder** on Ubuntu using **Rust**, optimized for spoken-word capture with a USB microphone. The system must prioritize **data integrity, correct device selection, and repeatability** over convenience features.

This document defines **Phase 1** in detail and reserves structure for later phases.

---

## Guiding Principles (Applies to All Phases)

1. **Never record from the wrong microphone**

   * USB broadcast mic only
   * No fallback to webcam, Bluetooth, headset, or built-in mic

2. **No irreversible DSP during capture**

   * No compression, normalization, EQ, noise reduction, or AGC

3. **Failure must be loud and obvious**

   * Missing device = blocking error
   * No silent misconfiguration

4. **Recorded files must be post-production friendly**

   * Lossless
   * Predictable naming
   * Clean session boundaries

---

## Audio Capture Baseline (All Phases)

These parameters are fixed unless explicitly revised in a future phase.

* Backend: **ALSA (hw: access)**
* Container: WAV (PCM)
* Bit depth: 24-bit (S24_LE written as pcm_s24le)
* Sample rate: 48,000 Hz
* Channels: Mono
* Gain control: External (ALSA mixer only)
* Peak target: < –6 dBFS
* Average narration level: ~ –18 dBFS

---

# PHASE 1 — Pure ALSA Recorder (MVP)

### Goal

Create a **rock-solid terminal-based recorder** that:

* Always records from the correct USB mic
* Produces correctly named files
* Gives clear visual feedback during recording
* Persists session state across launches

No monitoring, no PipeWire dependency, no DSP.

---

## Phase 1 Scope

### Included

* ALSA-only capture
* Terminal UI (TUI-lite)
* Device enforcement (USB mic only)
* Session persistence
* File naming and part enumeration

### Explicitly Excluded

* Audio monitoring
* PipeWire integration
* Editing or playback
* Loudness analysis
* ACX compliance validation

---

## Phase 1 Functional Requirements

### 1. Terminal UI Behavior

#### Startup

* On launch, the program:

  1. Loads last session state (if available)
  2. Displays current:

     * Book name
     * Chapter number
  3. Prompts user to confirm or edit values

If no prior session exists:

* Prompt for Book name
* Prompt for Chapter number

---

### 2. Session Persistence

* Session state must be written to disk on:

  * Program exit
  * Chapter completion
  * SIGINT / SIGTERM

#### Suggested format

`session.save.txt` (human-readable, line-based)

Example:

```
BOOK=The_Hobbit
CHAPTER=01
PART=3
```

#### Required behavior

* On startup, default to last saved values
* Never overwrite without explicit user action

---

### 3. File Naming and Directory Layout

#### Base directory

```
$PWD/BookName/
```

#### File naming pattern

```
Chapter_01_part001.wav
Chapter_01_part002.wav
...
```

#### Rules

* Chapter number is zero-padded (2 digits)
* Part number is zero-padded (3 digits)
* Part number increments automatically per recording
* Chapter completion resets part counter

---

### 4. Chapter Workflow

#### Recording loop

1. User starts recording
2. Audio is written to next part file
3. User stops recording
4. Program asks:

   * Continue chapter? (Y/n)
5. If yes:

   * Increment part counter
6. If no:

   * Mark chapter complete
   * Increment chapter number
   * Reset part counter
   * Save session state

---

### 5. USB Microphone Enforcement (Critical)

#### Requirements

* The recorder **must only record from the intended USB microphone**
* If the mic is not present:

  * Recording is blocked
  * A large, unmistakable warning is displayed

#### Device identification strategy

* Enumerate ALSA capture devices
* Match on one or more:

  * USB vendor ID
  * USB product ID
  * Stable ALSA device name substring

Example constraints:

* Device must be:

  * USB
  * Capture-capable
  * Match approved ID list

#### Hotplug behavior

* If mic is unplugged:

  * Recording immediately stops
  * File is closed cleanly
  * Warning is shown

* If mic is replugged:

  * Device is re-detected
  * User must explicitly restart recording

No automatic fallback. Ever.

---

### 6. Recording State Visibility

#### Mandatory on-screen indicators

At all times during recording, the UI must display:

* **RECORDING: RUNNING** (high-visibility text)
* Book name
* Chapter number
* Part number

#### Audio activity indicator

* A simple animated visual indicator:

  * Moves when audio samples are non-zero
  * Stays flat during silence

Notes:

* Indicator does NOT need to represent amplitude accurately
* Purpose is confirmation of signal presence only

---

### 7. Error Handling

#### Fatal errors (must stop recording)

* No approved USB microphone detected
* ALSA device read failure
* File write failure
* Buffer overrun (xrun)

#### Behavior

* Stop recording immediately
* Close file safely
* Print clear error message
* Require user action to continue

---

## Phase 1 Technical Architecture

### Audio Layer

* ALSA `hw:` device access
* Explicit format negotiation:

  * S24_LE
  * 48kHz
  * Mono

### Rust Crates (suggested)

* `alsa` — direct ALSA bindings
* `ratatui` — terminal UI
* `serde` (optional) — session file handling

---

## Phase 1 Deliverables

* Single Rust binary
* Terminal-based workflow
* Deterministic file output
* No dependency on PipeWire, PulseAudio, or FFmpeg

---



## Appendix A — Device Identification & Enforcement (USB VID/PID Strategy)

### A.1 Purpose

This appendix defines **how the recorder positively identifies the correct microphone** and guarantees that **no other audio capture device is ever used**, intentionally or accidentally.

Correct device identification is a **hard requirement**, not a preference.

---

### A.2 Approved Audio Capture Devices

The recorder SHALL operate **only** when one of the following devices is present.

| Manufacturer | Product     | USB VID  | USB PID  | Notes                        |
| ------------ | ----------- | -------- | -------- | ---------------------------- |
| Deity / RØDE | Deity VO-7U | `0x19f7` | `0x003c` | Primary narration microphone |

No other devices are approved unless explicitly added to this table in a future revision.

---

### A.3 Device Discovery Procedure

At program startup and immediately before entering a recording state, the recorder SHALL:

1. Enumerate all ALSA capture-capable devices
2. For each ALSA device:

   * Resolve its backing hardware path via `/sys/class/sound/card*/device`
   * Read the USB identifiers:

     * `idVendor`
     * `idProduct`
3. Compare the detected VID/PID pair against the approved device table
4. Select **exactly one** matching device

Failure cases:

* **Zero matches** → fatal error, recording disabled
* **More than one match** → fatal error, recording disabled

No heuristics, scoring, or fallback logic is permitted.

---

### A.4 Hotplug and Runtime Enforcement

The recorder SHALL continuously enforce device correctness:

* If the approved USB microphone is unplugged:

  * Recording MUST stop immediately
  * The active WAV file MUST be closed cleanly
  * A blocking, high-visibility warning MUST be displayed

* When the microphone is reconnected:

  * The device MUST be re-validated via VID/PID
  * Recording MUST NOT automatically resume
  * User action is required to restart recording

---

### A.5 Explicitly Rejected Devices

The following device classes MUST NEVER be used under any circumstances:

* Built-in laptop microphones
* Webcam microphones
* Bluetooth audio devices
* Headsets or headset microphones
* Any USB audio device whose VID/PID is not listed in Appendix A.2

If only rejected devices are present, the program SHALL refuse to record.

---

### A.6 Rationale

This strategy ensures:

* Deterministic device selection
* Immunity to ALSA card reordering
* Protection against silent misconfiguration
* Zero risk of recording from the wrong microphone

This approach is intentionally strict and aligns with the project’s core philosophy:

> "If recording starts, it is guaranteed to be the correct microphone."

---

## Appendix B — Signal Presence & Hardware Mute Protection

### B.1 Purpose

This appendix defines mandatory safeguards to prevent accidental recording with the microphone **hardware-muted** or otherwise producing silence.

Because USB microphone hardware mute buttons typically operate **before the ADC** and do not reliably expose a mute state to the operating system, **silence detection at the signal level is the only robust solution**.

---

### B.2 Definitions

* **Silence**: Audio whose RMS level remains below a defined threshold for a sustained period.
* **Non-silent audio**: Audio whose RMS level exceeds the silence threshold.

Recommended baseline values (subject to tuning):

| Parameter                      | Value          |
| ------------------------------ | -------------- |
| Silence threshold              | RMS < –60 dBFS |
| Pre-record validation window   | 2 seconds      |
| Runtime silence warning window | 15 seconds     |

---

### B.3 Pre-Recording Signal Validation (Mandatory)

Before any audio is written to disk, the recorder SHALL:

1. Open the approved ALSA capture device
2. Capture audio for the validation window duration
3. Compute RMS levels over sliding windows
4. Determine whether **any** window exceeds the silence threshold

If **no non-silent audio** is detected:

* Recording SHALL NOT begin
* No file SHALL be created
* A blocking, high-visibility error MUST be displayed, indicating likely causes:

  * Microphone hardware mute enabled
  * Incorrect microphone placement
  * Faulty or disconnected microphone

Example message (illustrative):

```
ERROR: No audio detected.
Check microphone mute button.
Recording not started.
```

This check is REQUIRED every time recording is initiated.

---

### B.4 Runtime Silence Watchdog (Strongly Recommended)

While recording is active, the recorder SHOULD continuously monitor signal presence.

Behavior:

* Track elapsed time since last detected non-silent audio
* If silence exceeds the configured runtime silence window:

  * A prominent warning MUST be displayed
  * Warning MUST remain visible until non-silent audio resumes or recording stops

Example warning (illustrative):

```
⚠ WARNING: NO AUDIO DETECTED FOR 15 SECONDS ⚠
Check microphone mute button.
```

Optional but acceptable behaviors:

* Pause recording automatically
* Require user acknowledgement to continue

---

### B.5 UI Integration

The audio activity indicator defined in Phase 1 SHALL be driven by the same signal-presence logic:

* Animated / moving indicator → non-silent audio detected
* Flat / static indicator → silence detected

This ensures the visual feedback is:

* Semantically meaningful
* Directly tied to recording safety

---

### B.6 Failure Semantics

Extended silence SHALL be treated as a **potential fault condition**, not a normal operating state, during active recording.

This appendix intentionally biases toward:

* False positives (brief pauses triggering warnings)
* Over-communication to the user

Rather than risking silent loss of recorded narration.

---

### B.7 Rationale

This strategy:

* Requires no device-specific mute APIs
* Works uniformly across USB microphones
* Protects against the most common narration failure mode
* Aligns with the project philosophy of loud, early failure

---

## Non-Goals (Explicit)

* Live audio effects
* Auto-gain control
* Streaming or broadcast
* GUI application (for now)

---

## Summary

Phase 1 produces a **trustworthy narration capture tool** whose primary value is:

> "If it recorded, it recorded the right mic, correctly labeled, with clean audio."

Everything else can be layered later without compromising captured data.

---

---

# PHASE 2 — Punch-and-Roll & Timeline Index

## Goal

Extend the recorder with the **punch-and-roll** technique used in professional audiobook narration, and introduce a **human-readable timeline index** that tracks every clip's position in the overall book recording as a single continuous timeline.

No complex database. No audio editing. Files are never modified after recording — only metadata is added.

---

## Guiding Principles (Phase 2 additions)

5. **The timeline is the source of truth for clip ordering and overlap**

   * File timestamps and part numbers alone do not define playback order or splice points
   * The timeline index file is the authoritative record

6. **Files are immutable after recording**

   * Punch-and-roll creates a new file; it never overwrites or truncates an existing file
   * The old file remains intact on disk — the timeline marks it as partially superseded

7. **The rollback and punch-in points are derived from measured data, not estimates**

   * Clip duration is computed from bytes written (`data_bytes / byte_rate`)
   * Elapsed playback time is measured from the moment playback begins

---

## Phase 2 Scope

### Included

* Timeline index file (one per chapter)
* `TIMELINE_POS` field added to `session.save.txt`
* Punch-and-roll workflow: stop → rewind N seconds → play → punch in → record
* Configurable rewind duration via `PUNCH_BACK_TIME` environment variable (default: 15 s)
* `.env` file in the repository root for local configuration
* Playback via default ALSA PCM output (default device, no configuration)
* New application states: `PunchRollback` and `PunchReady`
* UI for rollback playback and punch-in waiting

### Explicitly Excluded

* Configurable playback output device (reserved for Phase 3)
* Automatic assembly or mixdown of clips
* Waveform display during rollback
* Undo / redo beyond a single punch operation
* Editing or trimming of existing files

---

## Core Concept: The Timeline Model

All recorded clips for a chapter are treated as segments of a single continuous timeline. Each clip has an **absolute start position** in that timeline, measured in seconds from the beginning of the chapter.

```
Timeline (seconds):
0         127.3     254.9
│─────────────│─────────│──────────→
  part001       part002    part003
```

A clip **owns** audio from its `START` until the earlier of:
* Its own end (start + duration), or
* The `START` of the next clip in the timeline

If a later clip starts *before* the previous clip ends, the overlap region belongs exclusively to the later clip. This is the punch-in point — no audio surgery required.

```
Timeline with punch-in overlap:
0              142.3
│──────────────────────────────→  part001 (full duration)
         127.3
         │──────────────────────→  part002 (starts mid-part001)
                                ↑
         part001 is "owned" only up to 127.3 s
         part002 owns everything from 127.3 s onward
```

---

## Phase 2 Functional Requirements

### 1. Timeline Index File

#### Location

```
BookName/Chapter_01_timeline.txt
```

One file per chapter, stored in the same directory as the WAV files.

#### Format

```
# rustcorder timeline — The_Hobbit / Chapter 01
CLIP Chapter_01_part001.wav START=0.000
CLIP Chapter_01_part002.wav START=127.342
CLIP Chapter_01_part003.wav START=254.891
```

#### Rules

* `#` lines are comments and are ignored during parsing
* `CLIP` entries are in order of recording (not necessarily ascending `START` — a punch clip's START is less than the previous clip's end)
* `START` is in seconds, three decimal places (millisecond precision)
* A `CLIP` entry is appended at the **moment recording begins** for that clip, not at stop
  * A crash mid-recording still leaves a valid (if short) entry
* Entries are never deleted or rewritten — only appended

#### Interpretation

To reconstruct the canonical audio for a chapter:
1. Sort clips by `START` ascending
2. For each clip, take audio from `START` until `min(clip_end, next_clip.START)`
3. If no next clip exists, take audio until clip end

---

### 2. Session State: `TIMELINE_POS` Field

`session.save.txt` gains one new field:

```
BOOK=The_Hobbit
CHAPTER=01
PART=3
TIMELINE_POS=254.891
```

* `TIMELINE_POS` is the absolute timeline position (seconds) at which the **most recently started** clip begins
* It is updated:
  * At start of each new recording (written immediately)
  * After a normal stop: `TIMELINE_POS += clip_duration` — ready for the next clip
  * After punch-and-roll: `TIMELINE_POS = punch_in_time` — set to the exact punch-in point
* It is never decremented except by a punch-and-roll operation
* If missing from a saved session (e.g. upgrading from Phase 1), it defaults to `0.0`

---

### 3. Punch-and-Roll Workflow

#### Trigger

* During active recording, user presses **`P`**

#### Step-by-step behavior

1. **Stop capture**: recording halts; WAV file is finalized
2. **Compute rollback point**:
   * `clip_duration = data_bytes / 144_000.0` (seconds)
   * `punch_back_time = PUNCH_BACK_TIME env var (default: 15.0 s)`
   * `rollback_offset = max(0.0, clip_duration - punch_back_time)` (seconds into the clip file)
   * `rollback_abs = clip_start_timeline + rollback_offset` (absolute timeline position)
3. **Begin playback** of the just-recorded file starting at `rollback_offset` seconds
   * Output device: ALSA default PCM (`default`)
   * No output device selection — uses whatever the OS considers default
4. **Enter `PunchRollback` mode**: UI shows "ROLLING BACK... press SPACE to punch in"
5. **User presses SPACE**: playback stops; elapsed playback time is recorded
6. **Compute punch-in time**:
   * `punch_in_time = rollback_abs + elapsed_since_playback_started`
7. **Update state**:
   * Increment part counter
   * Set `TIMELINE_POS = punch_in_time`
   * Save session
8. **Append timeline entry**:
   * `CLIP Chapter_NN_partNNN.wav START=<punch_in_time>`
9. **Enter `PunchReady` mode**: briefly display punch-in time, then transition to `PreCheck` → `Recording`

#### Escape / Abort during rollback

* User presses **`Esc`** during `PunchRollback`: playback stops; state returns to `PostRecording` (normal stop prompt)
* The partially-played rollback leaves no trace — no timeline entry is written until recording actually begins

---

### 4. Normal Recording Flow (updated)

On normal start of recording (non-punch):

1. Compute `clip_start = TIMELINE_POS` (current value from session)
2. Append to timeline: `CLIP <filename> START=<clip_start>`
3. Begin recording
4. On stop: `TIMELINE_POS += clip_duration` → save session → `PostRecording` as before

This ensures every clip, punch or not, has a timeline entry.

---

### 5. Application States (additions)

Two new states are added to the `AppMode` enum:

| State | Description |
|---|---|
| `PunchRollback` | Playback of the last 15 s is in progress; waiting for SPACE to punch in |
| `PunchReady` | Punch-in time confirmed; transitioning to pre-check before new recording |

Updated state machine (abbreviated):

```
Recording ──P──→ PunchRollback ──SPACE──→ PunchReady ──→ PreCheck ──→ Recording
                      │
                     ESC
                      │
                      ▼
               PostRecording
```

---

### 6. Environment Variable Configuration

#### `.env` file

A `.env` file in the repository root provides local defaults. It is loaded at application startup before any other initialization. Variables already set in the shell environment take precedence over the `.env` file.

| Variable | Type | Default | Description |
|---|---|---|---|
| `PUNCH_BACK_TIME` | `f64` (seconds) | `15.0` | How far before the end of the current recording to begin rollback playback |

#### Loading rules

* The application reads `.env` from the **current working directory** at startup
* If the file is absent, built-in defaults are used silently
* If `PUNCH_BACK_TIME` is present but unparseable as a positive number, the application prints a warning and falls back to `15.0`
* Minimum enforced value: `1.0` s (values below this are clamped and a warning is shown)
* No maximum is enforced

#### Implementation note

Use the `dotenvy` crate (successor to `dotenv`) to load the file. It is a minimal, well-maintained crate with no transitive dependencies beyond `std`. Call `dotenvy::dotenv().ok()` early in `main()` — the `.ok()` discards the error silently when no `.env` file is present.

---

### 7. Playback Implementation


#### Constraints

* Output to ALSA `default` PCM device — no device selection in Phase 2
* Read directly from the finalized WAV file (seek to `rollback_offset * byte_rate`)
* Same fixed format as recording: 24-bit, 48 kHz, mono
* Playback runs in a background thread (same pattern as capture), sending events to the main loop
* Playback thread signals `PlaybackDone` when the file ends (in case the clip is shorter than 15 s)

#### Output device note

The `default` ALSA PCM device is used unconditionally. On most desktop Ubuntu systems this routes to PulseAudio/PipeWire and therefore the default speaker output. A future phase will allow the user to specify a preferred output device with a fallback priority list.

---

### 8. UI During Punch-and-Roll

#### `PunchRollback` screen

* Large text: **ROLLING BACK**
* Playback position counter (e.g. `+3.2 s / 15.0 s` where 15.0 reflects `PUNCH_BACK_TIME`)
* Filename being played
* Instruction: `SPACE = punch in here   ESC = cancel`

#### `PunchReady` screen (brief, ~1 s)

* Large text: **PUNCHING IN**
* Show the absolute punch-in timestamp: e.g. `Timeline position: 127.342 s`
* Automatically transitions to `PreCheck`

---

## Phase 2 Technical Architecture

### New / Modified Components

| File | Change |
|---|---|
| `src/session.rs` | Add `timeline_pos: f64` to `Session`; load/save it; add `append_timeline_entry()` function |
| `src/app.rs` | Add `AppMode::PunchRollback` and `AppMode::PunchReady`; add playback handle and punch-in timer fields; handle `P` key in `Recording` state; read `punch_back_time` from env at startup |
| `src/audio.rs` | Add `start_playback(path, offset_secs)` function returning a `PlaybackHandle` with an event channel |
| `src/render.rs` | Add rendering for `PunchRollback` and `PunchReady` states |
| `.env` | New file — `PUNCH_BACK_TIME=15` (repository-level default; users may override in shell environment) |

### No new crates required

* WAV reading for playback uses the standard `std::fs::File` + `std::io::Seek` — same as `WavWriter` but in reverse
* ALSA playback uses the existing `alsa` crate (write path instead of read path)

---

## Phase 2 Deliverables

* `P` key triggers punch-and-roll during recording
* Rollback plays the last 15 seconds (or full clip if shorter) to default speakers
* `SPACE` during rollback sets the punch-in point and begins a new recording
* `Chapter_NN_timeline.txt` is created and maintained automatically
* `session.save.txt` tracks `TIMELINE_POS`
* All existing Phase 1 behavior and guarantees remain intact

---

## Appendix C — Timeline File Interpretation Reference

### C.1 Example: Three clean takes (no punch)

```
CLIP Chapter_01_part001.wav START=0.000
CLIP Chapter_01_part002.wav START=142.317
CLIP Chapter_01_part003.wav START=289.004
```

Canonical audio: part001[0→142.317 s] + part002[0→146.687 s] + part003[0→end]

### C.2 Example: Punch-and-roll on second take

```
CLIP Chapter_01_part001.wav START=0.000
CLIP Chapter_01_part002.wav START=127.342
```

`part001` duration: 142.317 s.
`part001` owns: 0 → 127.342 s
`part002` owns: 127.342 s → end

Audio from `part001` between 127.342 s and 142.317 s is discarded in the final assembly.

### C.3 Example: Two punches in succession

```
CLIP Chapter_01_part001.wav START=0.000
CLIP Chapter_01_part002.wav START=127.342
CLIP Chapter_01_part003.wav START=118.500
```

`part002` owns: 127.342 → 118.500 s — wait, this is *earlier* than part002's start.
This means part003 punched *back into part001's* territory (user rolled back further than part002's start).
Result: part001 owns 0 → 118.500 s; part003 owns 118.500 s → end; part002 is entirely superseded.

The interpretation algorithm handles this correctly by always taking `min(clip_end, next_clip.START)`.

---

## Non-Goals (Phase 2 Explicit)

* Configurable playback output device (Phase 3)
* Waveform scrubbing or variable rewind duration (configurable in a future phase)
* Timeline visualization in the TUI
* Automatic mixdown or export
* Undo of a completed punch (the old file still exists; the user can manually edit the timeline file)
