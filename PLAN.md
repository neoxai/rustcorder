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

# PHASE 2 — (TBD)

### Placeholder Goals

Potential future features (not yet specified):

* PipeWire-based monitoring
* Real-time RMS / peak metering
* ACX compliance analysis
* Automated silence trimming
* Chapter validation reports

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
