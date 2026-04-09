# State Machine Implementation Plan

## Overview

This plan replaces the existing TUI-centric state machine (Ready, PreCheck,
PostRecording, PunchRollback) with a simpler three-state model driven primarily
from the browser. The terminal UI is retained for export and diagnostics only.

---

## New States

| State | Colour | Meaning |
|---|---|---|
| `Standby` | Yellow | Idle, cursor parked at a timeline position |
| `Playing` | Green | Streaming audio from the timeline |
| `Recording` | Red | Capturing audio to a new part file |
| `MicError` | — | No approved mic detected (non-fatal, retryable) |
| `Fatal` | — | Unrecoverable error (unchanged) |

**Removed states:** `Ready`, `PreCheck`, `PostRecording`, `PunchRollback`

---

## Key Bindings

### From STANDBY
| Key | Action |
|---|---|
| `Space` | → RECORDING — start new part at current `timeline_pos` |
| `L` | → PLAYING — begin playback from current `timeline_pos` |
| `J` | Stay STANDBY — move `timeline_pos` to start of current part |
| `K` | Stay STANDBY — move `timeline_pos` to start of next part (or end of last recorded audio if already on the last part) |
| `N` | Stay STANDBY — advance to next chapter, reset part=1, timeline_pos=0 |
| `Q` / `Esc` | Quit |

### From PLAYING
| Key | Action |
|---|---|
| `Space` | → RECORDING — stop playback, start new part at current playhead position (punch-in) |
| `L` | → STANDBY — stop playback, park `timeline_pos` at current playhead |
| `P` | Stay PLAYING — rewind `PUNCH_BACK_TIME` seconds (floor: 0.0), restart playback |
| `J` | Stay PLAYING — jump to start of current part, restart playback |
| `K` | Stay PLAYING — jump to start of next part, restart playback |
| *(end of audio)* | → STANDBY — park `timeline_pos` at end of last recorded audio |

### From RECORDING
| Key | Action |
|---|---|
| `Space` | → STANDBY — finalize WAV, park `timeline_pos` after new clip |
| `P` | → PLAYING — finalize WAV, rewind `PUNCH_BACK_TIME` seconds, begin playback |

---

## Background Mic Check

- Runs **once at startup only**, during the initial transition into Standby.
- Spawned as a background thread (reusing the existing precheck infrastructure).
- If the check passes, no visible change.
- If the check fails (no mic signal / no device), transition to `MicError`.
- **Does not block any key transitions.** If the user acts before the 2 seconds are up, the action proceeds immediately and the background thread is discarded.
- Re-entering Standby (e.g. after stopping a recording) does not re-run the check.

---

## Detailed Transition Logic

### STANDBY → RECORDING (`Space`)
1. Abort the background mic check if still running.
2. `clip_start_timeline = timeline_pos`
3. `session.advance_part()`, append timeline entry, save session.
4. Start capture thread + WAV writer.

### STANDBY → PLAYING (`L`)
1. Call `timeline_segments_from(timeline_pos)` (see §New Timeline Functions).
2. Start playback thread with the resulting segments.
3. `playhead_origin = timeline_pos` (the field currently called `punch_rollback_abs`).
4. Reset `playback_start = Instant::now()`, `playback_done = false`.

### STANDBY `J`
1. Call `current_part_at(timeline_pos)` → returns the part whose canonical range covers `timeline_pos`.
2. `timeline_pos = part.start`; save session.

### STANDBY `K`
1. Call `next_part_after(timeline_pos)` → returns the next canonical part start, or `None` at end.
2. If `Some(next_start)`: `timeline_pos = next_start`.
3. If `None` (already on last part): `timeline_pos = end_of_last_recorded_audio()`.
4. Save session.

### STANDBY `N`
1. `session.chapter += 1`, `session.part = 1`, `session.timeline_pos = 0.0`.
2. Save session.

### PLAYING → STANDBY (`L`)
1. Stop playback.
2. `timeline_pos = playhead_origin + elapsed_secs`.
3. Save session.

### PLAYING → RECORDING (`Space`) — punch-in
1. Stop playback (non-blocking signal, same as current `punch_in`).
2. `punch_in_time = playhead_origin + elapsed_secs`, capped at end of last recorded audio.
3. `session.timeline_pos = punch_in_time`.
4. `clip_start_timeline = punch_in_time`.
5. `session.advance_part()`, append timeline entry, save session.
6. Start capture thread + WAV writer.

### PLAYING `P`
1. Stop playback.
2. `playhead_origin = (playhead_origin - punch_back_time).max(0.0)`.
3. Rebuild segments via `timeline_segments_from(playhead_origin)`.
4. Restart playback from the new origin. (Same as current `rewind_punch_rollback`.)

### PLAYING `J`
1. Compute `current_pos = playhead_origin + elapsed_secs`.
2. Call `current_part_at(current_pos)` → `part_start`.
3. Stop playback.
4. `playhead_origin = part_start`.
5. Rebuild segments via `timeline_segments_from(part_start)` and restart.

### PLAYING `K`
1. Compute `current_pos = playhead_origin + elapsed_secs`.
2. Call `next_part_after(current_pos)`.
3. If `Some(next_start)`: `playhead_origin = next_start`.
4. If `None`: `playhead_origin = end_of_last_recorded_audio()`.
5. Stop playback, rebuild segments, restart.

### PLAYING → STANDBY (end of audio)
- Detected in `tick_playing` when `playback_done == true`.
- `timeline_pos = end_of_last_recorded_audio()`.
- Save session, enter Standby.

### RECORDING → STANDBY (`Space`)
1. Finalize WAV, stop capture.
2. `session.timeline_pos = clip_start_timeline + last_clip_duration`.
3. Save session.

### RECORDING → PLAYING (`P`)
1. Finalize WAV, stop capture. (Same as the front half of current `begin_punch`.)
2. Handle short-clip discard if configured.
3. `clip_end = clip_start_timeline + last_clip_duration`.
4. `playhead_origin = (clip_end - punch_back_time).max(0.0)`.
5. Call `start_punch_playback()` (already refactored), which calls `timeline_segments_from` or the existing punch-segment builder and starts the playback thread.
6. Enter PLAYING.

---

## New Timeline Functions (`src/export/timeline.rs`)

### `timeline_segments_from(timeline_path, book_dir, start_pos) -> Result<Vec<PunchSegment>>`

Generalization of `punch_rollback_segments` that does **not** require a
"current clip" argument. Builds segments covering `[start_pos, end_of_chapter]`
by applying "last recorded wins" across all entries in the timeline file.

- Iterate all entries, compute effective range for each (same as `resolve_clips`).
- For each canonical clip that overlaps `[start_pos, ∞)`, emit a `PunchSegment`
  with the appropriate `file_offset_secs` and `play_secs`.
- The last segment always has `play_secs = f64::INFINITY` (play to EOF).

This replaces the ad-hoc segment assembly in `begin_punch` / `start_punch_playback`
for all cases where we do not have an unfinished "current clip" in flight.

### `current_part_at(timeline_path, book_dir, pos) -> Result<PartInfo>`

Returns the canonical clip whose effective range covers `pos`.
- Parse timeline, resolve clips.
- Find the entry `i` where `entry[i].start <= pos < effective_end[i]`.
- Returns `PartInfo { filename, start_secs }`.
- If `pos` is beyond all clips, returns the last clip.
- If the timeline is empty, returns an error.

### `next_part_after(timeline_path, book_dir, pos) -> Result<Option<f64>>`

Returns the absolute timeline start of the next canonical clip after `pos`,
or `None` if `pos` is already within the last clip.

### `end_of_last_recorded_audio(timeline_path, book_dir) -> Result<f64>`

Returns `last_entry.start + last_entry.wav_duration`. Used when K is pressed
on the last part and for end-of-playback cursor parking.

---

## App State Changes (`src/app.rs`)

### Renamed/removed fields
- `punch_rollback_abs` → rename to `playhead_origin` (used for both PLAYING and
  RECORDING→PLAYING; semantics are identical).
- Remove `precheck_*` signal-detection fields — repurpose as background-check
  state (or keep field names if reusing the existing precheck thread logic).

### Removed handlers / methods
| Removed | Replaced by |
|---|---|
| `handle_key_ready` | `handle_key_standby` |
| `handle_key_precheck` | background check, no blocking handler |
| `handle_key_post` | removed (chapter advance → `N` in Standby) |
| `handle_key_punch_rollback` | `handle_key_playing` |
| `begin_punch` | `punch_to_playing` (same logic, mode target changes) |
| `abort_punch_rollback` | `playing_to_standby` |
| `tick_precheck` | background thread check |

### New methods
| Method | Purpose |
|---|---|
| `handle_key_standby` | J / K / L / N / Space / Q |
| `handle_key_playing` | J / K / L / P / Space |
| `begin_playing(from_pos)` | start timeline playback from an arbitrary position |
| `standby_jump_back` | J in Standby — move cursor to current part start |
| `standby_jump_forward` | K in Standby — move cursor to next part start or end |
| `playing_jump_back` | J in Playing — restart from current part start |
| `playing_jump_forward` | K in Playing — restart from next part start |
| `playing_to_standby` | L in Playing — park cursor, stop playback |
| `advance_chapter` | N in Standby |
| `stop_recording_to_standby` | Space in Recording |
| `punch_to_playing` | P in Recording — replaces `begin_punch` |

---

## Browser Changes (`static/index.html`)

### ABSORB set
Add `KeyJ`, `KeyK`, `KeyL`, `KeyN` to the absorbed key set.

### Key handler additions

```
case 'Standby':
  if (key === 'Space')              send({ action: 'start' });
  else if (key === 'KeyL')          send({ action: 'play' });
  else if (key === 'KeyJ')          send({ action: 'jump_back' });
  else if (key === 'KeyK')          send({ action: 'jump_forward' });
  else if (key === 'KeyN')          send({ action: 'next_chapter' });
  else if (key === 'KeyQ' || key === 'Escape') send({ action: 'quit' });
  break;

case 'Playing':
  if (key === 'Space')              send({ action: 'start' });   // punch-in
  else if (key === 'KeyL')          send({ action: 'pause' });
  else if (key === 'KeyP')          send({ action: 'punch' });
  else if (key === 'KeyJ')          send({ action: 'jump_back' });
  else if (key === 'KeyK')          send({ action: 'jump_forward' });
  break;

case 'Recording':
  if (key === 'Space' || key === 'Enter') send({ action: 'stop' });
  else if (key === 'KeyP')          send({ action: 'punch' });
  break;
```

### Timeline position display

A persistent position readout must be visible at all times in the browser UI,
formatted as `MM:SS.s` (e.g. `03:42.5`):

- **Standby**: show `timeline_pos_secs` (the cursor / next-record position).
- **Playing**: show `playhead_pos_secs` (live, updating ~10 Hz via WebSocket).
- **Recording**: show `timeline_pos_secs` (the punch-in start of the current clip).

The readout should sit prominently in the status bar alongside the existing
book/chapter/part fields, not buried in a corner.

### Status bar colour
Map `mode` to a CSS class or inline colour:
- `Standby` → yellow / amber
- `Playing` → green
- `Recording` → red
- `MicError` / `Fatal` → existing error styling

---

## `BrowserState` Changes (`src/web/state.rs`)

- Add field `playhead_pos_secs: f64` — live playhead during PLAYING (= `playhead_origin + elapsed`); mirrors `timeline_pos_secs` otherwise. The browser always uses this field for the position readout regardless of mode.
- `timeline_pos_secs` (already present) continues to carry the cursor/next-record position; the browser can show whichever is appropriate per mode, or always show `playhead_pos_secs` since they converge when not playing.
- Update `mode` string mapping: `Standby`, `Playing`, `Recording`, `MicError`, `Fatal`.
- Update `footer_hints` for all new modes.
- Remove entries for `Ready`, `PreCheck`, `PostRecording`, `PunchRollback`.

---

## `drain_actions` Changes (`src/app.rs` or `src/web/mod.rs`)

Map new action strings to `KeyEvent`s:

| Action string | Synthetic key |
|---|---|
| `"play"` | `KeyCode::Char('l')` |
| `"pause"` | `KeyCode::Char('l')` |
| `"jump_back"` | `KeyCode::Char('j')` |
| `"jump_forward"` | `KeyCode::Char('k')` |
| `"next_chapter"` | `KeyCode::Char('n')` |
| `"start"` | `KeyCode::Char(' ')` (existing) |
| `"stop"` | `KeyCode::Char(' ')` (existing) |
| `"punch"` | `KeyCode::Char('p')` (existing) |
| `"quit"` | `KeyCode::Char('q')` (existing) |

---

## Render Changes (`src/render.rs`)

- Remove rendering for `Ready`, `PreCheck`, `PostRecording`, `PunchRollback`.
- Add `Standby` screen: show book/chapter/part, `timeline_pos` timestamp, key hints (J/K/L/N/Space).
- Add `Playing` screen: show playhead position, progress through chapter, key hints (J/K/L/P/Space).
- `Recording` screen: essentially unchanged (timer, VU meter, `Space`/`P` hints).

---

## Implementation Order

1. **`src/export/timeline.rs`** — add the four new public functions (`timeline_segments_from`, `current_part_at`, `next_part_after`, `end_of_last_recorded_audio`). These are pure, testable, and block no other work.

2. **`src/app.rs`** — replace the state machine:
   a. Rename `AppMode` variants.
   b. Rename `punch_rollback_abs` → `playhead_origin`.
   c. Wire up new handlers/methods using the new timeline functions.
   d. Background mic check (reuse existing precheck thread, remove the blocking tick loop).

3. **`src/web/state.rs`** — update `BrowserState` fields and mode strings.

4. **`src/render.rs`** — update TUI screens.

5. **`static/index.html`** — update key bindings and status bar colours.
