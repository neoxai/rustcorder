


List of states: 
- STANDBY (YELLOW): not recording, not playing back. Just waiting around. Default state when starting the program. Should be tied to a single TIMESTAMP in the chapter

- PLAYING (GREEN): playing the audio using the timeline to stack the most recent clips if any of them overlap.

- RECORDING (RED): actively recording from the mic, creating a audio file "part" for the chapter. The begininning of the "part" should have a timestamp in the timeline file.

Transitions:
    - From STANDBY: 
        SPACE - to begin RECORDING
        "L" - to begin PLAYING
        "J" - Move the timestamp back the the beginning of the current "part".
        "K" - move the timestamp to the beginning of the next "part" clip, or the very end of the last "part" if we are at the end of the current recording
    - From PLAYBACK:
        SPACE - to begin RECORDING. Playback will stop immediately
        "P" - Punch back X (configurable) seconds, start PLAYBACK again from that point. If reaches the beginning (zero timestamp) or earlier, start PLAYBACK from Zero time of this chapter.
        "L" -  to pause playback, change to STANDBY
        "J" - Stop playing the current audio, and move to the beginning of the current "part" and continue PLAYBACK from there
        "K" - Stop playing the current audio, and move to the end of the current "part" and continue PLAYBACK from there.
    - From RECORDING:
        SPACE - to stop recording and switch to STANDBY
        "P" - Punch back X (configurable) seconds, start PLAYBACK from that point.