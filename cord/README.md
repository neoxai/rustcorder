# Cord - Audio Recording and Transcription Tool

A Rust-based CLI tool for recording audio, transcribing with Whisper, and matching recordings to scripts.

## Prerequisites

### System Dependencies

Install required system packages:
```bash
sudo apt install pkg-config libasound2-dev build-essential cmake
```

### Whisper Model

Download a Whisper model for transcription:
```bash
mkdir -p models
curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin -o models/ggml-base.bin
```

## Building

```bash
cargo build --release
```

## Commands

### Configure Microphone

Select your microphone and set gain (volume):
```bash
cargo run --release -- config
```

This will prompt you to:
- Select a microphone from the available devices
- Set a gain multiplier (e.g., 1.0 = normal, 2.0 = double volume)

Settings are saved to `user.config`.

### Record Audio

Basic recording:
```bash
cargo run --release -- record
```

Record with specific duration (10 seconds):
```bash
cargo run --release -- record --duration=10
```

Record with custom filename:
```bash
cargo run --release -- record --output=my_recording.wav
```

Override gain for this recording:
```bash
cargo run --release -- record --gain=2.5
```

### Transcribe Audio

Parse an audio file and transcribe it:
```bash
cargo run --release -- parse --input=./recordings/recording_1759243955.wav
```

This will:
- Convert the audio to 16kHz mono (supports MP3, WAV, FLAC, OGG, etc.)
- Transcribe using Whisper
- Save to `./recordings/transcription_1759243955.txt`

### Script Matching

Match existing recordings to a script:
```bash
cargo run --release -- script --input=./scripts/MaryLamb.txt
```

This will:
- Read all transcriptions from the `recordings` folder
- Match them to the script text
- Generate a compilation list at `./scripts/temp/MaryLamb_compiled.txt`

### Guided Script Recording

Record with script prompts (recommended workflow):
```bash
cargo run --release -- record --script=./scripts/MaryLamb.txt
```

This will:
1. Check `./scripts/temp/MaryLamb_compiled.txt` to see what's already recorded
2. Display the next unrecorded sentence(s)
3. Wait for you to press Enter
4. Record audio (press Space, Enter, or Ctrl+C to stop)
5. Auto-transcribe the recording
6. Update the compiled list

Repeat this command to progressively record the entire script.

## Example Workflow: Recording "Mary Had a Little Lamb"

### 1. Initial Setup

```bash
# Configure microphone and gain
cargo run --release -- config

# Create script directory
mkdir -p scripts

# Create your script file (or use existing)
cat > scripts/MaryLamb.txt << 'EOF'
Mary had a little lamb, its fleece was white as snow.
And everywhere that Mary went the lamb was sure to go.
It followed her to school one day which was against the rule.
It made the children laugh and play to see the lamb at school.
EOF
```

### 2. Record the Script

Start recording with script guidance:
```bash
cargo run --release -- record --script=./scripts/MaryLamb.txt
```

The tool will:
- Show you: "Mary had a little lamb, its fleece was white as snow."
- Press Enter to start recording
- Read the displayed text
- Press Space/Enter/Ctrl+C to stop
- Wait for transcription to complete

Run the command again:
```bash
cargo run --release -- record --script=./scripts/MaryLamb.txt
```

It will show the next unrecorded portion and repeat the process.

### 3. Review Compilation

Check the compiled list:
```bash
cat scripts/temp/MaryLamb_compiled.txt
```

This shows all recordings matched to the script with their positions and match scores.

### 4. Manual Transcription (if needed)

If you have existing recordings without transcriptions:
```bash
cargo run --release -- parse --input=./recordings/recording_1759243955.wav
```

Then update the compilation:
```bash
cargo run --release -- script --input=./scripts/MaryLamb.txt
```

## File Structure

```
cord/
├── recordings/              # Audio recordings and transcriptions
│   ├── recording_*.wav     # Audio files
│   └── transcription_*.txt # Transcription files
├── scripts/                # Script files
│   ├── MaryLamb.txt       # Your script
│   └── temp/              # Generated compilations
│       └── MaryLamb_compiled.txt
├── models/                 # Whisper models
│   └── ggml-base.bin
└── user.config            # Microphone settings
```

## Tips

- **Gain too low?** Run `cargo run --release -- config` to increase the gain multiplier
- **Recording quality:** Use a quiet environment and speak clearly
- **Overlapping recordings:** The guided recording mode ensures complete sentences, allowing some overlap
- **Model size:** Larger Whisper models (medium, large) provide better accuracy but are slower
- **Format support:** Input files can be MP3, WAV, FLAC, OGG, M4A, etc.