# Rustcorder Phase 1 (minimal)

Requirements:
- Linux (Ubuntu)
- `arecord` (alsa-utils) installed
- Rust toolchain

Build:
```bash
cargo build --release
```

Run:
```bash
./target/release/rustcorder
```

Notes:
- Program enforces USB VID/PID 0x19f7:0x003c by scanning `/sys/class/sound/card*/device`.
- Output placed at `$PWD/<BookName>/Chapter_XX_partNNN.wav`.
- Session state persisted to `session.save.txt`.
