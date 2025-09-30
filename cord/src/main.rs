use clap::{Parser, Subcommand};
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};
use std::path::PathBuf;
use std::fs;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{WavSpec, WavWriter};
use crossterm::{terminal, event::{self, Event, KeyCode, KeyEvent}};
use dialoguer::{theme::ColorfulTheme, Select};
use serde::{Deserialize, Serialize};
use symphonia::core::audio::{SampleBuffer, SignalSpec};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction};

#[derive(Parser)]
#[command(name = "cord")]
#[command(about = "Audio transcription and search tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse audio file to text using Whisper
    Parse {
        /// Input audio file path
        #[arg(long)]
        input: PathBuf,

        /// Path to Whisper model file (e.g., ggml-base.bin)
        #[arg(long, default_value = "models/ggml-base.bin")]
        model: PathBuf,
    },
    /// Record audio from microphone
    Record {
        /// Output filename (optional, defaults to timestamp)
        #[arg(long)]
        output: Option<String>,

        /// Duration in seconds (optional, press Ctrl+C to stop)
        #[arg(long)]
        duration: Option<u64>,

        /// Gain multiplier (optional, uses config value if not specified)
        #[arg(long)]
        gain: Option<f32>,

        /// Path to Whisper model file for auto-transcription (e.g., ggml-base.bin)
        #[arg(long, default_value = "models/ggml-base.bin")]
        model: PathBuf,

        /// Path to script file for guided recording
        #[arg(long)]
        script: Option<PathBuf>,
    },
    /// Configure microphone settings
    Config,
    /// Match script to recordings and create compilation list
    Script {
        /// Path to script text file
        #[arg(long)]
        input: PathBuf,
    },
}

#[derive(Serialize, Deserialize, Debug)]
struct UserConfig {
    selected_device: String,
    #[serde(default = "default_gain")]
    gain: f32,
}

fn default_gain() -> f32 {
    1.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Parse { input, model } => {
            parse_audio(&input, &model)?;
        }
        Commands::Record { output, duration, gain, model, script } => {
            record_audio(output, duration, gain, model, script)?;
        }
        Commands::Config => {
            configure_microphone()?;
        }
        Commands::Script { input } => {
            match_script(&input)?;
        }
    }

    Ok(())
}

fn parse_audio(input_path: &PathBuf, model_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading Whisper model from: {:?}", model_path);

    // Load the Whisper model
    let ctx = WhisperContext::new_with_params(
        model_path.to_str().unwrap(),
        WhisperContextParameters::default()
    )?;

    println!("Transcribing audio file: {:?}", input_path);

    // Create a state
    let mut state = ctx.create_state()?;

    // Set up parameters for transcription
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    // Convert audio file to the required format
    // Note: whisper-rs expects 16kHz mono PCM data
    let audio_data = load_audio(input_path)?;

    // Run transcription
    state.full(params, &audio_data)?;

    // Get results
    let num_segments = state.full_n_segments()?;
    let mut transcription = String::new();
    for i in 0..num_segments {
        let segment = state.full_get_segment_text(i)?;
        println!("{}", segment);
        transcription.push_str(&segment);
    }

    // Generate output filename based on input filename
    let output_path = if let Some(parent) = input_path.parent() {
        let stem = input_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("transcription");

        // Replace "recording_" with "transcription_" if present
        let output_name = if stem.starts_with("recording_") {
            stem.replace("recording_", "transcription_")
        } else {
            format!("{}_transcription", stem)
        };

        parent.join(format!("{}.txt", output_name))
    } else {
        PathBuf::from("transcription.txt")
    };

    // Save transcription to file
    fs::write(&output_path, transcription)?;
    println!("\nTranscription saved to: {:?}", output_path);

    Ok(())
}

fn load_audio(path: &PathBuf) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Open the audio file
    let file = fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Create a probe hint using the file extension
    let mut hint = Hint::new();
    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            hint.with_extension(ext_str);
        }
    }

    // Probe the media source
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())?;

    let mut format = probed.format;
    let track = format.tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("No supported audio track found")?;

    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    // Create a decoder for the track
    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())?;

    let mut audio_samples = Vec::new();
    let mut spec_option: Option<SignalSpec> = None;

    // Decode all packets
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(_) => break,
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                if spec_option.is_none() {
                    spec_option = Some(*decoded.spec());
                }

                let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                sample_buf.copy_interleaved_ref(decoded);
                audio_samples.extend_from_slice(sample_buf.samples());
            }
            Err(_) => continue,
        }
    }

    let spec = spec_option.ok_or("No audio data decoded")?;
    let sample_rate = spec.rate;
    let channels = spec.channels.count();

    println!("Original format: {} Hz, {} channels", sample_rate, channels);

    // Convert to mono if stereo
    let mut mono_samples = if channels > 1 {
        audio_samples.chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect::<Vec<f32>>()
    } else {
        audio_samples
    };

    // Resample to 16kHz if needed
    const TARGET_SAMPLE_RATE: u32 = 16000;
    if sample_rate != TARGET_SAMPLE_RATE {
        println!("Resampling from {} Hz to {} Hz...", sample_rate, TARGET_SAMPLE_RATE);

        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };

        let mut resampler = SincFixedIn::<f32>::new(
            TARGET_SAMPLE_RATE as f64 / sample_rate as f64,
            2.0,
            params,
            mono_samples.len(),
            1,
        )?;

        let waves_in = vec![mono_samples];
        let waves_out = resampler.process(&waves_in, None)?;
        mono_samples = waves_out[0].clone();
    }

    println!("Converted to 16kHz mono, {} samples", mono_samples.len());

    Ok(mono_samples)
}

fn load_config() -> Result<UserConfig, Box<dyn std::error::Error>> {
    let config_path = PathBuf::from("user.config");
    if config_path.exists() {
        let config_str = fs::read_to_string(config_path)?;
        let config: UserConfig = serde_json::from_str(&config_str)?;
        Ok(config)
    } else {
        Err("No configuration found. Run 'config' command first.".into())
    }
}

fn save_config(config: &UserConfig) -> Result<(), Box<dyn std::error::Error>> {
    let config_str = serde_json::to_string_pretty(config)?;
    fs::write("user.config", config_str)?;
    Ok(())
}

fn configure_microphone() -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();

    // Get all input devices
    let devices: Vec<_> = host.input_devices()?
        .filter_map(|device| {
            device.name().ok().map(|name| (name, device))
        })
        .collect();

    if devices.is_empty() {
        return Err("No input devices found".into());
    }

    println!("Available microphones:\n");

    let device_names: Vec<String> = devices.iter()
        .map(|(name, _)| name.clone())
        .collect();

    // Use dialoguer to create an interactive selection menu
    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select a microphone")
        .items(&device_names)
        .default(0)
        .interact()?;

    let selected_device = &device_names[selection];

    // Ask for gain setting
    use dialoguer::Input;
    let gain: f32 = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Enter gain multiplier (1.0 = normal, 2.0 = double volume, etc.)")
        .default(1.0)
        .interact()?;

    // Save to config
    let config = UserConfig {
        selected_device: selected_device.clone(),
        gain,
    };
    save_config(&config)?;

    println!("\nMicrophone '{}' with gain {} has been saved to user.config", selected_device, gain);

    Ok(())
}

fn record_audio(output: Option<String>, duration: Option<u64>, gain_override: Option<f32>, model_path: PathBuf, script_path: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    // If script is provided, show the next unrecorded portion
    if let Some(script) = &script_path {
        let next_portion = get_next_unrecorded_portion(script)?;
        if let Some(text) = next_portion {
            println!("\n================================================================================");
            println!("SCRIPT PROMPT:");
            println!("================================================================================");
            println!("{}", text);
            println!("================================================================================\n");
            println!("Press Enter when ready to record...");

            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
        } else {
            println!("\nAll portions of the script have been recorded!");
            return Ok(());
        }
    }
    // Create recordings directory if it doesn't exist
    fs::create_dir_all("recordings")?;

    // Generate filename
    let filename = match output {
        Some(name) => name,
        None => {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();
            format!("recording_{}.wav", timestamp)
        }
    };

    let output_path = PathBuf::from("recordings").join(&filename);
    println!("Recording to: {:?}", output_path);

    // Get the audio host
    let host = cpal::default_host();

    // Try to load configured device, otherwise use default
    let (device, gain) = match load_config() {
        Ok(config) => {
            let gain = gain_override.unwrap_or(config.gain);
            // Find device by name
            let device = host.input_devices()?
                .find(|d| d.name().ok().as_ref() == Some(&config.selected_device))
                .ok_or_else(|| format!("Configured device '{}' not found. Run 'config' command again.", config.selected_device))?;
            (device, gain)
        }
        Err(_) => {
            println!("No configuration found, using default input device.");
            println!("Run 'cord config' to select a specific microphone.\n");
            let device = host.default_input_device()
                .ok_or("No input device available")?;
            let gain = gain_override.unwrap_or(1.0);
            (device, gain)
        }
    };

    println!("Using input device: {}", device.name()?);
    println!("Using gain: {}", gain);

    // Get the default input config
    let config = device.default_input_config()?;
    println!("Default input config: {:?}", config);

    // Create WAV writer
    let spec = WavSpec {
        channels: config.channels(),
        sample_rate: config.sample_rate().0,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let writer = Arc::new(Mutex::new(WavWriter::create(&output_path, spec)?));
    let writer_clone = writer.clone();

    // Build the input stream
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut writer = writer_clone.lock().unwrap();
                for &sample in data {
                    let amplified = (sample * gain).clamp(-1.0, 1.0);
                    let amplitude = (amplified * i16::MAX as f32) as i16;
                    writer.write_sample(amplitude).unwrap();
                }
            },
            |err| eprintln!("Error in audio stream: {}", err),
            None,
        )?,
        cpal::SampleFormat::I16 => {
            let gain_clone = gain;
            device.build_input_stream(
                &config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let mut writer = writer_clone.lock().unwrap();
                    for &sample in data {
                        let amplified = ((sample as f32) * gain_clone).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                        writer.write_sample(amplified).unwrap();
                    }
                },
                |err| eprintln!("Error in audio stream: {}", err),
                None,
            )?
        }
        cpal::SampleFormat::U16 => {
            let gain_clone = gain;
            device.build_input_stream(
                &config.into(),
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let mut writer = writer_clone.lock().unwrap();
                    for &sample in data {
                        let normalized = (sample as i32 - 32768) as f32;
                        let amplified = (normalized * gain_clone).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                        writer.write_sample(amplified).unwrap();
                    }
                },
                |err| eprintln!("Error in audio stream: {}", err),
                None,
            )?
        }
        _ => return Err("Unsupported sample format".into()),
    };

    // Start recording
    stream.play()?;

    if let Some(secs) = duration {
        println!("Recording for {} seconds...", secs);
        std::thread::sleep(std::time::Duration::from_secs(secs));
    } else {
        println!("Recording... Press Space, Enter, or Ctrl+C to stop.");

        // Enable raw mode to capture key presses
        terminal::enable_raw_mode()?;

        let should_stop = Arc::new(AtomicBool::new(false));
        let should_stop_clone = should_stop.clone();

        // Set up Ctrl+C handler
        ctrlc::set_handler(move || {
            should_stop_clone.store(true, Ordering::SeqCst);
        }).expect("Error setting Ctrl-C handler");

        // Wait for Space, Enter, or Ctrl+C
        loop {
            if should_stop.load(Ordering::SeqCst) {
                break;
            }

            if event::poll(std::time::Duration::from_millis(100))? {
                if let Event::Key(KeyEvent { code, .. }) = event::read()? {
                    match code {
                        KeyCode::Char(' ') | KeyCode::Enter | KeyCode::Char('c') => {
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Disable raw mode
        terminal::disable_raw_mode()?;
    }

    // Stop and finalize
    drop(stream);

    // Extract the writer from the Arc<Mutex<>> and finalize
    let writer = Arc::try_unwrap(writer)
        .map_err(|_| "Failed to unwrap Arc")?
        .into_inner()
        .unwrap();
    writer.finalize()?;

    println!("\nRecording saved to: {:?}", output_path);

    // Automatically transcribe the recording
    println!("\nStarting automatic transcription...");
    parse_audio(&output_path, &model_path)?;

    // If script was provided, regenerate the compiled file
    if let Some(script) = script_path {
        println!("\nUpdating script compilation...");
        match_script(&script)?;
    }

    Ok(())
}

fn get_next_unrecorded_portion(script_path: &PathBuf) -> Result<Option<String>, Box<dyn std::error::Error>> {
    // Read the script file
    let script_content = fs::read_to_string(script_path)?;
    let script_words: Vec<String> = script_content
        .split_whitespace()
        .map(|s| s.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|s| !s.is_empty())
        .collect();

    // Check if compiled file exists
    let script_name = script_path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("script");
    let compiled_path = PathBuf::from("scripts/temp").join(format!("{}_compiled.txt", script_name));

    let mut covered_positions = Vec::new();

    if compiled_path.exists() {
        // Parse the compiled file to find covered positions
        let compiled_content = fs::read_to_string(&compiled_path)?;
        for line in compiled_content.lines() {
            if line.trim().starts_with("Script Position: ") {
                if let Some(pos_str) = line.trim().strip_prefix("Script Position: ") {
                    if let Ok(pos) = pos_str.parse::<usize>() {
                        covered_positions.push(pos);
                    }
                }
            }
        }
    }

    covered_positions.sort();

    // Find the first gap in coverage
    let mut next_position = 0;
    for &pos in &covered_positions {
        if pos > next_position {
            break;
        }
        // Estimate that each recording covers about 10-20 words
        next_position = pos + 10;
    }

    // If we've covered the whole script
    if next_position >= script_words.len() {
        return Ok(None);
    }

    // Split script into sentences
    let sentences: Vec<&str> = script_content
        .split(|c| c == '.' || c == '!' || c == '?')
        .filter(|s| !s.trim().is_empty())
        .collect();

    // Calculate approximate word position for each sentence
    let mut sentence_positions = Vec::new();
    let mut current_pos = 0;
    for sentence in &sentences {
        let word_count = sentence.split_whitespace().count();
        sentence_positions.push((current_pos, sentence.trim()));
        current_pos += word_count;
    }

    // Find the first sentence that starts at or after next_position
    let mut selected_sentence = sentences.first().map(|s| s.trim()).unwrap_or("");
    for (pos, sentence) in &sentence_positions {
        if *pos >= next_position {
            selected_sentence = sentence;
            break;
        }
    }

    // Get 1-3 sentences starting from the selected one
    let start_idx = sentence_positions.iter()
        .position(|(_, s)| *s == selected_sentence)
        .unwrap_or(0);
    let end_idx = (start_idx + 3).min(sentences.len());

    let mut next_text = String::new();
    for i in start_idx..end_idx {
        if i > start_idx {
            next_text.push(' ');
        }
        next_text.push_str(sentences[i].trim());
        // Add back the punctuation
        if i < sentences.len() - 1 {
            next_text.push('.');
        }
    }

    Ok(Some(next_text))
}

fn match_script(script_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading script from: {:?}", script_path);

    // Read the script file
    let script_content = fs::read_to_string(script_path)?;

    // Split script into words/phrases for matching
    let script_words: Vec<String> = script_content
        .split_whitespace()
        .map(|s| s.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|s| !s.is_empty())
        .collect();

    println!("Script contains {} words", script_words.len());

    // Read all transcription files from recordings directory
    let recordings_dir = PathBuf::from("recordings");
    if !recordings_dir.exists() {
        return Err("Recordings directory not found".into());
    }

    let mut transcriptions = Vec::new();
    for entry in fs::read_dir(recordings_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("txt")
            && path.file_name().and_then(|s| s.to_str()).map(|s| s.starts_with("transcription_")).unwrap_or(false) {
            let content = fs::read_to_string(&path)?;
            transcriptions.push((path.clone(), content));
        }
    }

    println!("Found {} transcription files", transcriptions.len());

    // Match script words to transcriptions
    let mut matches: Vec<(usize, PathBuf, String, f32)> = Vec::new(); // (position, file, matched_text, score)

    for (trans_path, trans_content) in &transcriptions {
        let trans_words: Vec<String> = trans_content
            .split_whitespace()
            .map(|s| s.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|s| !s.is_empty())
            .collect();

        // Find best matching subsequence in this transcription
        let mut best_match_score = 0.0;
        let mut best_script_pos = 0;

        // Try to find where this transcription best matches in the script
        for script_start in 0..script_words.len() {
            for trans_start in 0..trans_words.len() {
                let mut matches_count = 0;

                for i in 0..trans_words.len().min(script_words.len() - script_start) {
                    if trans_start + i >= trans_words.len() {
                        break;
                    }
                    if script_start + i >= script_words.len() {
                        break;
                    }

                    if trans_words[trans_start + i] == script_words[script_start + i] {
                        matches_count += 1;
                    } else if matches_count > 0 {
                        break; // Stop on first mismatch after matches
                    }
                }

                if matches_count > 0 {
                    let score = matches_count as f32 / trans_words.len() as f32;
                    if score > best_match_score {
                        best_match_score = score;
                        best_script_pos = script_start;
                    }
                }
            }
        }

        if best_match_score > 0.1 { // Only include matches above 10% threshold
            matches.push((best_script_pos, trans_path.clone(), trans_content.clone(), best_match_score));
        }
    }

    // Sort matches by script position
    matches.sort_by_key(|(pos, _, _, _)| *pos);

    println!("Found {} matching recordings", matches.len());

    // Create output directory
    fs::create_dir_all("scripts/temp")?;

    // Generate output filename
    let script_name = script_path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("script");
    let output_path = PathBuf::from("scripts/temp").join(format!("{}_compiled.txt", script_name));

    // Write compilation list
    let mut output = String::new();
    output.push_str(&format!("Script Compilation for: {}\n", script_name));
    output.push_str(&format!("Generated: {}\n\n", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs()));
    output.push_str("================================================================================\n\n");

    for (i, (pos, path, content, score)) in matches.iter().enumerate() {
        output.push_str(&format!("{}. Recording: {}\n", i + 1, path.file_name().unwrap().to_str().unwrap()));
        output.push_str(&format!("   Script Position: {}\n", pos));
        output.push_str(&format!("   Match Score: {:.2}%\n", score * 100.0));
        output.push_str(&format!("   Transcription: {}\n\n", content.trim()));
    }

    fs::write(&output_path, output)?;
    println!("\nCompilation saved to: {:?}", output_path);

    Ok(())
}