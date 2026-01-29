<!-- Marvin4000 - Real-time Audio Transcription & Translation -->
<!-- © 2025 XOREngine (WallyByte) -->
<!-- https://github.com/XOREngine/marvin4000 -->

# Marvin4000

> Real-time audio transcription and translation using SeamlessM4T end‑to‑end (STT / S2ST)

[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/GPU-Accelerated-green)](https://developer.nvidia.com/cuda-toolkit)

**🌐 Languages:** [English](README.md) | [Español](README.es.md)

<br>

**Marvin4000** captures, transcribes, and translates system audio in real-time using local hardware.

<br>

> ⚠️ **IMPORTANT:**
>
> * If you're on **Windows**, audio capture must be manually implemented using an alternative to `parec` that provides system audio data in `float32` format.

<br>

## 📊 Proven Performance

| GPU & Models Used                                                | Latency (s) | WER       | BLEU-1/4/Corpus | VRAM        |
| ---------------------------------------------------------------- | ----------- | --------- | --------------- | ----------- |
| RTX 4060 Ti 16GB<br>seamless-m4t-v2-large (STT/S2ST) | 2-3     | 6 % | 74/39/52    | 11.4 GB |

#### Test Corpus

* **Audio**: 25 random audiobook fragments from [LibriSpeech](https://www.openslr.org/12) (avg: 5 min/fragment)
* **Reference Transcription**: Official LibriSpeech transcriptions
* **Reference Translation**: Generated with Claude & GPT and manually reviewed (English → Spanish)
* **Total Evaluated**: ~120 minutes of audio

#### Metrics Calculation

* **WER**: Calculated with [jiwer](https://github.com/jitsi/jiwer), normalized for punctuation
* **BLEU**: Corpus-level implementation with lowercase tokenization, n-gram clipping and brevity penalty
* **BLEU-1/4/Corpus**: 1-gram / 4-gram precision / full corpus score
* **Latency**: Measured under real conditions with RTX 4060 Ti 16GB and RTX 2060 6GB

#### Limitations

While reference translations are high quality, we acknowledge they are not equivalent to professional human translations. However, they provide a consistent standard for comparing system performance, following methodologies similar to those employed in evaluations like [FLEURS](https://arxiv.org/abs/2205.12446) and [CoVoST 2](https://arxiv.org/abs/2007.10310).

<br>

## 🚀 Installation and Usage

### Requirements

```bash
sudo apt install python3-pip pulseaudio-utils ffmpeg
git clone https://github.com/XOREngine/marvin4000.git
cd marvin4000
pip install -r requirements.txt
```

### Basic Execution

```bash
# 1. Play some audio content on your system
vlc example_video.mp4
# ffmpeg.ffplay -nodisp -autoexit -ss 1 example.mp3
# or play audio from browser, etc.

# 2. Detect valid audio devices
python detect_audio_devices.py
# Example output:
# $ python marvin4000_seam.py --audio-device "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"

# 3. Start transcription (STT)
python marvin4000_seam.py --audio-device "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor" --mode stt --src-lang eng

# 4. Start speech-to-speech translation (S2ST + TTS)
python marvin4000_seam.py --audio-device "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor" --mode s2st --src-lang eng --tgt-lang spa --output-device "default"
```

> 💡 **PulseAudio tip:** create a virtual sink and use its `.monitor` as `--audio-device`. To avoid feedback, route TTS to your real output with `--output-device`.

### Language Configuration

Marvin4000 uses SeamlessM4T end‑to‑end for transcription and translation between 100+ languages. Supports real-time multilingual applications.

<br>

## 🔬 Technical Architecture

* **Threading Separation**: Audio capture | SeamlessM4T | TTS. 68% latency reduction
* **Int8 Quantization**: bits-and-bytes implementation for models
* **Intelligent VAD**: WebRTC + conservative segmentation (1.2s minimum silence) + linguistic validation
* **Memory Efficient**: Circular buffer and VAD-based segmentation
* **Hybrid Latency**: Progressive partials (2-3s perceived) in STT mode
* **Adaptive Segmentation**: Avoids <0.5s fragments, 2.5s minimum cuts
* **Controlled Decoding**: `task` and `tgt_lang` for STT/S2ST control

<br>

### Adjustable Configuration Parameters

> **Note:** If you experience too much latency, you can reduce `num_beams` or shorten `max_new_tokens`. This will make inferences faster at the cost of slight quality loss.

**Segmentation and Flow:**

```python
TIMEOUT_SEC = 12.0           # Maximum time without flush
MIN_SEGMENT_SEC = 0.5        # Minimum accepted segment duration
MIN_PARTIAL_WORDS = 5        # Minimum words to show partial
SILENCE_SEC = 0.8            # Silence required for segmentation
VAD_SILENCE_DURATION_SEC = 1.2
MIN_CUT_DURATION_SEC = 2.5
AUDIO_RMS_THRESHOLD = 0.0025 # Minimum accepted volume level
```

**SeamlessM4T Inference (STT/S2ST):**

```python
gen = self.model.generate(
    **inputs,
    tgt_lang="spa",
    task="s2st",             # or "transcribe" for STT
    generate_speech=True,
    max_new_tokens=256,
    num_beams=3,
    do_sample=False,
)
```

### Optimizations for High-End Hardware

For GPUs with >20GB VRAM (RTX 4090, A40, A100), **CUDA streams** can be implemented for SeamlessM4T parallelization:

```python
# Suggested modifications for high-end hardware:
audio_lock = threading.Lock()
tts_lock = threading.Lock()

stream_audio = torch.cuda.Stream()
stream_tts = torch.cuda.Stream()
# Estimated potential improvement: +15-25% throughput
```

<br>

## 📜 Models and Licenses

* Marvin4000 Code: [MIT](LICENSE)
* SeamlessM4T: [CC-BY-NC 4.0](https://github.com/facebookresearch/seamless_communication/blob/main/LICENSE) (Meta AI)

<br>

## 🙏 Acknowledgments and References

### Models and Libraries Used

* [Meta SeamlessM4T](https://github.com/facebookresearch/seamless_communication)
* [WebRTC VAD](https://webrtc.org/)

### Technical Inspiration and Papers

* [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp) – real-time execution
* [TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) – quantization
* [guillaumekln/faster-whisper](https://github.com/guillaumekln/faster-whisper) – efficient buffering
* [snakers4/silero-vad](https://github.com/snakers4/silero-vad) – optimized VAD
* [SeamlessM4T: Massively Multilingual & Multimodal Machine Translation](https://arxiv.org/abs/2308.11596)
* [Efficient Low-Bit Quantization of Transformer-Based Language Models](https://arxiv.org/abs/2305.12889)

---

<br>

This project is designed as a flexible foundation. If you want to modify it, use it creatively, improve it, or simply adapt it to your needs...

> 💪 **Go for it.**

If you also share improvements or mention us as a reference, it will always be welcome 🙌😜.

<br>

© [XOREngine](https://xorengine.com) · Open source commitment

<br>

<!-- keywords: seamlessM4T, realtime transcription, translation, streaming audio, cuda, multilingual, vad, low latency, STT, S2ST, TTS -->