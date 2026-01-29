<!-- Marvin4000 - Real-time Audio Transcription & Translation -->
<!-- © 2025 XOREngine (WallyByte) -->
<!-- https://github.com/XOREngine/marvin4000 -->

# Marvin4000

> Transcripción y traducción de audio en tiempo real con SeamlessM4T end‑to‑end (STT / S2ST)

[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/GPU-Accelerated-green)](https://developer.nvidia.com/cuda-toolkit)

**🌐 Idiomas:** [English](README.md) | [Español](README.es.md)

<br>

**Marvin4000** captura, transcribe y traduce audio del sistema en tiempo real usando hardware local.

## ✅ Características actuales

* **STT en tiempo real** (speech → text) con parciales y finales
* **S2ST end‑to‑end** (speech → speech) con salida de audio TTS
* **Segmentación por VAD** (WebRTC) y corte por silencio o timeout
* **Filtro por nivel de audio** (RMS) para evitar ruido muy bajo
* **Salida de audio configurable** (dispositivo y sample rate)
* **GPU/CPU automático** (CUDA si está disponible)

<br>

> ⚠️ **IMPORTANTE:**
>
> * Este proyecto está pensado para **Linux** con **PulseAudio** o **pipewire‑pulse** (usa `parec` y `pactl`).
> * En **Windows** no hay soporte nativo; la captura de audio requeriría una alternativa a `parec` que entregue audio `float32`.

<br>

## 🚀 Instalación y uso

### Requisitos

* Linux con PulseAudio o pipewire‑pulse
* CUDA **obligatorio** (se usa automáticamente si está disponible)

```bash
sudo apt install python3-pip pulseaudio-utils ffmpeg
pip install -r requirements.txt
```

> El `requirements.txt` incluye **solo** dependencias mínimas para ejecutar STT/S2ST con SeamlessM4T.

### Ejecución básica

```bash
# 1. Reproducir algún contenido con audio en tu sistema
vlc video_ejemplo.mp4
# ffmpeg.ffplay -nodisp -autoexit -ss 1 example.mp3
# o reproducir audio desde el navegador, etc.

# 2. Detectar dispositivos de audio válidos
python detect_audio_devices.py
# Ejemplo salida:
# $ python marvin4000_seam.py --audio-device "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"

# 3. Iniciar transcripción (STT)
python marvin4000_seam.py --audio-device "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor" --mode stt --src-lang eng

# 4. Iniciar traducción con voz (S2ST + TTS)
python marvin4000_seam.py --audio-device "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor" --mode s2st --src-lang eng --tgt-lang spa --output-device "default"
```

> 💡 **Tip PulseAudio:** crea un sink virtual y usa su `.monitor` como `--audio-device`. Para evitar realimentación, envía el TTS a tu salida real con `--output-device`.

**Cómo crear un sink virtual (PulseAudio):**

1. Crear el sink y darle un nombre:

```bash
pactl load-module module-null-sink sink_name=virtual_sink sink_properties=device.description=VirtualSink
```

2. Verifica el `.monitor` resultante para usarlo como `--audio-device`:

```bash
pactl list short sources | grep virtual_sink
# Ejemplo: virtual_sink.monitor
```

3. Si quieres enviar el audio del sistema al sink virtual, usa tu herramienta de audio (por ejemplo, `pavucontrol`) y selecciona **VirtualSink** como salida para la app que deseas capturar.

4. Para eliminar el sink cuando termines:

```bash
pactl unload-module module-null-sink
```

### Configuración de idiomas

Marvin4000 utiliza SeamlessM4T end‑to‑end para transcripción y traducción entre más de 100 idiomas. Soporta aplicaciones multilingües en tiempo real.

**Nota sobre el TTS de SeamlessM4T v2:**

* Para T2ST/S2ST, el modelo genera **unidades de audio discretas** y luego un **vocoder** las convierte en onda de audio.
* La versión v2 usa la arquitectura **UnitY2**, con mejoras en **calidad** y **latencia** de la generación de voz.
* SeamlessM4T no expone selección de voz; si necesitas voces específicas, usa un TTS externo y reemplaza el audio generado.

Referencia: https://github.com/facebookresearch/seamless_communication

### Parámetros CLI disponibles

* `--audio-device` (requerido): fuente monitor de PulseAudio
* `--mode`: `stt` o `s2st`
* `--src-lang`: idioma fuente (ej. `eng`)
* `--tgt-lang`: idioma destino (requerido en `s2st`)
* `--output-device`: dispositivo de salida para TTS (sounddevice)
* `--output-sr`: sample rate de salida (por defecto 16000)

<br>

## 🔬 Detalles técnicos actuales

* **Captura**: `parec` (PulseAudio / pipewire‑pulse) en `float32`
* **Procesamiento**: conversión a mono 16 kHz y VAD con WebRTC
* **Segmentación**: por silencio o timeout, con parciales en STT
* **Salida TTS**: reproducción con `sounddevice` en el dispositivo elegido

### Parámetros internos relevantes

> **Nota:** Para bajar latencia puedes reducir `num_beams` o `max_new_tokens` en el código.

```python
TIMEOUT_SEC = 12.0           # Tiempo máximo sin flush
MIN_SEGMENT_SEC = 0.5        # Mínima duración aceptada de segmento
MIN_PARTIAL_WORDS = 5        # Palabras mínimas para mostrar parcial
SILENCE_SEC = 0.8            # Silencio requerido para segmentar
VAD_SILENCE_DURATION_SEC = 1.2
MIN_CUT_DURATION_SEC = 2.5
AUDIO_RMS_THRESHOLD = 0.0025 # Nivel mínimo de volumen aceptado
```

## 📜 Modelos y licencias

* Código Marvin4000: [MIT](LICENSE)
* SeamlessM4T: [CC-BY-NC 4.0](https://github.com/facebookresearch/seamless_communication/blob/main/LICENSE) (Meta AI)

<br>

## 🙏 Agradecimientos y referencias

### Modelos y librerías usadas

* [Meta SeamlessM4T](https://github.com/facebookresearch/seamless_communication)
* [WebRTC VAD](https://webrtc.org/)

### Inspiración técnica y papers

* [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp) – ejecución tiempo real
* [guillaumekln/faster-whisper](https://github.com/guillaumekln/faster-whisper) – buffering eficiente (referencia)
* [SeamlessM4T: Massively Multilingual & Multimodal Machine Translation](https://arxiv.org/abs/2308.11596)
* [Efficient Low-Bit Quantization of Transformer-Based Language Models](https://arxiv.org/abs/2305.12889)

---

<br>

Este proyecto está pensado como una base flexible. Si quieres modificarlo, usarlo de forma creativa, mejorarlo o simplemente adaptarlo a tus necesidades...

> 💪 **Hazlo.**

Si además compartes mejoras o nos mencionas como referencia, será siempre bien recibido 🙌😜.

<br>

© [XOREngine](https://xorengine.com) · Compromiso open source

<br>

<!-- keywords: seamlessM4T, realtime transcription, translation, streaming audio, cuda, multilingual, vad, low latency, STT, S2ST, TTS -->

---

## 🧪 Ejemplos de comandos (marvin4000_seam.py)

```bash
# STT básico (inglés → texto)
python marvin4000_seam.py --audio-device "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor" --mode stt --src-lang eng

# STT con idioma fuente español
python marvin4000_seam.py --audio-device "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor" --mode stt --src-lang spa

# S2ST inglés → español con salida por dispositivo default
python marvin4000_seam.py --audio-device "virtual_sink.monitor" --mode s2st --src-lang eng --tgt-lang spa --output-device "default"

# S2ST francés → inglés con sample rate de salida personalizado
python marvin4000_seam.py --audio-device "virtual_sink.monitor" --mode s2st --src-lang fra --tgt-lang eng --output-device "default" --output-sr 22050

# S2ST alemán → italiano con dispositivo de salida específico (ID de sounddevice)
python marvin4000_seam.py --audio-device "virtual_sink.monitor" --mode s2st --src-lang deu --tgt-lang ita --output-device 3
```