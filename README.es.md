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

<br>

> ⚠️ **IMPORTANTE:**
>
> * Si estás en **Windows**, la captura de audio debe ser implementada manualmente mediante una alternativa a `parec` que proporcione datos de audio del sistema en formato `float32`.

<br>

## 📊 Rendimiento probado

| GPU & Modelos usados                                                | Latencia (s) | WER       | BLEU-1/4/Corpus | VRAM        |
| ---------------------------------------------------------------- | ----------- | --------- | --------------- | ----------- |
| RTX 4060 Ti 16GB<br>seamless-m4t-v2-large (STT/S2ST) | 2-3     | 6 % | 74/39/52    | 11.4 GB |

#### Corpus de prueba

* **Audio**: 25 fragmentos aleatorios de audiolibros de [LibriSpeech](https://www.openslr.org/12) (media: 5 min/fragmento)
* **Transcripción de referencia**: Transcripciones oficiales de LibriSpeech
* **Traducción de referencia**: Generada con Claude & GPT y revisada manualmente (Inglés → Español)
* **Total evaluado**: \~120 minutos de audio

#### Cálculo de métricas

* **WER**: Calculado con [jiwer](https://github.com/jitsi/jiwer), normalizado para puntuación
* **BLEU**: Implementación corpus-level con tokenización lowercase, clipping de n-gramas y brevity penalty
* **BLEU-1/4/Corpus**: Precisión 1-grama / 4-grama / score corpus completo
* **Latencia**: Medida en condiciones reales con RTX 4060 Ti 16GB y RTX 2060 6GB

#### Limitaciones

Aunque las traducciones de referencia son de alta calidad, reconocemos que no son equivalentes a traducciones humanas profesionales. Sin embargo, proveen un estándar consistente para comparar el rendimiento del sistema, siguiendo metodologías similares a las empleadas en evaluaciones como [FLEURS](https://arxiv.org/abs/2205.12446) y [CoVoST 2](https://arxiv.org/abs/2007.10310).

<br>

## 🚀 Instalación y uso

### Requisitos

```bash
sudo apt install python3-pip pulseaudio-utils ffmpeg
git clone https://github.com/XOREngine/marvin4000.git
cd marvin4000
pip install -r requirements.txt
```

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

<br>

## 🔬 Arquitectura técnica

* **Separación de hilos (Threading)**: Captura de audio | SeamlessM4T | TTS. Reducción 68% latencia
* **Cuantización Int8**: Implementación bits-and-bytes para los modelos
* **VAD inteligente**: WebRTC + segmentación conservadora (1.2s silencio mínimo) + validación lingüística
* **Memoria eficiente**: Buffer circular y segmentación por VAD
* **Latencia híbrida**: Parciales progresivos (2-3s percibida) en modo STT
* **Segmentación adaptativa**: Evita fragmentos <0.5s, cortes mínimos 2.5s
* **Decodificación controlada**: `task` y `tgt_lang` para controlar STT y S2ST

<br>

### Parámetros de configuración ajustables

> **Nota:** Si experimentas demasiada latencia, puedes reducir `num_beams` o acortar `max_new_tokens`. Esto hará las inferencias más rápidas a costa de una leve pérdida de calidad.

**Segmentación y flujo:**

```python
TIMEOUT_SEC = 12.0           # Tiempo máximo sin flush
MIN_SEGMENT_SEC = 0.5        # Mínima duración aceptada de segmento
MIN_PARTIAL_WORDS = 5        # Palabras mínimas para mostrar parcial
SILENCE_SEC = 0.8            # Silencio requerido para segmentar
VAD_SILENCE_DURATION_SEC = 1.2
MIN_CUT_DURATION_SEC = 2.5
AUDIO_RMS_THRESHOLD = 0.0025 # Nivel mínimo de volumen aceptado
```

**Inferencia SeamlessM4T (STT/S2ST):**

```python
gen = self.model.generate(
    **inputs,
    tgt_lang="spa",
    task="s2st",             # o "transcribe" para STT
    generate_speech=True,
    max_new_tokens=256,
    num_beams=3,
    do_sample=False,
)
```

### Optimizaciones para hardware potente

Para GPUs con >20GB VRAM (RTX 4090, A40, A100), se pueden implementar **CUDA streams** para paralelización en SeamlessM4T:

```python
# Modificaciones sugeridas para hardware potente:
audio_lock = threading.Lock()
tts_lock = threading.Lock()

stream_audio = torch.cuda.Stream()
stream_tts = torch.cuda.Stream()
# Potencial mejora estimada: +15-25% throughput
```

<br>

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
* [TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) – cuantización
* [guillaumekln/faster-whisper](https://github.com/guillaumekln/faster-whisper) – buffering eficiente
* [snakers4/silero-vad](https://github.com/snakers4/silero-vad) – VAD optimizado
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