#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Marvin4000 - Real-time Audio Transcription & Translation
# © 2025 XOREngine (WallyByte)
# https://github.com/XOREngine/marvin4000

from __future__ import annotations
import argparse
import queue
import signal
import subprocess as sp
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import math
import numpy as np
from scipy.signal import resample_poly
import torch
import webrtcvad
import sounddevice as sd

from transformers import AutoProcessor, SeamlessM4Tv2Model

CACHE_DIR          = Path("./models_cache")
SEAMLESS_MODEL     = "facebook/seamless-m4t-v2-large"
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF           = DEVICE == "cuda"

CHUNK_SEC          = 1.5
NATIVE_SR          = 48000
TARGET_SR          = 16000
CHANNELS           = 2
QUEUE_MAX          = 128
TIMEOUT_SEC        = 12.0
SILENCE_SEC        = 0.8
MIN_SEGMENT_SEC    = 0.5
MIN_WORDS_FOR_SENT = 4
MIN_PARTIAL_WORDS  = 5
PAREC_LATENCY_MS   = 20

# Audio processing thresholds
AUDIO_RMS_THRESHOLD = 0.0025  # Minimum RMS level to process audio

# VAD silence detection parameters
VAD_FRAME_MS       = 30    # VAD frame duration in milliseconds
VAD_SILENCE_DURATION_SEC = 1.2  # Required silence duration for detection
MIN_CUT_DURATION_SEC = 2.5      # Minimum audio duration before cut


# Langs
# SeamlessM4T: eng, spa, fra, deu, ita, por, rus, tur, pol, ces, nld, ukr, kor, jpn, zho, arb, etc.
SRC_LANG = "eng"
TGT_LANG = "spa"

MODE = "stt"  # stt | s2st

OUTPUT_SR = 16000

GREEN              = "\033[32m"
RESET              = "\033[0m"

# Lock to serialize GPU usage
gpu_lock = threading.Lock()

def log(msg: str, color: str = "") -> None:
    ts = datetime.now().isoformat(timespec="milliseconds")
    print(f"[{ts}] {color}{msg}{RESET}", flush=True)

def to_mono_16k(x: np.ndarray, sr_orig: int) -> np.ndarray:
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr_orig == TARGET_SR:
        return x.astype("float32")
    g = math.gcd(sr_orig, TARGET_SR)
    return resample_poly(x, TARGET_SR // g, sr_orig // g).astype("float32")

class AudioPlayer(threading.Thread):
    def __init__(self, output_device: Optional[str], sample_rate: int):
        super().__init__(daemon=True)
        self.output_device = output_device
        self.sample_rate = sample_rate
        self.q: queue.Queue[tuple[np.ndarray, int]] = queue.Queue(maxsize=8)
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()
        try:
            self.q.put_nowait((np.array([], dtype=np.float32), self.sample_rate))
        except queue.Full:
            pass

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        try:
            self.q.put((audio, sample_rate), timeout=1)
        except queue.Full:
            log("Audio playback queue full, dropping segment")

    def run(self) -> None:
        log("Audio player ready")
        with sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            device=self.output_device,
            dtype="float32",
        ) as stream:
            while not self._stop.is_set():
                try:
                    audio, sr = self.q.get(timeout=0.2)
                except queue.Empty:
                    continue

                if audio.size == 0:
                    continue

                audio = np.asarray(audio, dtype=np.float32)
                if audio.ndim > 1:
                    audio = audio.reshape(-1)

                if sr != self.sample_rate:
                    g = math.gcd(sr, self.sample_rate)
                    audio = resample_poly(audio, self.sample_rate // g, sr // g).astype(np.float32)

                stream.write(audio)

# Producer
class AudioProducer(threading.Thread):
    def __init__(self, q: queue.Queue[np.ndarray], audio_device: str, latency_ms: int):
        super().__init__(daemon=True)
        self.q = q
        self.audio_device = audio_device
        self.latency_ms = latency_ms
        self._stop = threading.Event()
    def stop(self):
        self._stop.set()
    def run(self):
        frames = int(max(self.latency_ms / 1000.0, 0.05) * NATIVE_SR)
        chunk_bytes = frames * CHANNELS * 4
        cmd = [
            "parec", f"--device={self.audio_device}", "--format=float32le",
            f"--rate={NATIVE_SR}", f"--channels={CHANNELS}", f"--latency-msec={self.latency_ms}"
        ]
        log(f"Producer start: {' '.join(cmd)}")
        proc = sp.Popen(cmd, stdout=sp.PIPE, bufsize=chunk_bytes*4)
        try:
            while not self._stop.is_set():
                buf = proc.stdout.read(chunk_bytes)
                if not buf:
                    log("No audio data from parec. Verifica el dispositivo de entrada y que haya audio reproduciéndose.")
                    break
                audio = np.frombuffer(buf, dtype="<f4").reshape(-1, CHANNELS)
                mono  = to_mono_16k(audio, NATIVE_SR)
                try:
                    self.q.put(mono, timeout=1)
                except queue.Full:
                    try:
                        _ = self.q.get_nowait()
                        self.q.put_nowait(mono)
                    except queue.Empty:
                        pass
                    except queue.Full:
                        pass
                    log("Producer queue full, dropped oldest chunk")
        finally:
            proc.terminate()
            log("Producer stopped")


# Transcriber and Translator
class Transcriber(threading.Thread):
    def __init__(self, q: queue.Queue[np.ndarray], mode: str, src_lang: str, tgt_lang: Optional[str], output_device: Optional[str], output_sr: int):
        super().__init__(daemon=True)
        self.q            = q
        self.audio_buffer : List[np.ndarray] = []
        self.last_flush   = time.time()
        self.vad          = webrtcvad.Vad(1)

        self.last_partial_text = ""
        self.mode = mode
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.output_sr = output_sr

        self.player: Optional[AudioPlayer] = None
        if self.mode == "s2st":
            self.player = AudioPlayer(output_device=output_device, sample_rate=output_sr)
            self.player.start()

        # SEAMLESSM4T end-to-end
        log(f"Loading SeamlessM4T model: {SEAMLESS_MODEL}")
        self.processor = AutoProcessor.from_pretrained(SEAMLESS_MODEL)
        self.model = SeamlessM4Tv2Model.from_pretrained(SEAMLESS_MODEL).to(DEVICE)
        log("SeamlessM4T model loaded: 16bit")


    def _has_sentence_end(self, txt: str) -> bool:
        if not txt:
            return False
        ends         = txt.rstrip().endswith((".", "?", "!"))
        starts_upper = txt[0].isupper()
        return ends and starts_upper and len(txt.split()) >= MIN_WORDS_FOR_SENT

    
    def _prepare_audio(self, audio: np.ndarray) -> Optional[np.ndarray]:
        dur = len(audio) / TARGET_SR
        if dur < MIN_SEGMENT_SEC:
            return None

        norm = audio / (np.max(np.abs(audio)) + 1e-8)
        return norm

    def _extract_speech(self, output) -> Optional[np.ndarray]:
        for attr in ("speech", "audio", "waveform", "waveforms"):
            if hasattr(output, attr):
                val = getattr(output, attr)
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    val = val[0]
                if torch.is_tensor(val):
                    return val.detach().cpu().float().numpy()
                if isinstance(val, np.ndarray):
                    return val.astype(np.float32)
        return None

    def _extract_speech_sr(self, output) -> Optional[int]:
        for attr in ("speech_sample_rate", "audio_sampling_rate", "sampling_rate"):
            if hasattr(output, attr):
                val = getattr(output, attr)
                if isinstance(val, int):
                    return val
        return None

    def _seamless_stt(self, audio: np.ndarray) -> Optional[str]:
        norm = self._prepare_audio(audio)
        if norm is None:
            return None

        inputs = self.processor(audio=norm, sampling_rate=TARGET_SR, return_tensors="pt")
        inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

        with gpu_lock, torch.no_grad():
            gen = self.model.generate(
                **inputs,
                tgt_lang=self.tgt_lang or self.src_lang,
                max_new_tokens=256,
                num_beams=3,
                do_sample=False,
            )

        txt = self.processor.batch_decode(gen.sequences, skip_special_tokens=True)[0]
        return txt.strip()

    def _seamless_s2st(self, audio: np.ndarray) -> tuple[Optional[str], Optional[np.ndarray], Optional[int]]:
        norm = self._prepare_audio(audio)
        if norm is None:
            return None, None, None

        inputs = self.processor(audio=norm, sampling_rate=TARGET_SR, return_tensors="pt")
        inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

        with gpu_lock, torch.no_grad():
            gen = self.model.generate(
                **inputs,
                tgt_lang=self.tgt_lang,
                generate_speech=True,
                max_new_tokens=256,
                num_beams=3,
                do_sample=False,
            )

        gen_text = gen
        gen_speech = gen
        if isinstance(gen, tuple):
            if len(gen) > 0:
                gen_text = gen[0]
            if len(gen) > 1:
                gen_speech = gen[1]

        seqs = gen_text.sequences if hasattr(gen_text, "sequences") else gen_text
        txt = ""
        try:
            if torch.is_tensor(seqs):
                if seqs.dtype in (torch.int32, torch.int64, torch.long, torch.int16, torch.int8, torch.uint8):
                    txt = self.processor.batch_decode(seqs, skip_special_tokens=True)[0]
            elif isinstance(seqs, (list, tuple)) and seqs and isinstance(seqs[0], (list, tuple, np.ndarray)):
                txt = self.processor.batch_decode(seqs, skip_special_tokens=True)[0]
        except Exception:
            txt = ""
        speech = self._extract_speech(gen_speech) or self._extract_speech(gen_text)
        speech_sr = self._extract_speech_sr(gen_speech) or self._extract_speech_sr(gen_text)
        return txt.strip(), speech, speech_sr


    def _find_silence_split(self, audio: np.ndarray) -> Optional[int]:
        audio_i16 = (audio * 32767).astype(np.int16)
        frame_sz = int(TARGET_SR * VAD_FRAME_MS / 1000)
        req = int(VAD_SILENCE_DURATION_SEC * 1000 / VAD_FRAME_MS)
        
        cnt = 0
        last_speech_idx = 0
        
        for i in range(0, len(audio_i16) - frame_sz, frame_sz):
            frame = audio_i16[i:i+frame_sz]
            if len(frame) < frame_sz:
                continue
                
            frame_bytes = frame.tobytes()
            
            if not self.vad.is_speech(frame_bytes, TARGET_SR):
                cnt += 1
            else:
                cnt = 0
                last_speech_idx = i
                
            if cnt >= req:
                silence_duration = (i - last_speech_idx) / TARGET_SR
                if silence_duration > SILENCE_SEC:
                    cut = last_speech_idx + frame_sz * 2
                    if cut / TARGET_SR >= MIN_CUT_DURATION_SEC:
                        return cut
                        
        return None

    def _post_process_asr(self, txt: str) -> str:
        # txt = txt.replace("?", "'").replace("?","\"").replace("?","\"")
        # txt = re.sub(r"[^\x20-\x7EáéíóúüÁÉÍÓÚÜñÑ¿¡]", "", txt)   # remove weird chars
        return txt
    
    def run(self):
        log("Transcriber ready")
        self.last_flush = time.time()
        last_empty_log = 0.0
        while True:
            try:
                chunk = self.q.get(timeout=0.5)
            except queue.Empty:
                now = time.time()
                if now - last_empty_log >= 5.0:
                    log("Sin audio en la cola. Revisa el routing al sink virtual y que haya audio reproduciéndose.")
                    last_empty_log = now
                continue

            # Calculate RMS - objective volume measure
            chunk_rms = np.sqrt(np.mean(np.square(chunk)))
            if chunk_rms < AUDIO_RMS_THRESHOLD:
                log(f"Audio descartado por nivel bajo (RMS={chunk_rms:.6f}). Puedes bajar --rms-threshold.")
                
                # Process accumulated audio as final segment
                if self.audio_buffer:
                    audio_cat = np.concatenate(self.audio_buffer)
                    if len(audio_cat) / TARGET_SR >= MIN_SEGMENT_SEC:
                        if self.mode == "stt":
                            txt_raw = self._seamless_stt(audio_cat) or ""
                            txt = self._post_process_asr(txt_raw)
                            if txt.strip():
                                log(f"[FINAL-SILENCE] {txt.upper()}", GREEN)
                        else:
                            txt, speech, speech_sr = self._seamless_s2st(audio_cat)
                            if txt:
                                log(f"[FINAL-SILENCE-TR] {txt.upper()}", GREEN)
                            if speech is not None and self.player:
                                self.player.play(speech, speech_sr or self.output_sr)
                    
                    self.audio_buffer.clear()
                
                self.last_flush = time.time()  # Reset timeout
                continue   # Skip current chunk - low volume
            
            self.audio_buffer.append(chunk)

            audio_cat = np.concatenate(self.audio_buffer)
            txt = ""
            
            # In S2ST mode, skip expensive partial transcription on every chunk
            # Only transcribe when we're about to flush
            now       = time.time()
            split_idx = self._find_silence_split(audio_cat)
            timed_out = (now - self.last_flush) >= TIMEOUT_SEC
            
            # Only run STT for partials in stt mode
            if self.mode == "stt":
                txt_raw = self._seamless_stt(audio_cat) or ""
                txt = self._post_process_asr(txt_raw)

                if len(txt.split()) >= MIN_PARTIAL_WORDS and txt != self.last_partial_text:
                    log(f"[PARTIAL] {txt}")
                    self.last_partial_text = txt
                
                end_sent = self._has_sentence_end(txt)
            else:
                end_sent = True  # In S2ST always ready when silence detected

            if (split_idx is not None and end_sent) or timed_out:
                cut = split_idx if split_idx is not None else len(audio_cat)
                out, rem = audio_cat[:cut], audio_cat[cut:]

                # Use MIN_SEGMENT_SEC to decide
                if len(out) / TARGET_SR >= MIN_SEGMENT_SEC:
                    # Process final block
                    if self.mode == "stt":
                        log(f"[FINAL] {txt.upper()}", GREEN)
                    else:
                        txt, speech, speech_sr = self._seamless_s2st(out)
                        if txt:
                            log(f"[FINAL-TR] {txt.upper()}", GREEN)
                        if speech is not None and self.player:
                            self.player.play(speech, speech_sr or self.output_sr)
                    
                    # Reset buffers, preserve remainder, clear translation cache
                    self.audio_buffer = [rem] if rem.size else []
                    self.last_flush = now
                else:
                    # Fragment too short: DON'T discard it, leave buffers intact
                    log(f"Fragment of {len(out)/TARGET_SR:.2f}s (< {MIN_SEGMENT_SEC}s), waiting for more before flush")


# Main
def main():
    global MODE, SRC_LANG, TGT_LANG, OUTPUT_SR, AUDIO_RMS_THRESHOLD, PAREC_LATENCY_MS
    parser = argparse.ArgumentParser(description="Marvin4000 - Real-time speech with SeamlessM4T end-to-end")
    parser.add_argument("--audio-device", required=True, help="Input monitor device name (PulseAudio, e.g. 'alsa_output.device.monitor')")
    parser.add_argument("--mode", choices=["stt", "s2st"], default="stt", help="Mode: stt (speech->text) or s2st (speech->speech)")
    parser.add_argument("--src-lang", default="eng", help="Source language (SeamlessM4T, e.g. eng/spa)")
    parser.add_argument("--tgt-lang", default=None, help="Target language (required for s2st; optional for stt)")
    parser.add_argument("--output-device", default=None, help="Output device for TTS (sounddevice name or id)")
    parser.add_argument("--output-sr", type=int, default=OUTPUT_SR, help="Output sample rate for TTS")
    parser.add_argument("--rms-threshold", type=float, default=AUDIO_RMS_THRESHOLD, help="Minimum RMS level to process audio")
    parser.add_argument("--parec-latency-ms", type=int, default=PAREC_LATENCY_MS, help="parec latency in milliseconds (reduce if no data)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA es obligatorio para ejecutar Marvin4000. Asegúrate de tener una GPU NVIDIA con drivers y CUDA instalados.")

    try:
        sink_list = sp.run(["pactl", "list", "short", "sinks"], capture_output=True, text=True, check=False)
        if sink_list.returncode != 0 or "virtual_sink" not in sink_list.stdout:
            parser.error("No se detectó un sink virtual activo (virtual_sink). Crea uno con: pactl load-module module-null-sink sink_name=virtual_sink")
    except FileNotFoundError:
        parser.error("No se encontró pactl. Instala PulseAudio/pipewire-pulse para usar sinks virtuales.")

    if args.mode == "s2st" and not args.tgt_lang:
        parser.error("--tgt-lang es requerido en modo s2st")

    # Override global variables
    MODE = args.mode
    SRC_LANG = args.src_lang
    TGT_LANG = args.tgt_lang
    OUTPUT_SR = args.output_sr
    AUDIO_RMS_THRESHOLD = args.rms_threshold
    PAREC_LATENCY_MS = args.parec_latency_ms

    q_audio = queue.Queue(maxsize=QUEUE_MAX)
    
    # Initialize transcriber blocking until models load
    transcriber = Transcriber(q_audio, mode=MODE, src_lang=SRC_LANG, tgt_lang=TGT_LANG, output_device=args.output_device, output_sr=OUTPUT_SR)
    log("Models loaded, starting audio capture")
    prod = AudioProducer(q_audio, args.audio_device, PAREC_LATENCY_MS)

    transcriber.start()
    prod.start()

    def stop_all(*_):
        prod.stop(); prod.join(timeout=1)
        if transcriber.player:
            transcriber.player.stop()
        log("Shutting down")
        raise SystemExit

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)
    signal.pause()

if __name__ == "__main__":
    main()