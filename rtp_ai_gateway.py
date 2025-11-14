# voice_assistant_working.py
import socket
import struct
import time
import random
import threading
from queue import Queue, Full, Empty
import numpy as np
import audioop
import whisper
import os
import warnings
import requests
import tempfile
import wave
import json
import subprocess
import sys
import traceback
import signal

# Optional Windows SAPI
try:
    import win32com.client
    import pythoncom
    HAVE_SAPI = True
except Exception:
    HAVE_SAPI = False

# Edge TTS optional
try:
    import edge_tts
    import asyncio
    HAVE_EDGE_TTS = True
except Exception:
    HAVE_EDGE_TTS = False

warnings.filterwarnings("ignore", category=DeprecationWarning)

# -----------------------------
# Configuration
# -----------------------------
LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 4000
RECV_SAMPLE_RATE = 8000
RECOG_RATE = 16000

WHISPER_TASK = "transcribe"
WHISPER_MODEL = "base"

# Ollama / local LLM streaming API
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MAX_HISTORY_MESSAGES = 10

SYSTEM_PROMPT = (
    "You are a helpful voice assistant having a real-time conversation. "
    "Listen carefully to what the user says and respond naturally and appropriately. "
    "Give clear, concise answers (2-3 sentences unless more detail is needed). "
    "Be conversational, friendly, and helpful. "
    "If you don't understand something, politely ask for clarification. "
    "Stay on topic and remember the conversation context."
)

SPEAK_ASSISTANT = True

# Edge TTS toggle (we'll fallback to SAPI first on Windows)
USE_EDGE_TTS = True
EDGE_TTS_VOICE = "en-US-AriaNeural"

# Fallbacks
USE_ESPEAK = True
USE_MAC_SAY = True

# VAD params
VAD_FRAME_MS = 20
VAD_FRAME_SAMPLES = RECV_SAMPLE_RATE * VAD_FRAME_MS // 1000
MIN_SPEECH_FRAMES = 5
SPEECH_THRESHOLD = 0.012
SILENCE_FRAMES = 150  # 3 seconds of silence (150 frames * 20ms = 3000ms)
MIN_AUDIO_LENGTH_SEC = 0.5

# -----------------------------
# Global state
# -----------------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LISTEN_IP, LISTEN_PORT))
sock.settimeout(0.5)

remote_addr = None
peer_fmt_pt = None
rtp_senders = {}
injection_queues = {}

speech_buffer = []
speech_frame_count = 0
silence_frame_count = 0
is_speaking = False
is_processing = False  # Flag to prevent concurrent processing

transcribe_lock = threading.Lock()
conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
history_lock = threading.Lock()

ai_speaking = threading.Event()
ai_should_stop = threading.Event()
audio_finish_event = threading.Event()

# Thread registry
_thread_registry = {}

def register_thread(name, thread_obj):
    _thread_registry[name] = thread_obj

# graceful shutdown
shutting_down = threading.Event()
def handle_shutdown(sig, frame):
    print("🛑 Shutdown requested")
    shutting_down.set()
    try:
        sock.close()
    except Exception:
        pass

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# -----------------------------
# Load Whisper
# -----------------------------
print(f"Loading Whisper model ({WHISPER_MODEL})...")
# If you have GPU and want to use it, replace device="cpu" with device="cuda"
whisper_model = whisper.load_model(WHISPER_MODEL, device="cpu")
print("✅ Whisper ready!")

def debug_startup_questions():
    print("\n================ DEBUG STARTUP =================")
    print("Before audio starts, let's test the AI.\n")
    test_questions = [
        "Hello, who are you?",
        "What can you help me with?",
        "What is 2 + 2?",
    ]
    for q in test_questions:
        try:
            print(f"\nUSER (debug): {q}")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}
            ]
            response = call_ollama(messages)
            print(f"AI (debug): {response}")
        except Exception as e:
            print("🔥 Debug startup call failed:", e)
            traceback.print_exc()
    print("=============== DEBUG COMPLETE ===============\n")

# -----------------------------
# Audio helpers
# -----------------------------
def decode_rtp(pkt: bytes):
    try:
        if len(pkt) < 12:
            return None, None, None
        b0 = pkt[0]
        version = (b0 >> 6) & 0x03
        if version != 2:
            return None, None, None
        pt = pkt[1] & 0x7F
        csrc_count = b0 & 0x0F
        header_len = 12 + 4 * csrc_count
        payload = pkt[header_len:]
        if not payload:
            return None, None, None
        if pt == 0:
            # PCMU -> linear
            pcm8k = audioop.ulaw2lin(payload, 2)
        elif pt == 8:
            pcm8k = audioop.alaw2lin(payload, 2)
        else:
            return None, None, None
        return pt, pcm8k, payload
    except Exception as e:
        print("❌ decode_rtp error:", e)
        traceback.print_exc()
        return None, None, None

def calculate_rms(pcm_bytes):
    if not pcm_bytes or len(pcm_bytes) < 2:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples ** 2)))

def resample_8k_to_16k(pcm8k_bytes):
    if not pcm8k_bytes:
        return b""
    samples = np.frombuffer(pcm8k_bytes, dtype=np.int16)
    resampled = np.repeat(samples, 2)
    return resampled.tobytes()

def resample_any_to_8k(pcm16_bytes, src_rate):
    if not pcm16_bytes:
        return b""
    samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
    if src_rate == 8000:
        return pcm16_bytes
    n_out = max(1, int(len(samples) * 8000.0 / src_rate))
    xp = np.linspace(0, len(samples) - 1, len(samples))
    xnew = np.linspace(0, len(samples) - 1, n_out)
    resampled = np.interp(xnew, xp, samples)
    return np.clip(np.round(resampled), -32768, 32767).astype(np.int16).tobytes()

# -----------------------------
# VAD and transcription pipeline
# -----------------------------
def process_user_speech(audio_data):
    """Process user speech: transcribe, get AI response, speak it back"""
    global is_processing, remote_addr, peer_fmt_pt
    
    try:
        audio_duration = len(audio_data) / (RECV_SAMPLE_RATE * 2)
        print(f"\n📝 Processing {audio_duration:.1f}s of user speech...")
        
        # Transcribe with Whisper
        audio_16k = resample_8k_to_16k(audio_data)
        samples = np.frombuffer(audio_16k, dtype=np.int16).astype(np.float32) / 32768.0
        
        start_time = time.time()
        with transcribe_lock:
            result = whisper_model.transcribe(
                samples,
                task=WHISPER_TASK,
                fp16=False,
                language="en",
                beam_size=1,
                best_of=1,
                temperature=0.0
            )
        elapsed = time.time() - start_time
        
        text = result.get("text", "").strip()
        if not text:
            print(f"⚠️  No speech detected ({elapsed:.1f}s)")
            return
        
        print(f"✅ TRANSCRIBED: \"{text}\" ({elapsed:.1f}s)")
        
        # Add to conversation history
        with history_lock:
            conversation_history.append({"role": "user", "content": text})
            prune_history()
            snapshot = list(conversation_history)
        
        # Get AI response
        print("🤔 Getting AI response...")
        start_time = time.time()
        assistant_reply = call_ollama(snapshot)
        elapsed = time.time() - start_time
        
        if not assistant_reply:
            print("⚠️ Empty response from AI")
            return
        
        print(f"🤖 AI: \"{assistant_reply}\" ({elapsed:.1f}s)")
        
        # Add AI response to history
        with history_lock:
            conversation_history.append({"role": "assistant", "content": assistant_reply})
            prune_history()
        
        # Convert to speech and play
        if SPEAK_ASSISTANT and remote_addr:
            ai_speaking.set()
            ai_should_stop.clear()
            
            try:
                pcm, rate = text_to_speech(assistant_reply)
                
                if pcm:
                    frames = enqueue_rtp_audio_from_pcm16(remote_addr, pcm, rate, peer_fmt_pt)
                    
                    if frames > 0:
                        speech_duration = frames * 0.02
                        wait_time = speech_duration + 0.5
                        print(f"🔊 Speaking... ({speech_duration:.1f}s)")
                        
                        start_wait = time.time()
                        while time.time() - start_wait < wait_time:
                            if ai_should_stop.is_set():
                                print("⏸️  Speech interrupted")
                                break
                            
                            try:
                                if remote_addr in injection_queues:
                                    qsize = injection_queues[remote_addr].qsize()
                                    if qsize == 0 and time.time() - start_wait > 0.5:
                                        break
                            except Exception:
                                pass
                            time.sleep(0.1)
                        
                        print("✅ Speech complete\n")
                else:
                    print("❌ TTS failed")
            except Exception as e:
                print("❌ TTS error:", e)
                traceback.print_exc()
            finally:
                ai_speaking.clear()
                ai_should_stop.clear()
                
    except Exception as e:
        print(f"❌ Error processing user speech: {e}")
        traceback.print_exc()
    finally:
        is_processing = False


def process_audio_vad(pcm8k_bytes):
    global speech_buffer, speech_frame_count, silence_frame_count, is_speaking, is_processing
    
    try:
        if len(pcm8k_bytes) < VAD_FRAME_SAMPLES * 2:
            return
        
        # Don't process while AI is speaking or already processing
        if ai_speaking.is_set() or is_processing:
            return

        for i in range(0, len(pcm8k_bytes), VAD_FRAME_SAMPLES * 2):
            frame = pcm8k_bytes[i:i + VAD_FRAME_SAMPLES * 2]
            if len(frame) < VAD_FRAME_SAMPLES * 2:
                break
            rms = calculate_rms(frame)
            
            if rms > SPEECH_THRESHOLD:
                # User is speaking
                if not is_speaking:
                    is_speaking = True
                    speech_frame_count = 0
                    silence_frame_count = 0
                    speech_buffer = []
                    print(f"\n🎤 User speaking...")
                    if ai_speaking.is_set():
                        ai_should_stop.set()
                        print("⏸️  [AI interrupted]")
                speech_buffer.append(frame)
                speech_frame_count += 1
                silence_frame_count = 0
            else:
                # Silence detected
                if is_speaking:
                    silence_frame_count += 1
                    speech_buffer.append(frame)
                    
                    # User stopped speaking (3 seconds silence) - process immediately
                    if silence_frame_count >= SILENCE_FRAMES:
                        if speech_frame_count >= MIN_SPEECH_FRAMES:
                            audio_data = b''.join(speech_buffer)
                            audio_duration = len(audio_data) / (RECV_SAMPLE_RATE * 2)
                            if audio_duration >= MIN_AUDIO_LENGTH_SEC:
                                is_processing = True
                                is_speaking = False
                                speech_buffer = []
                                speech_frame_count = 0
                                silence_frame_count = 0
                                
                                # Process in a new thread so VAD continues
                                threading.Thread(
                                    target=process_user_speech,
                                    args=(audio_data,),
                                    daemon=True
                                ).start()
                            else:
                                print(f"⏭️  Speech too short ({audio_duration:.1f}s)")
                                is_speaking = False
                                speech_buffer = []
                                speech_frame_count = 0
                                silence_frame_count = 0
                        else:
                            is_speaking = False
                            speech_buffer = []
                            speech_frame_count = 0
                            silence_frame_count = 0
    except Exception as e:
        print("❌ process_audio_vad error:", e)
        traceback.print_exc()

def transcribe_worker():
    """Deprecated - kept for compatibility"""
    pass

# -----------------------------
# TTS helpers
# -----------------------------
def ffmpeg_available():
    return subprocess.run(["where" if os.name == "nt" else "which", "ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

def convert_to_wav(input_path, output_path):
    """Try to convert input audio (mp3 etc) to output WAV using ffmpeg if available."""
    if not ffmpeg_available():
        return False
    try:
        cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", output_path]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print("❌ ffmpeg conversion failed:", e)
        return False

def tts_edge_to_pcm16(text):
    if not HAVE_EDGE_TTS:
        return None, None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        # call Edge TTS using whatever signature is available; do not assume output_format param exists
        async def generate():
            # The Communicate signature differs across versions; use simple form and save
            comm = edge_tts.Communicate(text, EDGE_TTS_VOICE)
            await comm.save(tmp_path)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generate())
        loop.close()

        # Check WAV header
        try:
            with open(tmp_path, "rb") as f:
                hdr = f.read(12)
                if hdr[:4] != b"RIFF":
                    # Not a WAV — try to convert via ffmpeg (e.g., mp3 saved with .wav extension)
                    print("⚠️ Edge returned non-RIFF file; attempting ffmpeg conversion")
                    # write a proper output file and try converting from tmp_path to tmp_wav
                    tmp_wav = tmp_path + ".conv.wav"
                    if convert_to_wav(tmp_path, tmp_wav):
                        with wave.open(tmp_wav, "rb") as wf:
                            frames = wf.readframes(wf.getnframes())
                            sr = wf.getframerate()
                            if wf.getnchannels() == 2:
                                frames = audioop.tomono(frames, wf.getsampwidth(), 0.5, 0.5)
                            if wf.getsampwidth() != 2:
                                frames = audioop.lin2lin(frames, wf.getsampwidth(), 2)
                            try:
                                os.unlink(tmp_wav)
                            except:
                                pass
                            return frames, sr
                    else:
                        print("⚠️ Conversion failed or ffmpeg missing; falling back")
                        return None, None
        except Exception:
            pass

        # Try opening as WAV
        try:
            with wave.open(tmp_path, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                sample_rate = wf.getframerate()
                if wf.getnchannels() == 2:
                    frames = audioop.tomono(frames, wf.getsampwidth(), 0.5, 0.5)
                if wf.getsampwidth() != 2:
                    frames = audioop.lin2lin(frames, wf.getsampwidth(), 2)
                return frames, sample_rate
        except wave.Error as we:
            print(f"❌ Edge TTS produced unreadable WAV: {we}")
            return None, None
    except Exception as e:
        print(f"❌ Edge TTS error: {e}")
        traceback.print_exc()
        return None, None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

def tts_sapi_to_pcm16(text):
    if not HAVE_SAPI:
        return None, None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        # Initialize COM for this thread
        import pythoncom
        pythoncom.CoInitialize()
        try:
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            voice.Rate = 2
            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            stream.Open(tmp_path, 3, False)
            voice.AudioOutputStream = stream
            voice.Speak(text)
            stream.Close()
            with wave.open(tmp_path, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                sample_rate = wf.getframerate()
                if wf.getnchannels() == 2:
                    frames = audioop.tomono(frames, wf.getsampwidth(), 0.5, 0.5)
                if wf.getsampwidth() != 2:
                    frames = audioop.lin2lin(frames, wf.getsampwidth(), 2)
                return frames, sample_rate
        finally:
            pythoncom.CoUninitialize()
    except Exception as e:
        print(f"❌ SAPI TTS error: {e}")
        traceback.print_exc()
        return None, None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

def tts_espeak_to_pcm16(text):
    if USE_ESPEAK:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            subprocess.run(["espeak", "-w", tmp_path, text], check=True)
            with wave.open(tmp_path, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                sample_rate = wf.getframerate()
                if wf.getnchannels() == 2:
                    frames = audioop.tomono(frames, wf.getsampwidth(), 0.5, 0.5)
                if wf.getsampwidth() != 2:
                    frames = audioop.lin2lin(frames, wf.getsampwidth(), 2)
                return frames, sample_rate
        except Exception as e:
            print(f"❌ espeak TTS error: {e}")
            traceback.print_exc()
            return None, None
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    return None, None

def text_to_speech(text):
    try:
        print(f"🔊 [TTS] Request text ({len(text)} chars): {text[:180]}")
        # Prefer SAPI on Windows (reliable)
        if HAVE_SAPI:
            print("🎵 [TTS] Trying Windows SAPI...")
            pcm, rate = tts_sapi_to_pcm16(text)
            if pcm:
                print(f"✅ [TTS] SAPI returned {len(pcm)} bytes @ {rate}Hz")
                return pcm, rate
            print("⚠️ [TTS] SAPI failed, falling through...")
        # Try Edge if available
        if USE_EDGE_TTS and HAVE_EDGE_TTS:
            print("🎵 [TTS] Trying Edge TTS (fallback)...")
            pcm, rate = tts_edge_to_pcm16(text)
            if pcm:
                print(f"✅ [TTS] Edge returned {len(pcm)} bytes @ {rate}Hz")
                return pcm, rate
            print("⚠️ [TTS] Edge failed, falling through...")
        # Fallback espeak
        pcm, rate = tts_espeak_to_pcm16(text)
        if pcm:
            print(f"✅ [TTS] espeak returned {len(pcm)} bytes @ {rate}Hz")
            return pcm, rate
        print("❌ No TTS available")
        return None, None
    except Exception as e:
        print("❌ text_to_speech error:", e)
        traceback.print_exc()
        return None, None

# -----------------------------
# RTP enqueue & sender
# -----------------------------
def enqueue_rtp_audio_from_pcm16(peer_addr, pcm16_bytes, src_rate, pt_hint=0):
    try:
        if not pcm16_bytes or peer_addr is None:
            print("⚠️ Cannot enqueue audio: missing data or address")
            return 0
        print(f"📤 Enqueueing {len(pcm16_bytes)} bytes at {src_rate}Hz for {peer_addr} (pt_hint={pt_hint})")
        pcm8k = resample_any_to_8k(pcm16_bytes, src_rate)
        print(f"   Resampled to 8kHz: {len(pcm8k)} bytes")
        if pt_hint == 8:
            encoded = audioop.lin2alaw(pcm8k, 2)
            pad_byte = b"\xD5"
        else:
            encoded = audioop.lin2ulaw(pcm8k, 2)
            pad_byte = b"\xFF"
        print(f"   Encoded to {'PCMA' if pt_hint == 8 else 'PCMU'}: {len(encoded)} bytes")
        if peer_addr not in injection_queues:
            injection_queues[peer_addr] = Queue(maxsize=1000)
            print(f"   Created new injection queue for {peer_addr}")
        q = injection_queues[peer_addr]
        frame_len = 160
        frame_count = 0
        dropped_count = 0
        for i in range(0, len(encoded), frame_len):
            chunk = encoded[i:i + frame_len]
            if len(chunk) < frame_len:
                chunk += pad_byte * (frame_len - len(chunk))
            try:
                q.put_nowait(chunk)
                frame_count += 1
            except Full:
                dropped_count += 1
        print(f"   ✅ Queued {frame_count} frames ({frame_count * 20}ms)")
        if dropped_count > 0:
            print(f"   ⚠️ Dropped {dropped_count} frames")
        return frame_count
    except Exception as e:
        print("❌ enqueue_rtp_audio_from_pcm16 error:", e)
        traceback.print_exc()
        return 0

def rtp_keepalive_sender(peer_addr, pt_hint=0):
    try:
        if peer_addr not in rtp_senders:
            rtp_senders[peer_addr] = {
                "seq": random.randint(0, 65535),
                "ts": random.randint(0, 2**32 - 1),
                "ssrc": random.randint(0, 2**32 - 1),
                "pt": pt_hint,
                "in_speech": False
            }
        st = rtp_senders[peer_addr]
        seq, ts, ssrc = st["seq"], st["ts"], st["ssrc"]
        pt_to_send = 8 if pt_hint == 8 else 0
        pad_byte = b"\xD5" if pt_to_send == 8 else b"\xFF"
        if peer_addr not in injection_queues:
            injection_queues[peer_addr] = Queue(maxsize=1000)
        q = injection_queues[peer_addr]
        print(f"📡 RTP sender started for {peer_addr} (PT={pt_to_send}) SSRC={ssrc}")
        register_thread(f"rtp_sender_{peer_addr}", threading.current_thread())
        packets_sent = 0
        last_report = time.time()
        silence_count = 0
        while not shutting_down.is_set():
            if ai_should_stop.is_set() and ai_speaking.is_set():
                drained = 0
                try:
                    while not q.empty():
                        q.get_nowait()
                        drained += 1
                except Empty:
                    pass
                if drained > 0:
                    print(f"🗑️ Drained {drained} frames due to interruption")
                    ai_speaking.clear()
                    st["in_speech"] = False
            try:
                payload = q.get(timeout=0.02)
                non_silence = True
                silence_count = 0
            except Empty:
                payload = pad_byte * 160
                non_silence = False
                silence_count += 1

            # marker bit
            marker_bit = 0
            if non_silence and not st["in_speech"]:
                marker_bit = 0x80
                st["in_speech"] = True
                print("🎯 Starting new audio sequence (marker bit set)")
            elif not non_silence and silence_count > 25:
                st["in_speech"] = False

            first_byte = (2 << 6)
            second_byte = marker_bit | (pt_to_send & 0x7F)
            header = struct.pack("!BBHII", first_byte, second_byte, seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc)
            try:
                sock.sendto(header + payload, peer_addr)
                packets_sent += 1
                if non_silence and (packets_sent % 50 == 0 or packets_sent < 5):
                    try:
                        qsize = q.qsize()
                    except Exception:
                        qsize = -1
                    print(f"📨 RTP: sent non-silence packet #{packets_sent}, queue={qsize}")
            except Exception as e:
                if packets_sent % 1000 == 0:
                    print("❌ Send error:", e)
                    traceback.print_exc()
            seq = (seq + 1) & 0xFFFF
            ts = (ts + 160) & 0xFFFFFFFF
            st["seq"], st["ts"] = seq, ts
            if time.time() - last_report >= 10.0:
                try:
                    qsize = q.qsize()
                except Exception:
                    qsize = -1
                print(f"📊 RTP: {packets_sent} pkts, queue: {qsize}")
                last_report = time.time()
            time.sleep(0.02)
    except Exception as e:
        print("❌ rtp_keepalive_sender error:", e)
        traceback.print_exc()

# -----------------------------
# Ollama call (streaming)
# -----------------------------
def call_ollama(messages, timeout=60):
    # Convert messages to a single prompt string
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    
    prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 150,
            "num_ctx": 2048,
            "num_thread": 4
        }
    }
    assembled = ""
    try:
        print("🌐 [DEBUG] Calling Ollama (streaming)...")
        r = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=timeout)
        r.raise_for_status()
        print("🌐 [DEBUG] Connected to Ollama stream, reading...")
        for raw in r.iter_lines():
            if shutting_down.is_set():
                break
            if not raw:
                continue
            try:
                line = raw.decode() if isinstance(raw, bytes) else raw
                j = json.loads(line)
                chunk = j.get("response", "")
                if chunk:
                    assembled += chunk
                    print(f"🧩 [Ollama chunk] {chunk}", end="", flush=True)
                if j.get("done", False):
                    break
            except json.JSONDecodeError:
                try:
                    line = raw.decode() if isinstance(raw, bytes) else raw
                    print("\n⚠️ [Ollama] Non-JSON chunk:", line)
                except Exception:
                    pass
        print("")  # newline after streaming chunks
    except requests.exceptions.RequestException as e:
        print("❌ Ollama request error:", e)
        traceback.print_exc()
        return ""
    except Exception as e:
        print("❌ Ollama error:", e)
        traceback.print_exc()
        return ""
    return assembled.strip()

def prune_history():
    with history_lock:
        non_system = conversation_history[1:]
        if len(non_system) > MAX_HISTORY_MESSAGES:
            del conversation_history[1:1 + len(non_system) - MAX_HISTORY_MESSAGES]

# -----------------------------
# Greeting handler
# -----------------------------
def send_greeting():
    """Send AI-generated greeting when user connects"""
    global remote_addr, peer_fmt_pt
    
    try:
        print("👋 Generating AI greeting...")
        greeting_prompt = "[SYSTEM: User just connected. Generate a warm, brief greeting (1-2 sentences) introducing yourself and offering to help.]"
        temp_messages = list(conversation_history) + [{"role": "user", "content": greeting_prompt}]
        
        assistant_reply = call_ollama(temp_messages)
        
        if not assistant_reply:
            assistant_reply = "Hello! I'm your AI assistant. How can I help you today?"
        
        print(f"🤖 Greeting: \"{assistant_reply}\"")
        
        # Speak the greeting
        if SPEAK_ASSISTANT and remote_addr:
            ai_speaking.set()
            try:
                pcm, rate = text_to_speech(assistant_reply)
                if pcm:
                    frames = enqueue_rtp_audio_from_pcm16(remote_addr, pcm, rate, peer_fmt_pt)
                    if frames > 0:
                        speech_duration = frames * 0.02
                        print(f"🔊 Speaking greeting... ({speech_duration:.1f}s)")
                        time.sleep(speech_duration + 0.5)
                        print("✅ Greeting complete\n")
            except Exception as e:
                print(f"❌ Greeting error: {e}")
            finally:
                ai_speaking.clear()
    except Exception as e:
        print(f"❌ Error sending greeting: {e}")
        traceback.print_exc()


def llm_worker():
    """Deprecated - processing is now synchronous"""
    pass

# -----------------------------
# Thread health monitor
# -----------------------------
def thread_health_monitor():
    print("[DEBUG] Thread health monitor started")
    register_thread("thread_health_monitor", threading.current_thread())
    while not shutting_down.is_set():
        try:
            statuses = []
            for name, t in list(_thread_registry.items()):
                statuses.append(f"{name}: {'alive' if t.is_alive() else 'dead'}")
            print("[THREADS] " + " | ".join(statuses))
        except Exception as e:
            print("❌ thread_health_monitor error:", e)
            traceback.print_exc()
        time.sleep(10)

# -----------------------------
# Main loop
# -----------------------------
def main():
    global remote_addr, peer_fmt_pt
    print("=" * 60)
    print("🎧 VOICE ASSISTANT - SIMPLIFIED")
    print(f"📡 Listening on {LISTEN_IP}:{LISTEN_PORT}")
    print("=" * 60)

    debug_startup_questions()

    t_monitor = threading.Thread(target=thread_health_monitor, daemon=True, name="thread_health_monitor")
    t_monitor.start()
    register_thread("thread_health_monitor", t_monitor)

    audio_accumulator = bytearray()
    while not shutting_down.is_set():
        try:
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            continue
        except OSError:
            break
        except Exception as e:
            print("❌ Socket recv error:", e)
            traceback.print_exc()
            continue

        if remote_addr is None or addr != remote_addr:
            remote_addr = addr
            pt_detected = data[1] & 0x7F if len(data) >= 12 else 0
            peer_fmt_pt = pt_detected if pt_detected in (0, 8) else 0
            print("\n" + "=" * 60)
            print("📞 CALL CONNECTED")
            print(f"   Address: {remote_addr}")
            print(f"   PT detected: {peer_fmt_pt}")
            print("=" * 60 + "\n")
            try:
                threading.Thread(target=rtp_keepalive_sender, args=(remote_addr, peer_fmt_pt), daemon=True).start()
            except Exception as e:
                print("❌ Failed to start rtp_keepalive_sender:", e)
                traceback.print_exc()
            
            # Send AI-generated greeting
            time.sleep(0.3)
            if SPEAK_ASSISTANT:
                threading.Thread(target=send_greeting, daemon=True).start()

        pt, pcm8k, _ = decode_rtp(data)
        if pcm8k is None:
            continue
        try:
            audio_accumulator.extend(pcm8k)
            if len(audio_accumulator) >= VAD_FRAME_SAMPLES * 2 * 2:
                process_audio_vad(bytes(audio_accumulator))
                audio_accumulator = audio_accumulator[-(VAD_FRAME_SAMPLES * 2):]
        except Exception as e:
            print("❌ error handling audio_accumulator:", e)
            traceback.print_exc()

    print("🧾 main exiting, waiting for threads to finish...")
    time.sleep(0.5)
    print("✅ shutdown complete")

if __name__ == "__main__":
    main()
