# voice_assistant_fixed_full.py
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
from collections import deque

# Optional Windows SAPI
try:
    import win32com.client
    HAVE_SAPI = True
except Exception:
    HAVE_SAPI = False

# OpenAI client optional
try:
    from openai import OpenAI
    HAVE_OPENAI = True
except Exception:
    HAVE_OPENAI = False

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
LISTEN_IP = '0.0.0.0'
LISTEN_PORT = 4000
RECV_SAMPLE_RATE = 8000
RECOG_RATE = 16000

WHISPER_TASK = "transcribe"
WHISPER_MODEL = "base"

# Use the chat endpoint (streaming)
OLLAMA_MODEL = "phi3:mini"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
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

# OpenAI TTS toggles (optional)
USE_OPENAI_TTS = False
OPENAI_API_KEY = "your-openai-api-key-here"
OPENAI_TTS_MODEL = "tts-1"
OPENAI_TTS_VOICE = "nova"

# Edge TTS (free) toggle
USE_EDGE_TTS = True
EDGE_TTS_VOICE = "en-US-AriaNeural"

# Fallbacks
USE_ESPEAK = True
USE_MAC_SAY = True

# Initialize OpenAI client if requested
openai_client = None
if HAVE_OPENAI and USE_OPENAI_TTS and OPENAI_API_KEY != "your-openai-api-key-here":
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI TTS initialized")
    except Exception as e:
        print("⚠️ OpenAI init failed:", e)
        openai_client = None

# Check for edge_tts
if USE_EDGE_TTS and HAVE_EDGE_TTS:
    print("✅ Edge TTS available")
elif USE_EDGE_TTS:
    print("⚠️ Edge TTS requested but not installed. pip install edge-tts")

# Check espeak / say presence
HAVE_ESPEAK = False
HAVE_SAY = False
if USE_ESPEAK:
    try:
        subprocess.run(["espeak", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        HAVE_ESPEAK = True
        print("✅ espeak available (fallback TTS)")
    except Exception:
        HAVE_ESPEAK = False
if sys.platform == "darwin" and USE_MAC_SAY:
    HAVE_SAY = True
    print("✅ macOS say available (fallback TTS)")

# VAD params
VAD_FRAME_MS = 20
VAD_FRAME_SAMPLES = RECV_SAMPLE_RATE * VAD_FRAME_MS // 1000
MIN_SPEECH_FRAMES = 5
SPEECH_THRESHOLD = 0.012
SILENCE_FRAMES = 15
MIN_AUDIO_LENGTH_SEC = 0.4
TRANSCRIBE_COOLDOWN = 0.3

# Silence timeout configuration
USER_SILENCE_TIMEOUT = 8.0  # seconds of total silence before prompting user
SILENCE_PROMPT_MESSAGES = [
    "Hello? Are you still there? I can't hear you.",
    "I'm having trouble hearing you. Are you there?",
    "Hey, I didn't catch that. Can you speak up?",
]

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
last_transcribe_time = 0
last_user_activity_time = time.time()
silence_prompt_index = 0
last_silence_prompt_time = 0

transcribe_queue = Queue(maxsize=8)
transcribe_lock = threading.Lock()
ollama_queue = Queue(maxsize=16)
conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
history_lock = threading.Lock()

ai_speaking = threading.Event()
ai_should_stop = threading.Event()
audio_finish_event = threading.Event()

# -----------------------------
# Load Whisper
# -----------------------------
print(f"Loading Whisper model ({WHISPER_MODEL})...")
whisper_model = whisper.load_model(WHISPER_MODEL, device="cpu")
print("✅ Whisper ready!")

# -----------------------------
# Audio helpers
# -----------------------------
def decode_rtp(pkt: bytes):
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
        pcm8k = audioop.ulaw2lin(payload, 2)
    elif pt == 8:
        pcm8k = audioop.alaw2lin(payload, 2)
    else:
        return None, None, None
    return pt, pcm8k, payload

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
def process_audio_vad(pcm8k_bytes):
    global speech_buffer, speech_frame_count, silence_frame_count, is_speaking, last_transcribe_time
    global last_user_activity_time, silence_prompt_index, last_silence_prompt_time
    if len(pcm8k_bytes) < VAD_FRAME_SAMPLES * 2:
        return
    current_time = time.time()
    if ai_speaking.is_set():
        return
    
    # Check for prolonged user silence and prompt if needed
    silence_duration = current_time - last_user_activity_time
    if (silence_duration > USER_SILENCE_TIMEOUT and 
        not is_speaking and 
        current_time - last_silence_prompt_time > USER_SILENCE_TIMEOUT * 1.5 and
        remote_addr is not None):
        print(f"⏰ User silent for {silence_duration:.1f}s, prompting...")
        prompt_msg = SILENCE_PROMPT_MESSAGES[silence_prompt_index % len(SILENCE_PROMPT_MESSAGES)]
        silence_prompt_index += 1
        last_silence_prompt_time = current_time
        # Queue the prompt message
        try:
            ollama_queue.put_nowait(f"[SYSTEM: User has been silent. Say: '{prompt_msg}']")
        except Full:
            pass
    
    for i in range(0, len(pcm8k_bytes), VAD_FRAME_SAMPLES * 2):
        frame = pcm8k_bytes[i:i + VAD_FRAME_SAMPLES * 2]
        if len(frame) < VAD_FRAME_SAMPLES * 2:
            break
        rms = calculate_rms(frame)
        if rms > SPEECH_THRESHOLD:
            if not is_speaking:
                is_speaking = True
                speech_frame_count = 0
                silence_frame_count = 0
                speech_buffer = []
                last_user_activity_time = current_time  # Update activity time
                print("\n🎤 User speaking...")
                if ai_speaking.is_set():
                    ai_should_stop.set()
                    print("⏸️  [AI interrupted by user]")
            speech_buffer.append(frame)
            speech_frame_count += 1
            silence_frame_count = 0
        else:
            if is_speaking:
                silence_frame_count += 1
                speech_buffer.append(frame)
                if silence_frame_count >= SILENCE_FRAMES:
                    if speech_frame_count >= MIN_SPEECH_FRAMES:
                        if current_time - last_transcribe_time >= TRANSCRIBE_COOLDOWN:
                            audio_data = b''.join(speech_buffer)
                            audio_duration = len(audio_data) / (RECV_SAMPLE_RATE * 2)
                            if audio_duration >= MIN_AUDIO_LENGTH_SEC:
                                print(f"📝 Processing speech ({audio_duration:.1f}s)...")
                                audio_16k = resample_8k_to_16k(audio_data)
                                try:
                                    transcribe_queue.put_nowait(audio_16k)
                                    last_transcribe_time = current_time
                                    last_user_activity_time = current_time  # Update activity time
                                except Full:
                                    print("⚠️  Transcription queue full, skipping...")
                            else:
                                print(f"⏭️  Speech too short ({audio_duration:.1f}s)")
                    is_speaking = False
                    speech_buffer = []
                    speech_frame_count = 0
                    silence_frame_count = 0

def transcribe_worker():
    print("🎯 Transcription worker ready")
    while True:
        try:
            pcm16_bytes = transcribe_queue.get(timeout=0.5)
        except Empty:
            continue
        try:
            samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
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
            if text:
                print(f"\n✅ TRANSCRIBED: \"{text}\"")
                print(f"   (Transcription took {elapsed:.1f}s)")
                try:
                    print(f"📤 [DEBUG] Putting '{text}' into ollama_queue (current size: {ollama_queue.qsize()})...")
                    ollama_queue.put_nowait(text)
                    print(f"✅ [DEBUG] Message queued! New queue size: {ollama_queue.qsize()}")
                except Full:
                    print("⚠️  AI queue full, dropping message")
            else:
                print(f"⚠️  No speech detected in audio ({elapsed:.1f}s)")
        except Exception as e:
            print(f"❌ Transcription error: {e}")
        finally:
            transcribe_queue.task_done()

# -----------------------------
# TTS functions (Edge, OpenAI, SAPI, espeak/say)
# -----------------------------
def tts_edge_to_pcm16(text):
    if not HAVE_EDGE_TTS:
        return None, None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        async def generate():
            communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE, rate="+10%")
            await communicate.save(tmp_path)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generate())
        loop.close()
        with wave.open(tmp_path, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            sample_rate = wf.getframerate()
            if wf.getnchannels() == 2:
                frames = audioop.tomono(frames, wf.getsampwidth(), 0.5, 0.5)
            if wf.getsampwidth() != 2:
                frames = audioop.lin2lin(frames, wf.getsampwidth(), 2)
            return frames, sample_rate
    except Exception as e:
        print(f"❌ Edge TTS error: {e}")
        return None, None
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

def tts_openai_to_pcm16(text):
    if not openai_client:
        return None, None
    try:
        response = openai_client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text,
            response_format="pcm",
            speed=1.1
        )
        pcm_data = response.content
        sample_rate = 24000
        return pcm_data, sample_rate
    except Exception as e:
        print(f"❌ OpenAI TTS error: {e}")
        return None, None

def tts_sapi_to_pcm16(text):
    if not HAVE_SAPI:
        return None, None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
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
            if wf.getnchannels() == 2:
                frames = audioop.tomono(frames, wf.getsampwidth(), 0.5, 0.5)
            if wf.getsampwidth() != 2:
                frames = audioop.lin2lin(frames, wf.getsampwidth(), 2)
            return frames, wf.getframerate()
    except Exception as e:
        print(f"❌ SAPI TTS error: {e}")
        return None, None
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

def tts_espeak_to_pcm16(text):
    if HAVE_ESPEAK:
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
            return None, None
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    elif HAVE_SAY:
        tmp = tempfile.NamedTemporaryFile(suffix=".aiff", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            subprocess.run(["say", "-o", tmp_path, text], check=True)
            wav_tmp = tmp_path + ".wav"
            try:
                subprocess.run(["afconvert", tmp_path, wav_tmp], check=True)
                with wave.open(wav_tmp, 'rb') as wf:
                    frames = wf.readframes(wf.getnframes())
                    sample_rate = wf.getframerate()
                    if wf.getnchannels() == 2:
                        frames = audioop.tomono(frames, wf.getsampwidth(), 0.5, 0.5)
                    if wf.getsampwidth() != 2:
                        frames = audioop.lin2lin(frames, wf.getsampwidth(), 2)
                    return frames, sample_rate
            except Exception:
                print("⚠️ macOS 'say' conversion failed")
                return None, None
        except Exception as e:
            print(f"❌ macOS say TTS error: {e}")
            return None, None
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    else:
        return None, None

def text_to_speech(text):
    print(f"🔊 [TTS] Request text ({len(text)} chars): {text[:200]}")
    # Try Edge TTS
    if USE_EDGE_TTS and HAVE_EDGE_TTS:
        print("🎵 [TTS] Trying Edge TTS...")
        pcm, rate = tts_edge_to_pcm16(text)
        if pcm:
            print(f"✅ [TTS] Edge returned {len(pcm)} bytes @ {rate}Hz")
            return pcm, rate
        print("⚠️ [TTS] Edge failed, falling through...")
    # Try OpenAI TTS
    if USE_OPENAI_TTS and openai_client:
        print("🎵 [TTS] Trying OpenAI TTS...")
        pcm, rate = tts_openai_to_pcm16(text)
        if pcm:
            print(f"✅ [TTS] OpenAI returned {len(pcm)} bytes @ {rate}Hz")
            return pcm, rate
        print("⚠️ [TTS] OpenAI failed, falling through...")
    # Try SAPI
    if HAVE_SAPI:
        print("🎵 [TTS] Trying Windows SAPI...")
        pcm, rate = tts_sapi_to_pcm16(text)
        if pcm:
            print(f"✅ [TTS] SAPI returned {len(pcm)} bytes @ {rate}Hz")
            return pcm, rate
        print("⚠️ [TTS] SAPI failed, falling through...")
    # Fallback espeak/say
    print("🎵 [TTS] Trying local espeak/say fallback...")
    pcm, rate = tts_espeak_to_pcm16(text)
    if pcm:
        print(f"✅ [TTS] Fallback TTS returned {len(pcm)} bytes @ {rate}Hz")
        return pcm, rate
    print("❌ No TTS available")
    return None, None

# -----------------------------
# RTP enqueue & sender
# -----------------------------
def enqueue_rtp_audio_from_pcm16(peer_addr, pcm16_bytes, src_rate, pt_hint=0):
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

def rtp_keepalive_sender(peer_addr, pt_hint=0):
    if peer_addr not in rtp_senders:
        rtp_senders[peer_addr] = {
            'seq': random.randint(0, 65535),
            'ts': random.randint(0, 2**32 - 1),
            'ssrc': random.randint(0, 2**32 - 1),
            'pt': pt_hint,
            'in_speech': False
        }
    st = rtp_senders[peer_addr]
    seq, ts, ssrc = st['seq'], st['ts'], st['ssrc']
    pt_to_send = 8 if pt_hint == 8 else 0
    pad_byte = b"\xD5" if pt_to_send == 8 else b"\xFF"
    if peer_addr not in injection_queues:
        injection_queues[peer_addr] = Queue(maxsize=1000)
    q = injection_queues[peer_addr]
    print(f"📡 RTP sender started for {peer_addr} (PT={pt_to_send}) SSRC={ssrc}")
    packets_sent = 0
    last_report = time.time()
    silence_count = 0
    while True:
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
                st['in_speech'] = False
        try:
            payload = q.get(timeout=0.02)
            non_silence = True
            silence_count = 0
        except Empty:
            payload = pad_byte * 160
            non_silence = False
            silence_count += 1
        
        # Set marker bit on first packet of new speech
        marker_bit = 0
        if non_silence and not st['in_speech']:
            marker_bit = 0x80  # Set marker bit
            st['in_speech'] = True
            print("🎯 Starting new audio sequence (marker bit set)")
        elif not non_silence and silence_count > 25:  # ~500ms of silence
            st['in_speech'] = False
        
        header = struct.pack('!BBHII', 2 << 6, pt_to_send | marker_bit, seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc)
        try:
            sock.sendto(header + payload, peer_addr)
            packets_sent += 1
            if non_silence and (packets_sent % 50 == 0 or packets_sent < 5):
                print(f"📨 RTP: sent non-silence packet #{packets_sent}, queue={q.qsize()}")
        except Exception as e:
            if packets_sent % 1000 == 0:
                print("❌ Send error:", e)
        seq = (seq + 1) & 0xFFFF
        ts = (ts + 160) & 0xFFFFFFFF
        st['seq'], st['ts'] = seq, ts
        if time.time() - last_report >= 10.0:
            print(f"📊 RTP: {packets_sent} pkts, queue: {q.qsize()}")
            last_report = time.time()
        time.sleep(0.02)

# -----------------------------
# Ollama call (streaming) - returns full assembled text and prints incremental chunks
# -----------------------------
def call_ollama(messages, timeout=60):
    """
    Use Ollama /api/chat streaming endpoint.
    'messages' is a list of dicts like {"role": "user","content":"..."} (system message first).
    We stream the response lines, print each chunk, and assemble a final reply string.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
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
            if not raw:
                continue
            try:
                line = raw.decode() if isinstance(raw, bytes) else raw
                # Some lines may be plain JSON objects; parse
                j = json.loads(line)
                # Ollama returns {"response":"...","done":false} chunks (based on test)
                chunk = ""
                if "response" in j and j["response"]:
                    chunk = j["response"]
                elif "message" in j and isinstance(j["message"], dict):
                    # compatibility with /api/chat style
                    chunk = j["message"].get("content", "")
                if chunk:
                    assembled += chunk
                    # print incremental chunk to terminal
                    print(f"🧩 [Ollama chunk] {chunk}", end="", flush=True)
                # if done flag present and true, break
                if j.get("done") or j.get("done_reason") or j.get("final", False):
                    break
            except json.JSONDecodeError:
                # not JSON, print raw
                try:
                    line = raw.decode() if isinstance(raw, bytes) else raw
                    print("\n⚠️ [Ollama] Non-JSON chunk:", line)
                except Exception:
                    pass
        print("")  # newline after streaming chunks
    except requests.exceptions.RequestException as e:
        print("❌ Ollama request error:", e)
        return ""
    except Exception as e:
        print("❌ Ollama error:", e)
        return ""
    return assembled

def prune_history():
    with history_lock:
        non_system = conversation_history[1:]
        if len(non_system) > MAX_HISTORY_MESSAGES:
            del conversation_history[1:1 + len(non_system) - MAX_HISTORY_MESSAGES]

# -----------------------------
# LLM worker: consumes ollama_queue, calls Ollama, TTS, RTP
# -----------------------------
def llm_worker():
    global remote_addr, peer_fmt_pt
    print("🤖 AI assistant worker ready")
    while True:
        try:
            print("🔄 [DEBUG] LLM worker waiting for messages...")
            user_text = ollama_queue.get(timeout=0.5)
            print(f"🔔 [DEBUG] Received message from queue: '{user_text}'")
        except Empty:
            continue
        try:
            if not user_text.strip():
                print("⚠️ Empty message, skipping")
                continue
            
            # Check if this is a system silence prompt
            is_silence_prompt = user_text.startswith("[SYSTEM: User has been silent.")
            if is_silence_prompt:
                # Extract the prompt message
                start_idx = user_text.find("Say: '") + 6
                end_idx = user_text.rfind("']")
                if start_idx > 5 and end_idx > start_idx:
                    assistant_reply = user_text[start_idx:end_idx]
                    print("\n" + "="*60)
                    print(f"🔕 SILENCE DETECTED - Prompting user")
                    print(f"🤖 AI: \"{assistant_reply}\"")
                    print("="*60)
                else:
                    continue
            else:
                print("\n" + "="*60)
                print(f"👤 USER: \"{user_text}\"")
                print("="*60)
                ai_should_stop.clear()
                audio_finish_event.clear()
                with history_lock:
                    conversation_history.append({"role": "user", "content": user_text})
                    prune_history()
                    snapshot = list(conversation_history)
                print(f"📚 [DEBUG] History snapshot length: {len(snapshot)}")
                print("🤔 Thinking (Ollama)...")
                start_time = time.time()
                assistant_reply = call_ollama(snapshot)
                elapsed = time.time() - start_time
                if ai_should_stop.is_set():
                    print("🛑 Interrupted during thinking")
                    continue
                if not assistant_reply:
                    print("⚠️ Empty response from AI")
                    continue
                # Print final assistant reply
                print("\n" + "="*60)
                print(f"🤖 AI (final): \"{assistant_reply.strip()}\"")
                print("="*60)
                print(f"   (LLM took {elapsed:.1f}s)")
                with history_lock:
                    conversation_history.append({"role": "assistant", "content": assistant_reply})
                    prune_history()
            # TTS + RTP
            if SPEAK_ASSISTANT and remote_addr:
                print(f"🔊 [DEBUG] Speech enabled; remote_addr={remote_addr}")
                ai_speaking.set()
                print("🚦 [DEBUG] Set ai_speaking flag")
                print("🔊 Generating speech from assistant reply...")
                tts_start = time.time()
                try:
                    pcm, rate = text_to_speech(assistant_reply)
                except Exception as e:
                    print("❌ TTS generation exception:", e)
                    pcm, rate = None, None
                tts_elapsed = time.time() - tts_start
                print(f"   (TTS generation took {tts_elapsed:.2f}s)")
                if not pcm:
                    print("❌ TTS failed — clearing ai_speaking")
                    ai_speaking.clear()
                    continue
                # enqueue audio
                try:
                    frames = enqueue_rtp_audio_from_pcm16(remote_addr, pcm, rate, peer_fmt_pt)
                    print(f"📤 [DEBUG] Enqueued {frames} frames to RTP queue")
                except Exception as e:
                    print("❌ Enqueue error:", e)
                    frames = 0
                if frames > 0:
                    speech_duration = frames * 0.02
                    # Add extra buffer time to ensure audio plays completely
                    wait_time = speech_duration + 1.5
                    print(f"✅ Playing (queued) {speech_duration:.1f}s of audio; waiting {wait_time:.1f}s for sender to transmit")
                    # Monitor the queue to ensure it empties
                    start_wait = time.time()
                    while time.time() - start_wait < wait_time:
                        if remote_addr in injection_queues:
                            qsize = injection_queues[remote_addr].qsize()
                            if qsize == 0 and time.time() - start_wait > 0.5:
                                print(f"✅ Audio queue emptied after {time.time() - start_wait:.1f}s")
                                time.sleep(0.3)  # Small buffer after queue empty
                                break
                        time.sleep(0.1)
                    print("✅ Audio finished sending")
                else:
                    print("⚠️ Failed to queue audio frames")
                ai_speaking.clear()
                audio_finish_event.set()
                print("🚦 [DEBUG] Cleared ai_speaking flag")
            else:
                if not SPEAK_ASSISTANT:
                    print("⚠️ TTS disabled (SPEAK_ASSISTANT=False)")
                elif not remote_addr:
                    print("⚠️ No remote address connected - cannot play audio")
        except Exception as e:
            print("❌ LLM worker error:", e)
            import traceback
            traceback.print_exc()
            ai_speaking.clear()
        finally:
            print("🔄 [DEBUG] LLM task done")
            try:
                ollama_queue.task_done()
            except Exception:
                pass

# -----------------------------
# Main loop (receives RTP from client), starts workers and RTP sender per peer
# -----------------------------
def main():
    global remote_addr, peer_fmt_pt
    print("=" * 60)
    print("🎧 VOICE ASSISTANT - FIXED & STREAMING OLLAMA")
    print(f"📡 Listening on {LISTEN_IP}:{LISTEN_PORT}")
    print("=" * 60)
    threading.Thread(target=transcribe_worker, daemon=True).start()
    threading.Thread(target=llm_worker, daemon=True).start()
    audio_accumulator = bytearray()
    while True:
        try:
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            continue
        except Exception:
            continue
        # New connection detection
        if remote_addr is None or addr != remote_addr:
            remote_addr = addr
            pt_detected = data[1] & 0x7F if len(data) >= 12 else 0
            peer_fmt_pt = pt_detected if pt_detected in (0, 8) else 0
            print("\n" + "="*60)
            print("📞 CALL CONNECTED")
            print(f"   Address: {remote_addr}")
            print(f"   PT detected: {peer_fmt_pt}")
            print("="*60 + "\n")
            threading.Thread(target=rtp_keepalive_sender, args=(remote_addr, peer_fmt_pt), daemon=True).start()
            # greeting
            time.sleep(0.2)
            if SPEAK_ASSISTANT:
                print("👋 Sending greeting via TTS...")
                pcm, rate = text_to_speech("Hello! I'm ready to help. What can I do for you?")
                if pcm:
                    ai_speaking.set()
                    frames = enqueue_rtp_audio_from_pcm16(remote_addr, pcm, rate, peer_fmt_pt)
                    if frames and frames > 0:
                        time.sleep((frames * 0.02) + 0.5)
                    ai_speaking.clear()
                    print("   ✅ Greeting complete")
        # Decode RTP and feed VAD
        pt, pcm8k, _ = decode_rtp(data)
        if pcm8k is None:
            continue
        audio_accumulator.extend(pcm8k)
        if len(audio_accumulator) >= VAD_FRAME_SAMPLES * 2 * 2:
            process_audio_vad(bytes(audio_accumulator))
            audio_accumulator = audio_accumulator[-(VAD_FRAME_SAMPLES * 2):]

if __name__ == "__main__":
    main()
