import socket, struct, time, random, threading
from queue import Queue, Full
import numpy as np
import audioop
import whisper
import os
import warnings

# Suppress noisy deprecation warning from audioop on Python 3.11+
warnings.filterwarnings("ignore", category=DeprecationWarning)

LISTEN_IP = '0.0.0.0'
LISTEN_PORT = 4000
RECV_SAMPLE_RATE = 8000  # Incoming RTP PCMU/PCMA
RECOG_RATE = 16000       # Whisper expects 16kHz or higher
# ---- Configuration (edit here, no environment variables) ----
WHISPER_TASK = "transcribe"   # "transcribe" or "translate"
DUMP_INPUT_WAV = False         # set True to periodically save inbound audio

# RTP state
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LISTEN_IP, LISTEN_PORT))
sock.settimeout(1.0)

remote_addr = None
peer_fmt_pt = None
sent_hello = set()
rtp_senders = {}
keepalive_threads = {}

# ---------------------------
# Whisper Model (local)
# ---------------------------
WHISPER_MODEL = "medium"       # medium = best balance of speed and accuracy
print(f"Loading Whisper model ({WHISPER_MODEL}, CPU, fp16 disabled)...")
whisper_model = whisper.load_model(WHISPER_MODEL, device="cpu")
transcribe_queue = Queue(maxsize=8)
transcribe_lock = threading.Lock()  # guard model if needed

# ---------------------------
# Helper functions
# ---------------------------
def decode_rtp(pkt: bytes):
    """Decode RTP payload to 16-bit PCM at 8k using proper RTP header parsing (CSRC/extension).
    Returns (pt, pcm8k_bytes, raw_payload_bytes) or (None, None, None) on failure."""
    if len(pkt) < 12:
        return None, None, None
    b0 = pkt[0]
    version = (b0 >> 6) & 0x03
    padding = (b0 >> 5) & 0x01
    extension = (b0 >> 4) & 0x01
    csrc_count = b0 & 0x0F
    if version != 2:
        return None, None, None
    pt = pkt[1] & 0x7F
    header_len = 12 + (4 * csrc_count)
    if len(pkt) < header_len:
        return None, None, None
    # Header extension
    if extension:
        if len(pkt) < header_len + 4:
            return None, None, None
        ext_profile = struct.unpack_from('!H', pkt, header_len)[0]
        ext_len_words = struct.unpack_from('!H', pkt, header_len + 2)[0]
        header_len += 4 + (ext_len_words * 4)
        if len(pkt) < header_len:
            return None, None, None
    payload = pkt[header_len:]
    if not payload:
        return None, None, None
    # Decode based on payload type
    if pt == 0:  # PCMU
        pcm8k = audioop.ulaw2lin(payload, 2)
    elif pt == 8:  # PCMA
        pcm8k = audioop.alaw2lin(payload, 2)
    else:
        return None, None, None
    return pt, pcm8k, payload

def resample_8k_to_16k(pcm8k_bytes: bytes) -> bytes:
    """Simple linear interpolation resampling from 8kHz to 16kHz (just repeat each sample)."""
    if not pcm8k_bytes:
        return b""
    x = np.frombuffer(pcm8k_bytes, dtype=np.int16)
    # Since 16k is exactly 2x 8k, just repeat each sample
    y = np.repeat(x, 2)
    return y.tobytes()

def save_wav_8k(path: str, pcm8k_bytes: bytes):
    try:
        import wave
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RECV_SAMPLE_RATE)
            wf.writeframes(pcm8k_bytes)
    except Exception as e:
        print("save_wav_8k error:", e)

def send_probe_tone(peer_addr, pt_hint: int = 0, freq: int = 1000, ms: int = 800):
    """Send a short RTP tone to establish symmetric RTP / NAT and verify audio path."""
    try:
        samples = int(RECV_SAMPLE_RATE * ms / 1000)
        t = np.arange(samples)
        tone = (0.6 * 32767 * np.sin(2 * np.pi * freq * t / RECV_SAMPLE_RATE)).astype(np.int16)
        pcm8 = tone.tobytes()

        if pt_hint == 8:
            payload = audioop.lin2alaw(pcm8, 2)
            pt_to_send = 8
            pad_byte = b"\xD5"
        else:
            payload = audioop.lin2ulaw(pcm8, 2)
            pt_to_send = 0
            pad_byte = b"\xFF"

        st = rtp_senders.get(peer_addr)
        if not st or st.get('pt') != pt_to_send:
            st = {
                'seq': random.randint(0, 0xFFFF),
                'ts': random.randint(0, 0xFFFFFFFF),
                'ssrc': random.randint(0, 0xFFFFFFFF),
                'pt': pt_to_send,
            }
            rtp_senders[peer_addr] = st
        seq = st['seq']
        ts = st['ts']
        ssrc = st['ssrc']
        samples_per_packet = 160  # 20ms
        sent = 0
        for i in range(0, len(payload), samples_per_packet):
            chunk = payload[i:i+samples_per_packet]
            if len(chunk) < samples_per_packet:
                chunk = chunk + (pad_byte * (samples_per_packet - len(chunk)))
            marker = 1 if sent == 0 else 0
            b1 = (2 << 6)
            b2 = (marker << 7) | pt_to_send
            hdr = struct.pack('!BBHII', b1, b2, seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc & 0xFFFFFFFF)
            try:
                sock.sendto(hdr + chunk, peer_addr)
            except Exception:
                break
            seq = (seq + 1) & 0xFFFF
            ts = (ts + samples_per_packet) & 0xFFFFFFFF
            sent += 1
            time.sleep(0.02)
        st['seq'] = seq
        st['ts'] = ts
        print(f"Sent probe tone ({sent} packets) to {peer_addr} using PT {pt_to_send}")
    except Exception as e:
        print("Probe tone error:", e)

def rtp_keepalive_sender(peer_addr, pt_hint: int = 0):
    """Continuously send RTP silence frames (20ms) to maintain symmetric RTP and coax audio."""
    try:
        pt_to_send = 8 if pt_hint == 8 else 0
        pad_byte = b"\xD5" if pt_to_send == 8 else b"\xFF"

        st = rtp_senders.get(peer_addr)
        if not st or st.get('pt') != pt_to_send:
            st = {
                'seq': random.randint(0, 0xFFFF),
                'ts': random.randint(0, 0xFFFFFFFF),
                'ssrc': random.randint(0, 0xFFFFFFFF),
                'pt': pt_to_send,
            }
            rtp_senders[peer_addr] = st

        seq = st['seq']
        ts = st['ts']
        ssrc = st['ssrc']
        samples_per_packet = 160  # 20ms @8k
        silence_payload = pad_byte * samples_per_packet
        while True:
            marker = 0
            b1 = (2 << 6)
            b2 = (marker << 7) | pt_to_send
            hdr = struct.pack('!BBHII', b1, b2, seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc & 0xFFFFFFFF)
            try:
                sock.sendto(hdr + silence_payload, peer_addr)
            except Exception:
                pass
            seq = (seq + 1) & 0xFFFF
            ts = (ts + samples_per_packet) & 0xFFFFFFFF
            # persist occasionally
            st['seq'] = seq
            st['ts'] = ts
            time.sleep(0.02)
    except Exception as e:
        print("keepalive error:", e)

def transcribe_worker():
    """Single worker thread that runs Whisper sequentially to avoid concurrency issues."""
    while True:
        pcm16_bytes = transcribe_queue.get()
        try:
            # Convert to float32 waveform in range [-1, 1]
            x = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            # Run transcribe with optimized settings for high accuracy
            with transcribe_lock:
                result = whisper_model.transcribe(
                    x, 
                    task=WHISPER_TASK, 
                    fp16=False,
                    language=None,  # auto-detect for multilingual
                    beam_size=5,  # beam search for accuracy
                    best_of=5,  # sample multiple candidates
                    temperature=0.0,  # deterministic
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.5,  # balanced threshold
                    condition_on_previous_text=False  # disable to prevent hallucination loops
                )
            text = result.get("text", "").strip()
            # Filter out common hallucination phrases
            hallucinations = [
                "thank you for watching",
                "thanks for watching",
                "please subscribe",
                "like and subscribe",
                "mbc 뉴스",
                "subscribe to",
                "ご視聴ありがとうございました"
            ]
            if text:
                # Check if it's likely a hallucination
                text_lower = text.lower()
                is_hallucination = any(phrase in text_lower for phrase in hallucinations)
                if not is_hallucination:
                    tag = "Translation" if WHISPER_TASK == "translate" else "Transcription"
                    print(f"[Whisper {tag}]:", text)
        except Exception as e:
            print("Whisper worker error:", e)
        finally:
            transcribe_queue.task_done()

# ---------------------------
# RTP Listening Loop
# ---------------------------
# For simplicity, we buffer short utterances (~1 sec) before sending to Whisper
audio_buffer = bytearray()
BUFFER_MS = 1200  # target chunk ~1.2s for faster response
BUFFER_SAMPLES = RECV_SAMPLE_RATE * BUFFER_MS // 1000
OVERLAP_MS = 150  # ~150ms overlap to avoid cutting words
OVERLAP_SAMPLES = RECV_SAMPLE_RATE * OVERLAP_MS // 1000
MIN_CHUNK_MS = 500  # flush at least 500ms when idle
MIN_CHUNK_SAMPLES = RECV_SAMPLE_RATE * MIN_CHUNK_MS // 1000
IDLE_FLUSH_MS = 800  # flush after 0.8s idle for fast response

# Rolling inbound audio debug buffer (8 kHz PCM16 after G.711 decode)
debug_buffer_8k = bytearray()
DEBUG_MAX_SEC = 3
DEBUG_MAX_BYTES = RECV_SAMPLE_RATE * DEBUG_MAX_SEC * 2
last_debug_dump = 0.0

def rms16(pcm_bytes: bytes) -> float:
    if not pcm_bytes:
        return 0.0
    a = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean((a/32768.0)**2)))

def rms8(pcm8_bytes: bytes) -> float:
    if not pcm8_bytes:
        return 0.0
    a = np.frombuffer(pcm8_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean((a/32768.0)**2)))

def main():
    global remote_addr, peer_fmt_pt, audio_buffer, debug_buffer_8k, last_debug_dump
    print(f"Listening RTP on {LISTEN_IP}:{LISTEN_PORT}")

    # Start transcription worker
    threading.Thread(target=transcribe_worker, daemon=True).start()

    last_enqueue_ts = time.time()
    last_silence_log = 0.0

    while True:
        try:
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            continue
        except Exception as e:
            print("Recv error:", e)
            continue

        if (remote_addr is None) or (addr != remote_addr):
            remote_addr = addr
            if len(data) >= 12:
                pt_det = data[1] & 0x7F
                peer_fmt_pt = pt_det if pt_det in (0, 8) else 0
            else:
                peer_fmt_pt = 0
            print("RTP peer:", remote_addr, "PT=", peer_fmt_pt)
            # Kick off a short probe to establish symmetric RTP
            try:
                threading.Thread(target=send_probe_tone, args=(remote_addr, peer_fmt_pt or 0), daemon=True).start()
            except Exception:
                pass
            # Start continuous keepalive silence to maintain path
            try:
                if remote_addr not in keepalive_threads:
                    t = threading.Thread(target=rtp_keepalive_sender, args=(remote_addr, peer_fmt_pt or 0), daemon=True)
                    t.start()
                    keepalive_threads[remote_addr] = t
                    print("Started RTP keepalive sender.")
            except Exception as e:
                print("Failed to start keepalive:", e)

        # Decode RTP to PCM
        pt, pcm8k, raw_payload = decode_rtp(data)
        if pcm8k is None:
            continue

        # Append to debug buffer (bounded)
        debug_buffer_8k.extend(pcm8k)
        if len(debug_buffer_8k) > DEBUG_MAX_BYTES:
            debug_buffer_8k = debug_buffer_8k[-DEBUG_MAX_BYTES:]

        # Optional: periodically dump inbound audio to wav for verification
        if DUMP_INPUT_WAV:
            now_t = time.time()
            if now_t - last_debug_dump > 3.0 and len(debug_buffer_8k) > RECV_SAMPLE_RATE * 2:
                # Only dump if there's reasonable energy
                a = np.frombuffer(debug_buffer_8k, dtype=np.int16).astype(np.float32)
                r = float(np.sqrt(np.mean((a/32768.0)**2))) if a.size else 0.0
                if r > 0.0005:
                    fn = f"in_debug_{int(now_t)}.wav"
                    save_wav_8k(fn, bytes(debug_buffer_8k))
                    print(f"Saved inbound debug wav: {fn} (rms={r:.4f}, {len(debug_buffer_8k)//2/RECV_SAMPLE_RATE:.1f}s)")
                    last_debug_dump = now_t

        # Buffer for a short window before translating
        audio_buffer.extend(pcm8k)
        now = time.time()
        # Condition 1: size-based flush (preferred)
        if len(audio_buffer) >= BUFFER_SAMPLES * 2:
            pcm16 = resample_8k_to_16k(audio_buffer)
            # Check audio quality before sending to Whisper
            energy = rms16(pcm16)
            if energy < 0.0005:  # Lower threshold to catch more speech
                last_enqueue_ts = now
                tail = audio_buffer[-OVERLAP_SAMPLES * 2:] if OVERLAP_SAMPLES > 0 else b""
                audio_buffer = bytearray(tail)
            else:
                # Always send audio to Whisper - let it handle silence detection
                try:
                    transcribe_queue.put_nowait(pcm16)
                except Full:
                    # Drop oldest by clearing queue once to keep up in realtime
                    try:
                        _ = transcribe_queue.get_nowait()
                        transcribe_queue.task_done()
                    except Exception:
                        pass
                    try:
                        transcribe_queue.put_nowait(pcm16)
                    except Full:
                        pass
            last_enqueue_ts = now
            # Keep small overlap from tail to avoid chopping words
            tail = audio_buffer[-OVERLAP_SAMPLES * 2:] if OVERLAP_SAMPLES > 0 else b""
            audio_buffer = bytearray(tail)
        # Condition 2: time-based flush to ensure we see text even with short speech
        elif (now - last_enqueue_ts) * 1000 >= IDLE_FLUSH_MS and len(audio_buffer) >= MIN_CHUNK_SAMPLES * 2:
            pcm16 = resample_8k_to_16k(audio_buffer)
            # Check audio quality before sending to Whisper
            energy = rms16(pcm16)
            if energy < 0.0005:  # Skip very low energy audio
                last_enqueue_ts = now
                tail = audio_buffer[-OVERLAP_SAMPLES * 2:] if OVERLAP_SAMPLES > 0 else b""
                audio_buffer = bytearray(tail)
            else:
                # Always send audio to Whisper - let it handle silence detection
                try:
                    transcribe_queue.put_nowait(pcm16)
                except Full:
                    try:
                        _ = transcribe_queue.get_nowait()
                        transcribe_queue.task_done()
                    except Exception:
                        pass
                    try:
                        transcribe_queue.put_nowait(pcm16)
                    except Full:
                        pass
            last_enqueue_ts = now
            tail = audio_buffer[-OVERLAP_SAMPLES * 2:] if OVERLAP_SAMPLES > 0 else b""
            audio_buffer = bytearray(tail)

if __name__ == "__main__":
    main()
