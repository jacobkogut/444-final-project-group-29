import socket
import time
import pyaudio

###########################
# Audio + Network Settings
###########################
UDP_IP = "0.0.0.0"
UDP_PORT = 4000

SAMPLE_RATE = 20000  # 20 kHz
CHANNELS = 1  # mono
SAMPLE_WIDTH = 2  # bytes per sample (16-bit PCM)

###########################
# Set up audio playback
###########################
audio = pyaudio.PyAudio()

stream = audio.open(
    format=audio.get_format_from_width(SAMPLE_WIDTH),
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    output=True,
    frames_per_buffer=512,
)

###########################
# Set up UDP socket
###########################
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for audio packets on udp://{UDP_IP}:{UDP_PORT}")

prev_time = time.time()
sample_cnt = 0

###########################
# Main loop
###########################
try:
    while True:
        data, addr = sock.recvfrom(4096)  # blocks until packet arrives
        if not data:
            continue

        # Write raw audio bytes directly to output device
        stream.write(data)
        
        sample_cnt += len(data) / 2

        curr_time = time.time()

        if  curr_time - prev_time >= 1.0:
            print(f"\rReceived {sample_cnt} samples per sec", end="")
            prev_time = curr_time
            sample_cnt = 0


except KeyboardInterrupt:
    print("\nExiting...")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
    sock.close()
