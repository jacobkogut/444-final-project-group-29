import socket
import time
import sys

PORT = 4000  # change to your port
BUFFER = 65535

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("", PORT))

print(f"Listening on UDP port {PORT}...")

start_time = time.time()
last_time = start_time
bytes_this_interval = 0
packets_this_interval = 0

while True:
    data, addr = s.recvfrom(BUFFER)
    now = time.time()

    bytes_this_interval += len(data)
    packets_this_interval += 1

    if now - last_time >= 1.0:
        interval = now - last_time
        mbps = (bytes_this_interval * 8 / 1_000) / interval
        pps = packets_this_interval / interval
        total_time = now - start_time

        sys.stdout.write(
            f"\r[{total_time:6.1f}s] "
            f"{mbps:8.3f} Kbps, "
            f"{pps:8.0f} pkt/s, "
            f"{bytes_this_interval/1_000:10.1f} KB      "
        )
        sys.stdout.flush()

        last_time = now
        bytes_this_interval = 0
        packets_this_interval = 0
