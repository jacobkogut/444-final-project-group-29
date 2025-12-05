import socket
import threading
import sys
import time

TARGET_IP = "192.168.2.35"
TARGET_PORT = 90

RECV_IP = "0.0.0.0"
RECV_PORT = 4000

BUFFER_SIZE = 128

# Target rate
TARGET_BITRATE = 20_000  # bits per second
BYTES_PER_PACKET = BUFFER_SIZE
PACKETS_PER_SEC = TARGET_BITRATE / 8 / BYTES_PER_PACKET
SECONDS_PER_PACKET = 1.0 / PACKETS_PER_SEC  # ~0.03 sec


def input_thread(send_event):
    """Waits for user to press ENTER and signals main thread."""
    while True:
        _ = sys.stdin.readline()
        send_event.set()


def main():
    # Sender socket
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Receiver socket
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind((RECV_IP, RECV_PORT))
    recv_sock.settimeout(2)

    buffer = bytearray()
    completed_buffers = []

    send_event = threading.Event()
    threading.Thread(target=input_thread, args=(send_event,), daemon=True).start()

    print("Receiving on 0.0.0.0:4000 ...")
    print("Press ENTER at any time to send all buffered 1200-byte packets.\n")

    try:
        while True:
            try:
                data, _ = recv_sock.recvfrom(BUFFER_SIZE)
                buffer.extend(data)
            except socket.timeout:
                pass

            while len(buffer) >= BUFFER_SIZE:
                completed_buffers.append(bytes(buffer[:BUFFER_SIZE]))
                buffer = buffer[BUFFER_SIZE:]

            if send_event.is_set():
                send_event.clear()

                print(
                    f"Sending {len(completed_buffers)} buffers at exactly 320 kbps..."
                )

                next_send_time = time.perf_counter()

                for pkt in completed_buffers:
                    # Wait until scheduled send time
                    now = time.perf_counter()
                    if now < next_send_time:
                        time.sleep(next_send_time - now)

                    send_sock.sendto(pkt, (TARGET_IP, TARGET_PORT))
                    next_send_time += SECONDS_PER_PACKET

                completed_buffers.clear()
                print("Done.\n")

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        send_sock.close()
        recv_sock.close()


if __name__ == "__main__":
    main()
