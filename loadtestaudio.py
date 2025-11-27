import numpy as np
from datasets import load_dataset, Audio
import torch

# Configuration
SAMPLE_RATE = 8000
FRAME_SIZE = 512

# 1. Load one sample from the dataset
print("Loading dataset...")
dataset = load_dataset('nguyenvulebinh/libris_clean_100', split='train.clean.100', streaming=True)
dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
item = next(iter(dataset)) # Get first item

# 2. Get the array
audio_array = item['audio']['array']
waveform = torch.from_numpy(audio_array).float()

# 3. Normalize (-1.0 to 1.0)
max_val = torch.max(torch.abs(waveform))
if max_val > 0:
    waveform = waveform / max_val

# 4. Extract exactly one frame (512 samples)
# Let's take a slice from the middle to avoid silence at the start
start_idx = 2000 
if len(waveform) < start_idx + FRAME_SIZE:
    start_idx = 0
    
test_frame = waveform[start_idx : start_idx + FRAME_SIZE].numpy()

# 5. Generate C Header File
header_content = f"""
#ifndef TEST_SAMPLE_H
#define TEST_SAMPLE_H

// Raw Audio Sample (Linear, Normalized -1.0 to 1.0)
const float TEST_INPUT_AUDIO[{FRAME_SIZE}] = {{
"""

for i, val in enumerate(test_frame):
    header_content += f"{val:.6f}f, "
    if (i+1) % 10 == 0:
        header_content += "\n"

header_content += "};\n\n#endif\n"

with open("test_sample.h", "w") as f:
    f.write(header_content)

print(f"Success! Saved 'test_sample.h' with {len(test_frame)} samples.")
print(f"Sample values (First 5): {test_frame[:5]}")