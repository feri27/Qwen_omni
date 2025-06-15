#!/bin/bash

# Set offline mode
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL_DIR="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-7B"

# Check if model already exists
if [ ! -d "$MODEL_DIR" ]; then
  echo "Downloading Qwen2.5-Omni-7B..."
  python3 -c "
import os
os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Allow download
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
print('Downloading model...')
Qwen2_5OmniForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-Omni-7B')
print('Downloading processor...')
Qwen2_5OmniProcessor.from_pretrained('Qwen/Qwen2.5-Omni-7B')
print('Download complete!')
"
else
  echo "Model already exists at $MODEL_DIR"
fi

# Set offline mode for runtime
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Run your application
exec python3 main.py