FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install OS dependencies & build tools
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview

RUN pip install --upgrade pip setuptools wheel packaging

# RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies dari requirements.txt
RUN pip install -r requirements.txt

# Install flash-attn manual build supaya compatible dengan CUDA
# RUN pip install flash-attn --no-build-isolation

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


COPY . .

# Salin entrypoint script ke container
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8010

# Gunakan entrypoint custom
ENTRYPOINT ["/entrypoint.sh"]
