FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Salin requirements.txt
COPY requirements.txt .

# Upgrade pip dan tools build
RUN pip install --upgrade pip setuptools wheel packaging

# Install semua dependencies dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Reinstall flash-attn manual agar build lebih stabil
RUN pip uninstall -y flash-attn && \
    pip install --no-cache-dir flash-attn --no-build-isolation

# Salin seluruh source code
COPY . .

# Buka port aplikasi
EXPOSE 8000

# Jalankan aplikasi
CMD ["python", "main.py"]
