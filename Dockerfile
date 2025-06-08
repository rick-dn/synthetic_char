# Use NVIDIA base image with CUDA
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Disable interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set timezone
ENV TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg libgl1-mesa-glx \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgtk2.0-dev libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy everything into container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install \
    git+https://github.com/tencent-ailab/IP-Adapter.git diffusers transformers accelerate safetensors opencv-python insightface onnxruntime-gpu einops peft

# Automatically run your script
CMD ["python3", "scripts/inference.py"]
