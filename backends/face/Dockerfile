# Use an official NVIDIA CUDA base image for GPU access
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set up the environment and install wget
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10.3 python3.10.3-venv python3-pip git ffmpeg libsm6 libxext6 wget \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# --- BUNDLE MODELS ---
# Create the models directory
RUN mkdir -p /app/models
# Download the required models directly into the image
RUN wget -O /app/models/inswapper_128_fp16.onnx https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128_fp16.onnx
RUN wget -O /app/models/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

# Copy and install Python requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the face backend code into the container
COPY . .

# Expose the port the server will run on
EXPOSE 8081

# Command to run the application
CMD ["python3", "face_main.py"]