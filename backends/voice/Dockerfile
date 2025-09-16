# Use the same NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set up the environment and install wget/unzip
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10.13 python3.10.13-venv python3-pip git libsndfile1 wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# --- BUNDLE MODELS ---
# Create directories for pretrained and voice models
RUN mkdir -p /app/rvc_models
# Download pretrained models
RUN wget -O /app/rvc_models/hubert_base.pt https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt
RUN wget -O /app/rvc_models/rmvpe.pt https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt
# Download and unzip the default Klee voice model
RUN wget -O /app/klee.zip https://huggingface.co/qweshkka/Klee/resolve/main/Klee.zip
RUN unzip /app/klee.zip -d /app/rvc_models/Klee && rm /app/klee.zip

# Copy and install Python requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the voice backend code into the container
COPY . .

# Expose the port the server will run on
EXPOSE 8080

# Command to run the application
CMD ["python3", "voice_main.py"]