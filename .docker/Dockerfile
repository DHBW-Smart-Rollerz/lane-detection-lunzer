# CPU-Variante; für CUDA tausche Base-Image gegen nvidia/cuda:12.3.2-cudnn-runtime-ubuntu22.04
FROM mcr.microsoft.com/devcontainers/python:3.12

# System-Abhängigkeiten
RUN apt-get update && apt-get install -y --no-install-recommends \
git ffmpeg libsm6 libxext6 \
&& rm -rf /var/lib/apt/lists/*


RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio

RUN pip install --no-cache-dir \
    opencv-python-headless onnx onnxruntime numpy scipy matplotlib \
    albumentations tqdm pyyaml rich hydra-core jupyter ipykernel \
    pandas fastapi uvicorn plotly

# Arbeitsverzeichnis für deinen Quellcode festlegen
WORKDIR /workspace