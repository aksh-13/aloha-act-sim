# ACT on ALOHA (simulation) — training/eval image for the RTX 4090 server.
# Build:  docker build -t aloha-act:0.7.0 .
# Run:    see tracks/method/docker_run.sh
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# MuJoCo renders off-screen through EGL. The container needs the NVIDIA
# "graphics" driver capability for EGL to find the GPU, not just "compute".
ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH \
    HF_HOME=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git ffmpeg \
        libegl1 libgl1 libglvnd0 libgles2 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

# Some base images lack the EGL vendor file; without it libEGL cannot find the NVIDIA driver.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ] || \
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /workspace
COPY requirements.txt .
RUN uv venv --python 3.12 /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install -r requirements.txt

# Repo code is copied in; checkpoints/, outputs/ and the HF cache are mounted at run time.
COPY . .

CMD ["bash"]
