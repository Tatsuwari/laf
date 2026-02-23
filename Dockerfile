# CUDA + PyTorch base (includes CUDA runtime + torch preinstalled)
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    PORT=8000 \
    MODE_TO_RUN=serverless

WORKDIR /app

# System deps: git (optional), SSH (optional), curl (debug)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git openssh-server ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd

# Harden SSH defaults (key-only if enabled)
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/g' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/g' /etc/ssh/sshd_config

# Install python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/requirements.txt && \
    python -m pip install runpod

# Copy project and install as package (pip install .)
COPY . /app
RUN python -m pip install .

# Startup + handler
COPY deploy/start.sh /app/deploy/start.sh
COPY deploy/handler.py /app/deploy/handler.py
RUN chmod +x /app/deploy/start.sh

# Requested ports
EXPOSE 8000 22

CMD ["/app/deploy/start.sh"]