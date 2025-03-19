FROM python:3.9

# Install common development tools.
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    vim \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set a default working directory.
WORKDIR /workspace

CMD ["bash"]
