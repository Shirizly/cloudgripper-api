# Use an appropriate base image.
FROM python:3.12

# Install common development tools.
RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.tar.gz && \
                rm -rf /opt/nvim && \
                tar -C /opt -xzf nvim-linux-x86_64.tar.gz

# Set a default working directory.
WORKDIR /workspace

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.1/zsh-in-docker.sh)"

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


 RUN echo 'export PATH="$PATH:/opt/nvim-linux-x86_64/bin"' >> /root/.bashrc

CMD ["bash"]
