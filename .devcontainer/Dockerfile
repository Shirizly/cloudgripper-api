# FROM python:3.9
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Install common development tools.
RUN apt-get update && apt-get install -y -q --allow-unauthenticated \
    build-essential \
    procps \
    file \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxext6 \
    libx11-6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    x11-apps \
    xauth \
    zsh \
    ripgrep \
    fzf \
    ugrep \
    tmux \
    python3 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.tar.gz && \
                rm -rf /opt/nvim && \
                tar -C /opt -xzf nvim-linux-x86_64.tar.gz
RUN rm nvim-linux-x86_64.tar.gz

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ADD https://astral.sh/ruff/install.sh /ruff-installer.sh
RUN sh /ruff-installer.sh && rm /ruff-installer.sh

ADD https://starship.rs/install.sh /starship-installer.sh
RUN sh /starship-installer.sh --yes && rm /starship-installer.sh


# RUN git clone https://github.com/Homebrew/brew /home/linuxbrew/.linuxbrew/Homebrew \
#     && mkdir /home/linuxbrew/.linuxbrew/bin \
#     && ln -s ../Homebrew/bin/brew /home/linuxbrew/.linuxbrew/bin \
#     && eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv) \
#     && brew --version
#
# RUN chmod -R ugoa=rwx /home/linuxbrew


# RUN useradd -ms /bin/bash main
# RUN mkdir /home/main/.config
# RUN mkdir /home/main/.config/fish/
# RUN chmod -R ugoa=rwx /home/main/
# USER main

# Set a default working directory.
WORKDIR /workspace


RUN echo 'export PATH="$PATH:/opt/nvim-linux-x86_64/bin"' >> ~/.bashrc
RUN echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
RUN echo 'eval "$(uv generate-shell-completion zsh)"' >> ~/.zshrc
RUN echo 'eval "$(starship init bash)"' >> ~/.bashrc
RUN echo 'eval "$(starship init zsh)"' >> ~/.zshrc

RUN echo 'starship init fish | source' >> ~/.config/fish/config.fish
