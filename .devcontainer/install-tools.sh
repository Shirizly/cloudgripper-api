# install developer tools
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://pixi.sh/install.sh | bash

echo 'eval "$(uv generate-shell-completion bash)"' >> /root/.bashrc


echo 'eval "$(uv generate-shell-completion zsh)"' >> /root/.zshrc

curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.tar.gz && \
                rm -rf /opt/nvim && \
                tar -C /opt -xzf nvim-linux-x86_64.tar.gz

sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.1/zsh-in-docker.sh)"

echo 'export PATH="$PATH:/opt/nvim-linux-x86_64/bin"' >> /root/.bashrc


