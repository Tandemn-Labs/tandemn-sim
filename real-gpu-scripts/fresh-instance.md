# Run on fresh instance

## Auxiliary stuff

### Ubuntu repos
sudo apt update
sudo apt upgrade
sudo apt install neovim tree git login cmake postgresql unzip file

alias psql="PGPASSWORD=tandemn psql -h localhost -p 55432 -U tandemn"

### uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env


### awscli
curl -fsSL https://awscli.amazonaws.com/v2/install.sh | bash


### Set up `kubectl` access



### k9s
curl -fL -o k9s_linux_amd64.deb \
  https://github.com/derailed/k9s/releases/download/v0.51.0/k9s_linux_amd64.deb

sudo apt install ./k9s_linux_amd64.deb


### AIPerf
uv tool install aiperf==0.11.0


### Go
curl -fL -o go1.27.1.linux-amd64.tar.gz \
    https://go.dev/dl/go1.27.1.linux-amd64.tar.gz
rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.27.1.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin
export PATH="$PATH:$(go env GOPATH)/bin"


### Docker engine

#### Add Docker's official GPG key:
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

#### Add the repository to Apt sources:
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update

#### Install Docker
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

#### Docker non-root
sudo groupadd docker
sudo usermod -aG docker $USER

#### Autostart by systemd
sudo systemctl enable docker.service
sudo systemctl enable containerd.service



## Install Tandemn code

ssh-add ~/.ssh/gcp-real-gpu-cp

### tandemn-store
git clone git@github.com:Tandemn-Labs/tandemn-store.git
cd tandemn-store

#### Bring up
make up
make install


### tandemn-sim
cd
git clone git@github.com:Tandemn-Labs/tandemn-sim.git
cd tandemn-sim
uv venv .venv
uv sync

### chunk-manager
cd 
git clone git@github.com:Tandemn-Labs/chunk-manager.git
cd chunk-manager

#### Set up protobuf python examples
uv venv .venv
uv pip install -r examples/python/requirements.txt

uv run python -m grpc_tools.protoc \
  -I "/home/user/chunk-manager/proto/" \
  --python_out="/home/user/chunk-manager/examples/python" \
  --grpc_python_out="/home/user/chunk-manager/examples/python" \
  --pyi_out="/home/user/chunk-manager/examples/python" \
  tandemn/chunkmanager/v1/chunk_manager.proto

### tandemn-system (orca)
cd 
git clone git@github.com:Tandemn-Labs/tandemn-system.git
cd tandemn-system
uv venv .venv
uv sync

### tandemn-router
cd
git clone git@github.com:Tandemn-Labs/tandemn-router.git
cd tandemn-router

#### Build go executable
go build -o tandemn-router ./cmd/tandemn-router
