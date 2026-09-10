# Run on fresh instance

Setup instructions to setup a new machine for serving as the Tandemn experiments' control plane

## Auxiliary stuff

### Ubuntu repos
```bash
sudo apt update
sudo apt upgrade
sudo apt install neovim tree git login cmake \
  postgresql unzip file man-db python3.14-venv \
  nfs-common util-linux-extra python3-pip
```


### uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env


### awscli
curl -fsSL https://awscli.amazonaws.com/v2/install.sh | bash


### kubectl
curl -LO https://dl.k8s.io/release/v1.37.0/bin/linux/amd64/kubectl


### gcloud 
`gcloud` should already be installed. Just use it.

```bash
# This will show a link that you need to open in a browser to login
gcloud auth login
```

#### Install the `gke-cloud-auth-plugin` plugin
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates gnupg curl

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
  | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list

sudo apt-get update
sudo apt-get install -y google-cloud-cli-gke-gcloud-auth-plugin
```

#### Set up the many K8s contexts
For each cluster, run this:
```bash
gcloud container clusters get-credentials "${CLUSTER_NAME}" \
  --project="tandemn" \
  --region="${REGION}"
```

### k9s
curl -fL -o k9s_linux_amd64.deb \
  https://github.com/derailed/k9s/releases/download/v0.51.0/k9s_linux_amd64.deb
sudo apt install ./k9s_linux_amd64.deb


### HuggingFace CLI
curl -LsSf https://hf.co/cli/install.sh | bash
uvx hf auth login


### opencode
curl -fsSL https://opencode.ai/install | bash 


### AIPerf
uv tool install aiperf==0.11.0


### Go
curl -fL -o go1.27.1.linux-amd64.tar.gz \
    https://go.dev/dl/go1.27.1.linux-amd64.tar.gz
rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.27.1.linux-amd64.tar.gz

#### Add this to the .bashrc
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


### tandemn-koi
cd
git clone git@github.com:Tandemn-Labs/tandemn-koi.git
