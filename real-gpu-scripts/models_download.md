# Notes on downloading the models for vLLM deployment

```
mkdir models
cd models
```

### Download models locally (On a VM)

```bash
mkdir ~/models/Llama-3.2-1B-Instruct
uvx hf download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir ~/models/Llama-3.2-1B-Instruct \
  --revision 9213176726f574b556790deb65791e0c5aa438b6

mkdir ~/models/Mixtral-8x7B-Instruct-v0.1
uvx hf download mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --local-dir ~/models/Mixtral-8x7B-Instruct-v0.1 \
  --revision eba92302a2861cdc0098cc54bc9f17cb2c47eb61

mkdir ~/models/phi-4
uvx hf download microsoft/phi-4 \
  --local-dir ~/models/phi-4 \
  --revision 2db69c1c3e91a05d2c64a3185acfbaf36f744e25

mkdir ~/models/Qwen2.5-Coder-32B-Instruct
uvx hf download Qwen/Qwen2.5-Coder-32B-Instruct \
  --local-dir ~/models/Qwen2.5-Coder-32B-Instruct \
  --revision 381fc969f78efac66bc87ff7ddeadb7e73c218a7

mkdir ~/models/Llama-3.3-70B-Instruct
uvx hf download meta-llama/Llama-3.3-70B-Instruct \
  --local-dir ~/models/Llama-3.3-70B-Instruct \
  --revision 6f6073b423013f6a7d4d9f39144961bfbfbc386b

mkdir ~/models/Qwen3-Next-80B-A3B-Thinking
uvx hf download Qwen/Qwen3-Next-80B-A3B-Thinking \
  --local-dir ~/models/Qwen3-Next-80B-A3B-Thinking \
  --revision e502dd4100cc68c0de57643fd4317ec93a128670

mkdir ~/models/GLM-4.5-Air
uvx hf download zai-org/GLM-4.5-Air \
  --local-dir ~/models/GLM-4.5-Air \
  --revision a24ceef6ce4f3536971efe9b778bdaa1bab18daa

mkdir ~/models/Qwen3-235B-A22B-Instruct-2507-AWQ
uvx hf download QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ \
  --local-dir ~/models/Qwen3-235B-A22B-Instruct-2507-AWQ \
  --revision 96548f657d772187f16e9f04bc50aec78a9f9cdd
```

### Setup Filestore access on the VM

Example: 
The Filestore instance has the following details (Some stuff omitted):
```bash
$ gcloud filestore instances describe benchmark-llm-models --project tandemn --region us-central1
...
fileShares:
- capacityGb: '3072'
  name: llm
...
name: projects/tandemn/locations/us-central1/instances/benchmark-llm-models
networks:
- connectMode: PRIVATE_SERVICE_CONNECT
  ipAddresses:
  - 10.128.15.236
  modes:
  - MODE_IPV4
  network: default
...
protocol: NFS_V3
...
tier: REGIONAL
```

Make sure you have `nfs-common` package

```bash
# Create the mount and mount it
sudo mkdir -p /mnt/filestore-llm
sudo mount -t nfs \
  -o rw,vers=3,tcp,hard,timeo=600,retrans=3,rsize=524288,wsize=524288,resvport,async,nconnect=2 \
  10.128.15.236:/llm /mnt/filestore-llm

# Short validations
findmnt -T /mnt/filestore-llm
df -hT /mnt/filestore-llm
sudo ls -la /mnt/filestore-llm
test_file=$(sudo mktemp /mnt/filestore-llm/.filestore-write-test.XXXXXX)
sudo rm "$test_file"
```

#### Edit the /etc/fstab (Use `sudoedit /etc/fstab)
`10.128.15.236:/llm /mnt/filestore-llm nfs rw,vers=3,tcp,hard,timeo=600,retrans=3,rsize=524288,wsize=524288,resvport,async,nconnect=2,_netdev,nofail 0 0`