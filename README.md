# ServingLLMs

How to serve LLMs efficiently on CPU or GPU using efficiency techniques and tools.

## Configuration

This notebook has been tested on an Ubuntu cloud instance with a RTX 3090 (24 GB VRAM) GPU.

`nvidia-smi`

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.29.06              Driver Version: 545.29.06    CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3090        Off | 00000000:06:00.0 Off |                  N/A |
| 30%   37C    P8              20W / 300W |      6MiB / 24576MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
```

### Dataset

The dataset used is **Call Transcripts Scam Determinations**, available on [Kaggle](https://www.kaggle.com/datasets/mealss/call-transcripts-scam-determinations).

### Models

We run inference with the models below, limiting mode output to sentiment only: "Neutral", "Positive", or "Negative". Each inference generates 4 tokens.

mistralai/Mistral-7B-Instruct-v0.2 precision: torch.float16

## Experiments

### LLamacpp

The main goal of llama.cpp is to enable LLM inference with minimal setup and state-of-the-art performance on a wide variety of hardware - locally and in the cloud.
See [llamacpprepo](https://github.com/ggerganov/llama.cpp) for a list of supported platforms and models. We can install the python binding for llamacpp from the [repo](https://github.com/abetlen/llama-cpp-python), that will automatically install llamacpp too.

#### Mac with M1

Detailed instructions for Mac M1 installation can be found [here](https://llama-cpp-python.readthedocs.io/en/latest/install/macos/).

On a Mac with M1 processor (Metal), install with:

```
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

or an already pre-built wheel with Meta support

```
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```

## Download model

Choose a model to download from the [Higging Face](https://huggingface.co/models) repository. For example, to download a [quantized](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) version of [Mistral-7B-instruct](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) 4.37G, run:

```
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir models --local-dir-use-symlinks False

```

## Start API server

Run the llama-cpp-python API server with MacOS Metal GPU support:

```
# config your ggml model path
# make sure it is gguf v2
# make sure it is q4_0
export MODEL=[path to your llama.cpp ggml models]]/[ggml-model-name]]Q4_0.gguf
python3 -m llama_cpp.server --model $MODEL  --n_gpu_layers 1
```

```
./server -m ./models/quantized_q4_1.gguf -c 1024

```
