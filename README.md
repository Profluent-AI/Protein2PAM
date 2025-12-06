# Protein2PAM: Protein Language Models for CRISPR-Cas PAM Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.01.06.631536-green)](https://www.biorxiv.org/content/10.1101/2025.01.06.631536v1)
[![License: Models (CC BY–NC 4.0)](https://img.shields.io/badge/Models%20License-CC%20BY--NC%204.0-orange)](https://creativecommons.org/licenses/by-nc/4.0/)
[![License: Code (Polyform Noncommercial 1.0.0)](https://img.shields.io/badge/Code%20License-Polyform--Noncommercial%201.0.0-blueviolet)](https://polyformproject.org/licenses/noncommercial/1.0.0/)

This repository contains code for Protein2PAM, a tool that predicts CRISPR-Cas PAMs from protein sequences using protein language models. 
The models and their applications are described in [*Nayfach, S., Bhatnagar, A., Novichkov, A., et al. (2025)*](https://www.biorxiv.org/content/10.1101/2025.01.06.631536v1). 
For a browser-based interface, try the [Protein2PAM Webserver](https://protein2pam.profluent.bio/).

The figure below shows the overall Protein2PAM workflow.
```mermaid
flowchart LR
    A[CRISPR-Cas protein] --> B[Protein2PAM model] --> C[Predicted PAM motif]

    style A fill:#0f0f0f,stroke:#00ff99,stroke-width:2px,color:#00ff99
    style B fill:#0f0f0f,stroke:#00ff99,stroke-width:2px,color:#00ff99
    style C fill:#0f0f0f,stroke:#00ff99,stroke-width:2px,color:#00ff99
```

This repo has two Protein2PAM implementations:
* A full implementation in bare PyTorch (under the `protein2pam.models` submodule) which includes uncertainty estimation and visualization utilities, but requires users to manually download certain modeling resources.
* A Huggingface implementation (under the `protein2pam.huggingface` submodule) that includes only the PAM prediction model without uncertainty estimation. This implementation is intended for more low-level usage, but it is integrated with the Huggingface Hub for easier downloading of model weights. You can see the model collection [here](https://huggingface.co/collections/Profluent-Bio/protein2pam).


## Contents
- [Prerequisites](#Prerequisites)
- [Quickstart](#Quickstart)
- [Usage](#usage)
- [Models](#models)
- [Data](#data)
- [License](#license)
- [Citation](#cite-this-work)

## Prerequisites

Before installing Protein2PAM, ensure you have:

- Python 3.8+
  Available from [python.org](https://www.python.org/downloads/).
- NVIDIA GPU with CUDA support:  
  Refer to the [official NVIDIA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) or see our step-by-step instructions [here](docs/INSTALL_GPU.md). Use `nvidia-smi` to ensure GPU(s) are available on your system.
- pip
  If `pip` is not already installed, you can install it using:
  ```bash
  python3 -m ensurepip --upgrade
  ```
- Mamba or Conda:  
  You can install `mamba` using:
  ```bash
  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh
  ```  

## Quickstart

Download repo  
```bash
git clone https://github.com/Profluent-AI/protein2pam
cd protein2pam
```

Create environment and install
```bash
mamba env create -f environment.yml
source activate protein2pam
pip install -e .
```

Download and unpack database (only necessary if using the `protein2pam.models` implementation, not the `protein2pam.huggingface` implementation)
```bash
wget https://storage.googleapis.com/protein2pam-x83y9z7q4k/protein2pam_db_v1.0.tar
tar -xvf protein2pam_db_v1.0.tar
```

## Usage

### Python usage

Below are examples of how to run Protein2PAM locally through the Python API.

Import `protein2pam` and list available models:
```python
import protein2pam
protein2pam.model_info()
```

For the example, we'll use the main `cas9` model, the PID sequence of [SpCas9](https://www.uniprot.org/uniprotkb/Q99ZW2/entry), and the database located at `./protein2pam_db_v1.0`
```python
proteins  = ["VQTGGFSKESILPKRNSDKLIARKKDWDPKKYGGFDSPTVAYSVLVVAKVEKGKSKKLKSVKELLGITIMERSSFEKNPIDFLEAKGYKEVKKDLIIKLPKYSLFELENGRKRMLASAGELQKGNELALPSKYVNFLYLASHYEKLKGSPEDNEQKQLFVEQHKHYLDEIIEQISEFSKRVILADANLDKVLSAYNKHRDKPIREQAENIIHLFTLTNLGAPAAFKYFDTTIDRKRYTSTKEVLDATLIHQSITGLYETRIDLSQLGGD"]
oracle = protein2pam.PAMOracle(
    model_name = "cas9", 
    data_dir = "../protein2pam_db_v1.0"
)
predictions = oracle.evaluate(proteins)
for prediction in predictions:
    print(prediction)
```

We can plot PAM logos using:
```python
for pam in predictions:
    pam.plot_logo(
        file="example.png",
        side="downstream",
        title="example"
    )
```

Alternatively, if you would like to directly use a Huggingface implementation of the PAM prediction model, you can call
```python
import torch
import torch.nn.functional as F
from protein2pam.huggingface import EsmForSequenceClassification, get_tokenizer

# Tokenize proteins so they can be consumed by the model
proteins  = ["VQTGGFSKESILPKRNSDKLIARKKDWDPKKYGGFDSPTVAYSVLVVAKVEKGKSKKLKSVKELLGITIMERSSFEKNPIDFLEAKGYKEVKKDLIIKLPKYSLFELENGRKRMLASAGELQKGNELALPSKYVNFLYLASHYEKLKGSPEDNEQKQLFVEQHKHYLDEIIEQISEFSKRVILADANLDKVLSAYNKHRDKPIREQAENIIHLFTLTNLGAPAAFKYFDTTIDRKRYTSTKEVLDATLIHQSITGLYETRIDLSQLGGD"]
tokenizer = get_tokenizer()
encodings = tokenizer.encode_batch(proteins)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
input_batch = dict(
  input_ids=torch.tensor([encoding.ids for encoding in encodings], device=device),
  attention_mask=torch.tensor([encoding.attention_mask for encoding in encodings], device=device),
)

# Initialize model. You can use any of the supported model names (below) instead of cas9.
model = EsmForSequenceClassification.from_pretrained("Profluent-Bio/protein2pam-cas9", device_map=device)

# Get a batch of PAM probability matrices (as a torch tensor) from the model
# dimension 0 is batch size, dimension 1 is sequence position, and dimension 2 is nucleotide ID (ordered ACGT)
output = model(**input_batch)
pam_probability_matrix = F.softmax(output.logits, dim=-1)
```

## Models

### Models used by the Protein2PAM webserver

| Model Name       | Input Protein/Domain | CRISPR Type     | Samples  | Literature Datasets         |
|-------------------|---------------------|-----------------|----------|------------------------------|
| `cas8`            | Cas8 or Cas10d      | Type I          | 28,410   |                              |
| `cas9`            | Cas9 PI-domain      | Type II         | 15,843   | [1-9](docs/REFERENCES.md)   | 
| `cas12`           | Cas12 protein       | Type V          | 1,720    | [10-21](docs/REFERENCES.md)  |

*These models are exposed on the [Protein2PAM Webserver](https://protein2pam.profluent.bio/). See [Data](#data) for details on training samples.*

### Additional models

In addition to the models deployed on the webserver, we also trained several variants for benchmarking and ablation studies, summarized below:

| Model Name       | Input Protein/Domain | CRISPR Type | Samples | Literature Datasets | Notes |
|------------------|----------------------|-------------|---------|----------------------|-------|
| `cas9_full`       | Cas9 protein        | Type II     | 15,843  | [1-9](docs/REFERENCES.md) | Full-length Cas9 trained with literature PAMs |
| `cas9_full_nolit` | Cas9 protein        | Type II     | 15,731  |                      | Full-length Cas9, no literature PAMs |
| `cas9_pid_nolit`  | Cas9 PI-domain      | Type II     | 15,731  |                      | No literature PAMs |
| `cas9_pid_nme`    | Cas9 PI-domain      | Type II     | 15,843  | [1-9](docs/REFERENCES.md) | Nme orthologs upweighted |
| `cas12_no_lit`    | Cas12 protein       | Type V      | 1,675   |                      | Trained without literature PAMs |


## Data

Sequences and PAM profiles used for training can be downloaded from [Hugging Face](https://huggingface.co/datasets/Profluent-Bio/protein2pam-training-data) and Google Cloud:
```bash
wget https://storage.googleapis.com/protein2pam-x83y9z7q4k/protein2pam_train_seqs.tsv
```
The training set consists of protein:PAM pairs curated from evolutionary data and supplemented with experimental PAMs reported in the literature (see [REFERENCES.md](docs/REFERENCES.md))

## License

This repository contains both software and trained models under different licenses. 

- **Code:** Licensed under the [Polyform Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/). The code may be used, modified, and shared for **non-commercial research and academic purposes only**. Commercial use is prohibited without prior written consent.

- **Models and Data:** Licensed under the [Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). You may share and adapt the models for non-commercial use with appropriate attribution.

The full license texts are available in [`LICENSES.md`](./LICENSES.md).

For commercial licensing inquiries, please contact partnerships@profluent.bio.

## Cite this work

If you use Protein2PAM in your research, please cite the following [preprint](https://www.biorxiv.org/content/10.1101/2025.01.06.631536v1)
