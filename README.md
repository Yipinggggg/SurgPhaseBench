# Surgical Phase Recognition Benchmark

A unified benchmark repository for **surgical phase recognition**, providing a refactored and standardized framework to train and evaluate representative methods from the literature.

This repository integrates **third-party implementations** and **refactored code** into a common structure for more consistent comparison, reproducibility, and easier extension.

The current implementation is based on PyTorch Lightning and supports experiment tracking via Weights & Biases. Please ensure that all required packages are installed before use.

## Supported Model

| Model | Type | Temporal Modeling | Notes |
|------|------|-------------------|-------|
| [SV-RCNet](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8240734) | End-to-end | ResNet-50 + LSTM | Classic recurrent baseline |
| [TMRNet](https://arxiv.org/pdf/2103.16327) | End-to-end | ResNet-50 + LSTM + memory bank | Temporal memory relation modeling |
| [TeCNO](https://arxiv.org/pdf/2003.10751) | Two-stage / temporal | ResNet-50 + TCN | Multi-stage temporal convolution |
| [Trans-SVNet](https://arxiv.org/pdf/2103.09712) | Two-stage / temporal | ResNet-50 + TCN + Transformer | Hybrid temporal modeling |
| [ASFormer](https://arxiv.org/pdf/2110.08568) | Two-stage / temporal | Transformer | Transformer-based temporal modeling |
| [Causal-Transformer](https://arxiv.org/abs/2412.04039) | Two-stage / temporal | Transformer | Causal Implementation of ASFormer - transformer-based temporal modelling |
| [OperA](https://arxiv.org/pdf/2103.03873) | Two-stage / temporal | Transformer | Causal Transformer Temporal modelling |
| [TUNeS](https://arxiv.org/pdf/2307.09997) | Two-stage / temporal | Convolution + Transformer | U-Net like temporal modelling; current implementation requires further verification|

More incoming!

### Feature Encoder

The feature encoder supports both **Torchvision** and **Hugging Face** models with minimal modifications. In addition, several pretrained models are integrated, including:

- [GSViT](https://github.com/SamuelSchmidgall/GSViT)  
- [SurgeNet](https://github.com/timjaspers0801/surgenet)  
- [SurgeNet-DINO](https://github.com/rlpddejong/SurgeNetDINO)  
- [EndoFM](https://github.com/med-air/Endo-FM)  

## Evaluation
We follow established evaluation practices for surgical phase recognition. The current implementation is largely based on:

```bibtex
@article{funke2023metrics,
  title={Metrics matter in surgical phase recognition},
  author={Funke, Isabel and Rivoir, Dominik and Speidel, Stefanie},
  journal={arXiv preprint arXiv:2305.13961},
  year={2023}
}
```

## Acknowledgements

We thank the original authors and open-source contributors whose work made this benchmark possible! Please ensure that you cite the original papers when using their methods or implementations.

---

## 📦 Models & Features for RAMIE Dataset

We provide pretrained models and intermediate features for surgical phase recognition on the RAMIE dataset.

### 🧠 Model Checkpoints

#### 🔹 End-to-End Models
These models operate directly on raw inputs without requiring pre-extracted features.

- [SV-RCNet_RAMIE.ckpt](https://huggingface.co/Yipinggggg/SurgPhaseBench/resolve/main/SV-RCNet_RAMIE.ckpt)  
- [TMRNet_RAMIE.ckpt](https://huggingface.co/Yipinggggg/SurgPhaseBench/resolve/main/TMRNet_RAMIE.ckpt)

#### 🔹 Temporal Models (Stage 2)
These models take pre-extracted features as input and focus on temporal modeling.
- [TeCNO_RAMIE.ckpt](https://huggingface.co/Yipinggggg/SurgPhaseBench/resolve/main/TeCNO_RAMIE.ckpt)  
- [Causal-Transformer_RAMIE.ckpt](https://huggingface.co/Yipinggggg/SurgPhaseBench/resolve/main/Causal-Transformer_RAMIE.ckpt)  
- [DINO+Causal-Transformer_RAMIE.ckpt](https://huggingface.co/Yipinggggg/SurgPhaseBench/resolve/main/DINO%2BCausal-Transformer_RAMIE.ckpt)  
- [DINO+TeCNO_RAMIE.ckpt](https://huggingface.co/Yipinggggg/SurgPhaseBench/resolve/main/DINO%2BTeCNO_RAMIE.ckpt)  

- [Trans-SVNet_RAMIE.ckpt](https://huggingface.co/Yipinggggg/SurgPhaseBench/resolve/main/Trans-SVNet_RAMIE.ckpt)


### 🗂️ Feature Encoders & Extracted Features

#### 🔹 Feature Encoder Checkpoints (Stage 1)
These models are used to extract frame-level features.
- [ResNet50_RAMIE.ckpt](https://huggingface.co/Yipinggggg/SurgPhaseBench/resolve/main/ResNet50_RAMIE.ckpt)  
- [DINOv2_ViTL_RAMIE.ckpt](https://huggingface.co/Yipinggggg/SurgPhaseBench/resolve/main/DINOv2_ViTL_RAMIE.ckpt)

#### 🔹 Pre-extracted Features
We also release intermediate features to facilitate research on temporal modeling **without requiring access to the original surgical videos**.

- [ResNet50_RAMIE_features.zip](https://huggingface.co/Yipinggggg/SurgPhaseBench/resolve/main/ResNet50_RAMIE_features.zip)  
- [DINOv2_ViTL_RAMIE_features.zip](https://huggingface.co/Yipinggggg/SurgPhaseBench/resolve/main/DINOv2_ViTL_RAMIE_features.zip)
