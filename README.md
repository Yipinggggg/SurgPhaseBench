# Surgical Phase Recognition Benchmark

A unified benchmark repository for **surgical phase recognition**, providing a refactored and standardized framework to train and evaluate representative methods from the literature.

This repository integrates **third-party implementations** and **refactored code** into a common structure for more consistent comparison, reproducibility, and easier extension.

## Model Zoo

| Model | Type | Temporal Modeling | Notes | Paper |
|------|------|-------------------|-------|-------|
| SV-RCNet | End-to-end | ResNet-50 + LSTM | Classic recurrent baseline | https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8240734 |
| TeCNO | Two-stage | ResNet-50 + TCN | Multi-stage temporal convolution | https://arxiv.org/pdf/2003.10751 |
| Trans-SVNet | Two-stage | ResNet-50 + TCN + Transformer | Hybrid temporal modeling | https://arxiv.org/pdf/2103.09712 |
| ASFormer | Two-stage / temporal | Transformer | Transformer-based temporal modeling | https://arxiv.org/pdf/2110.08568 |
| TMRNet | End-to-end | ResNet-50 + LSTM + memory bank | Temporal memory relation modeling | https://arxiv.org/pdf/2103.16327 |

## Notes

- This repository is intended as a **benchmark framework**, not necessarily an exact reproduction of the original repositories.
- Some components are adapted from **third-party codebases** and have been **refactored** for consistency and maintainability.
- Differences from original implementations may exist due to standardization of training, evaluation, and code structure.

## Goal

This benchmark is designed to support:

- fair comparison across methods
- reproducible experiments
- easier extension with new models
- a cleaner codebase for research and development

## Acknowledgements

We thank the original authors and open-source contributors whose work made this benchmark possible.