# Cross-media-Signatures
Code and data revealing differences between paper and electronic Chinese signatures.

[![DOI](https://zenodo.org/badge/649775887.svg)](https://zenodo.org/badge/latestdoi/649775887)

## Citation
`Luo, J., Pu, Y., Yin, J., Liu, X., Tan, T., Zhang, Y. and Hu, M. (2023), Is There a Difference between Paper and Electronic Chinese Signatures?. Adv. Intell. Syst., 5: 2300439. https://doi.org/10.1002/aisy.202300439`
```bibtex
@article{202300439,
author = {Luo, Ji-Feng and Pu, Yun-Zhu and Yin, Jie-Yang and Liu, Xiaohong and Tan, Tao and Zhang, Yudong and Hu, Menghan},
title = {Is There a Difference between Paper and Electronic Chinese Signatures?},
journal = {Advanced Intelligent Systems},
volume = {5},
number = {12},
pages = {2300439},
keywords = {Bézier curve, cross-media analysis, neural networks, signature authentication},
doi = {https://doi.org/10.1002/aisy.202300439},
year = {2023}
}
```

## Data
### Folder Structure
- Folder "origin" contains originally scanned data of signatures.
- Folder "control" contains data of control group, where volunteers wrote at least twice specified names.
- Folder "experiment" contains data of experimental group, where volunteers wrote specified names on paper, iPad, phone (in hand) and phone (on table).
- Folder "results" contains computing results of all models.
- Folder "models" contains trained neural network models.

## Code
### Folder Structure
Code for analysis is contained in folder "control" and folder "experiment", and that for image process is contained in folder "imageprocess".
