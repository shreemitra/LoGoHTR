# Learning Beyond Labels: Self-Supervised Handwritten Text Recognition

Official implementation of the paper "Learning Beyond Labels: Self-Supervised Handwritten Text Recognition" (WACV 2026).

## Overview

This repository contains the training code, data modules, and model components for self-supervised handwritten text recognition. The goal is to learn strong visual-text representations beyond labeled data and then transfer them to downstream recognition tasks.

## Repository layout

- Core training entry point: [train.py](train.py)
- Main configuration file: [config.yaml](config.yaml)
- Package metadata: [setup.py](setup.py) and [setup.cfg](setup.cfg)
- Requirements list: [requirements.txt](requirements.txt)
- PyTorch Lightning style module: [comer/lit_comer.py](comer/lit_comer.py)
- Data module, datasets, and transforms: [comer/datamodule](comer/datamodule)
- Model components (encoder, decoder, backbone): [comer/model](comer/model)
- Transformer blocks: [comer/transformer](comer/transformer)
- Decoding and utilities: [comer/utils](comer/utils)

## Setup

1. Create a Python environment suitable for your system.
2. Install dependencies listed in [requirements.txt](requirements.txt).
3. If you want an editable install, use the metadata in [setup.py](setup.py) and [setup.cfg](setup.cfg).

## Data

Dataset loading and preprocessing are implemented in the data module at [comer/datamodule](comer/datamodule). Dictionary files for specific datasets are provided alongside the datasets in that folder.

If you add a new dataset, extend the dataset logic and transforms in the same module to keep a consistent interface.

## Training

Training is orchestrated by the entry point in [train.py](train.py) and the Lightning style module in [comer/lit_comer.py](comer/lit_comer.py). Hyperparameters and paths are defined in [config.yaml](config.yaml). Logs and checkpoints are written under [lightning_logs](lightning_logs) by default.

## Evaluation and decoding

Decoding utilities and beam search live in [comer/utils](comer/utils). If you introduce new decoding logic, keep it there to stay consistent with the rest of the codebase.

## Citation

If you use this work, please cite the paper:

```
Learning Beyond Labels: Self-Supervised Handwritten Text Recognition. WACV 2026.
```

## Acknowledgements

Thanks to the CoMER implementation of CoMER by Green-Wood (https://github.com/Green-Wood/CoMER)
