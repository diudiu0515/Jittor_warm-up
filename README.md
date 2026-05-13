# Jittor_warm-up_contest

A Jittor implementation of Point Cloud Transformer (PCT) for ModelNet40 classification.

## Project Description

This project implements Point Cloud Transformer (PCT) using the Jittor deep learning framework for 3D point cloud classification on the ModelNet40 dataset.

## Requirements

- Python 3.10
- Jittor 1.3.11.0
- NumPy

## Installation

```bash
conda create -n pa3 python=3.10 -y
conda activate pa3
pip install jittor

```

## Dataset

The ModelNet40 dataset files are provided by the Educoder competition platform:https://www.educoder.net/competitions/Jittor-7.

Please place the following files in the `data/` directory:

- `train_points.npy`
- `train_labels.npy`
- `test_points.npy`

Directory structure:

```text
data/
├── train_points.npy
├── train_labels.npy
└── test_points.npy

```

## Usage

```bash
python pct.py --epochs 200 --batch_size 32 --lr 0.001
```
