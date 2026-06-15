# GPAFormer

Code release for GPAFormer: Graph-guided Patch Aggregation Transformer for Efficient 3D Medical Image Segmentation.

This repository is prepared for source-code release only. 

## Repository Contents

```text
gpaformer/
  GPAFormer.py
  region_growing.py
scripts/
  train.py
  smoke_test.py
configs/
  btcv.yaml
docs/
  dataset_preparation.md
```

## Installation

Create an environment with either conda or pip:

```bash
conda env create -f environment.yml
conda activate gpaformer
```

or:

```bash
pip install -r requirements.txt
```

Install a PyTorch build that matches your CUDA driver if the default package index does not provide the right GPU build for your system.

## Data Preparation

Prepare the dataset locally and create a MONAI Decathlon-style datalist JSON. No dataset or split file is distributed in this repository.

See [docs/dataset_preparation.md](docs/dataset_preparation.md).

## Smoke Test

Construct the model:

```bash
python scripts/smoke_test.py
```

Run a forward-pass smoke test with a dummy `96 x 96 x 96` volume:

```bash
python scripts/smoke_test.py --forward
```

## Training

Edit `configs/btcv.yaml` so that `data.root` and `data.datalist` point to your local dataset and datalist JSON.

Start training:

```bash
python scripts/train.py --config configs/btcv.yaml
```

Preflight one batch without entering the training loop:

```bash
python scripts/train.py --config configs/btcv.yaml --preflight
```

Resume from `output_dir/last.pt`:

```bash
python scripts/train.py --config configs/btcv.yaml --resume
```

Training outputs and checkpoints are written under `training.output_dir` and are ignored by git.

## Citation

```bibtex
@misc{lo2026gpaformer,
  title={GPAFormer: Graph-guided Patch Aggregation Transformer for Efficient 3D Medical Image Segmentation},
  author={Lo, Chung-Ming and Liu, I-Yun and Lin, Wei-Yang},
  year={2026},
  eprint={2604.06658},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## License

No license file is included yet. Confirm the intended release license with the project owner before making the repository public.
