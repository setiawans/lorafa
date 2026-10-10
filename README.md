# Gradient Leakage in LoRA-FA

Code for the paper "Gradient Leakage in LoRA-FA: What Freezing A Hides, and What Limits the Attacker".

## Setup

Tested with Python 3.13. A CUDA GPU is recommended.

```
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Pick the PyTorch build that matches your CUDA driver. CIFAR-10 and CIFAR-100 are downloaded to `data/` on first use. Run all commands from the repository root.

## 1. Pretrain the backbones

```
python scripts/pretrain.py --config configs/pretrain/resnet20_4_cifar100.yaml
python scripts/pretrain.py --config configs/pretrain/resnet18_cifar100.yaml
```

Checkpoints are saved to `checkpoints/backbone/`.

## 2. Fine-tune with LoRA-FA

```
for c in configs/finetune/[!_]*.yaml; do python scripts/finetune.py --config $c; done
python scripts/splice_checkpoint.py
```

Each config saves its checkpoints to `checkpoints/finetune/<name>/`. The splice script builds the two spliced checkpoints that separate the trained head from the trained B.

## 3. Run the attacks

```
for c in configs/attack/main/[!_]*.yaml; do python scripts/attack.py --config $c; done
```

Each config is one attack setting, and its results are saved to `runs/attack/main/<name>/`. Files starting with `_` are shared base configs. Add `--resume` to continue an interrupted run.

## 4. Local diagnostics

```
python scripts/jacobian.py --config configs/jacobian/resnet20_4.yaml
```

This uses the `pixel` and `dip` runs from step 3 and saves its results to `runs/jacobian/resnet20_4_r16/`.

## 5. Statistics

```
python analysis/identification.py
python analysis/baselines.py
python analysis/calibration.py
python analysis/grad_norms.py
python analysis/results.py
```

`identification.py` scores each reconstruction against all 100 targets and saves `identification.csv` in each run folder from step 3. The next three scripts save `baselines.csv`, `calibration.csv`, and `grad_norms.csv` to `runs/`. `results.py` reads these files and the results of steps 3 and 4, then prints the numbers reported in the paper, including the significance tests.

`analysis/check_backbone.py` and `analysis/check_checkpoints.py` print checks that only the B matrices and the classification head change between checkpoints.

## Tests

```
pytest tests
```

## Layout

- `lorafa/`: models, LoRA-FA layers, gradient observation, losses, and metrics
- `scripts/`: pretraining, fine-tuning, attacks, and local diagnostics
- `configs/`: one YAML file per run
- `analysis/`: identification scoring, baselines, and the reported results
