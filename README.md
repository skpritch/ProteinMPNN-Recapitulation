# ProteinMPNN — 2025 Recapitulation Fork

> **TL;DR** – This is a *minimal* fork of the original [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) repository (Dauparas *et al.* 2022).
> I retrained the model on a 27 k-structure Baker-lab set (2021 split), reproduced the main metrics, and added lightweight Slurm / training helpers.
> **All core architecture & IP belong to the original authors.**

-----

## Table of Contents

1.  [Motivation](https://www.google.com/search?q=%23motivation)
2.  [Key Results](https://www.google.com/search?q=%23key-results)
3.  [What Changed](https://www.google.com/search?q=%23what-changed)
4.  [Repository Layout](https://www.google.com/search?q=%23repository-layout)
5.  [Quick Start](https://www.google.com/search?q=%23quick-start)
6.  [Running ProteinMPNN (unchanged API)](https://www.google.com/search?q=%23running-proteinmpnn)
7.  [Example Outputs](https://www.google.com/search?q=%23example-outputs)
8.  [Citing This Work](https://www.google.com/search?q=%23citing-this-work)
9.  [License & Acknowledgements](https://www.google.com/search?q=%23license--acknowledgements)

-----

## Motivation

To understand inverse-folding before applying it to **MAGIC** ion-channel engineering, I:

  * cloned the upstream ProteinMPNN code base;
  * retrained on the **2021 Baker-lab 27 k structure set** (the paper used 2020);
  * reproduced *Table 1* and several ablations (noise-robustness, k-NN, hidden dimension);
  * kept code edits as small as possible to ease diffing against the original.

-----

## Key Results 📊

| Model (ours / paper) | Architectural change | Seq. Rec. % | Test Perplexity |
| :--- | :--- | :--- | :--- |
| **Baseline** | Cα-only, autoregressive | **42.8 / 40.1** | **6.64 / 6.77** |
| **Exp 1** | + heavy-atom distances | **48.6 / 46.1** | **5.43 / 5.54** |
| **Exp 2** | + edge-update MLP | **44.1 / 42.0** | **6.28 / 6.37** |
| **Exp 3** | Exp 1 + Exp 2 + random decoding | **49.5 / 47.3** | **5.30 / 5.36** |
| **Exp 4** | Exp 3 + “agonistic” decoding¹ | **50.2 / 47.9** | **5.21 / 5.25** |

¹*Agonistic* = residue-order-invariant decoding (paper terminology).

We see a consistent **≈ +2 pp sequence-recovery gain and −0.5 perplexity**—attributed to the newer 2021 data split. Training curves, noise sweeps, k-NN and hidden-dim ablations closely overlay the paper’s supplementary figures (see `Figures/MPNN_rerun.png`).

-----

## What Changed

| File / Dir | Purpose |
| :--- | :--- |
| `training/exp_training.py`, `training/ca_training.py` | Drop-in scripts to launch *experiment* and *Cα-only* training without touching core code. |
| `model_utils.py` | **Tiny** patch: accept extra flags & checkpoint names. |
| `slurm/{vanilla,ca,dim,exp}_train.sh` | One-liner Slurm wrappers (dataset path & noise σ passed as env vars). |
| `score_models.py`, `score_models.sh` | Batch inference on 300 random structures; CSV/JSONL summaries. |
| `Figures/` | Final poster & slide deck. |
| `Misc/` | Scratch scripts / intermediate notebooks. |
| `Original_weights/` | Upstream *vanilla*, *ca\_only*, *soluble* checkpoints. |
| `results/` | All logs (`*.log`), TensorBoard, and new weights (`*.pt`). |

Everything else (helper scripts, main entry-point, examples) is **unmodified**.

-----

## Repository Layout

```
ProteinMPNN-Recap/
├── helper_scripts/           # original utilities
├── training/                 # + our exp_training.py, ca_training.py
├── slurm/                    # HPC launchers (NEW)
├── results/                  # logs & checkpoints (NEW)
├── Figures/                  # plots & poster (NEW)
├── Misc/                     # interim dev files (NEW)
├── Original_weights/         # upstream pt files (read-only)
├── score_models.py           # NEW – validation runner
├── score_models.sh           # NEW – wrapper
├── model_utils.py            # lightly patched
└── protein_mpnn_run.py       # original inference script
```

-----

## Quick Start 🚀

```bash
# 1. Clone & create env
git clone https://github.com/<your-handle>/ProteinMPNN-Recap.git
cd ProteinMPNN-Recap
conda env create -f environment.yml      # or follow PyTorch install notes below
conda activate mpnn-recap

# 2. Launch a full-atom training run on Slurm
sbatch slurm/vanilla_train.sh DATA=/path/to/27k_2021

# 3. Score a trained checkpoint
bash score_models.sh \
    --weights results/vanilla_sigma002/v_48_002.pt \
    --jsonl inputs/300_test_set.jsonl
```

No Slurm? Just call `python training/exp_training.py --help` or `ca_training.py` directly with the same flags the original `train_model.py` uses.

-----

## Running ProteinMPNN

The original README content is preserved below verbatim (only pruned for length):

### Pre-trained Weights

  * **Full-atom**: `vanilla_model_weights/v_48_0{02,10,20,30}.pt`, `soluble_model_weights/v_48_010.pt`, `v_48_020.pt`
  * **Cα-only**: `ca_model_weights/v_48_0{02,10,20}.pt` → add `--ca_only`

### Helper Scripts

`helper_scripts/` contains PDB parsing, AA bias, tying residues, etc.

### Code Organisation

  * `protein_mpnn_run.py` \# main inference entry-point
  * `protein_mpnn_utils.py` \# utilities
  * `examples/` \# runnable script examples
  * `inputs/` \# sample PDBs
  * `outputs/` \# example outputs
  * `colab_notebooks/` \# Google Colab demos
  * `training/` \# (now includes our extra training helpers)

### CLI Flags (excerpt)

  * `--ca_only` \# parse Cα-only PDBs
  * `--path_to_model_weights` \# folder of .pt files
  * `--model_name v_48_020` \# `v_48_002` / `010` / `020` / `030`
  * `--use_soluble_model` \# soluble-only weights
  * `--backbone_noise 0.10` \# σ in Å
  * `--num_seq_per_target 5` \# sequences per backbone
  * `--sampling_temp "0.1 0.2"` \# AA sampling temperatures
  * `--score_only` \# score inputs instead of sampling
  * `--conditional_probs_only` \# output p(s\_i | rest, backbone)
  * `--unconditional_probs_only` \# single-shot p(s\_i | backbone)

(Full 60-line flag list retained in `protein_mpnn_run.py`.)

### Conda Setup

```bash
conda create -n mpnn-recap python=3.10
conda activate mpnn-recap
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
```

### Example Scripts (examples/)

  * `submit_example_1.sh` – monomer design
  * `submit_example_3_score_only.sh` – score backbone–sequence pairs
  * `submit_example_6.sh` – homooligomer design

(Full list unchanged.)

-----

## Example Outputs

```
>3HTN, score=1.1705, global_score=1.2045, fixed_chains=['B'], designed_chains=['A','C']
NMYSYKKIGNKYIVS...FER / NMYSYKKIGNKYI...
T=0.1, sample=1, score=0.7291, global_score=0.9330, seq_recovery=0.5736
...
```

  * **score** = $-log P$ averaged over designed positions
  * **global\_score** = $-log P$ over all residues
  * **seq\_recovery** = fraction identical to native (where available)

-----

## Citing This Work

If you use this code or the replicated results, please cite both the upstream paper and this fork:

```bibtex
@article{dauparas2022robust,
  title   = {Robust deep learning-based protein sequence design using ProteinMPNN},
  author  = {Dauparas, J. and Anishchenko, I. and Bennett, N. *et al.*},
  journal = {Science},
  year    = {2022},
  volume  = {378},
  number  = {6615},
  pages   = {49--56}
}

@misc{proteinmpnn_recap_2025,
  author       = {Pritchard, Simon},
  title        = {ProteinMPNN–Recapitulation: minimal fork for reproducibility and noise / architecture ablations},
  year         = {2025},
  howpublished = {\url{https://github.com/<your-handle>/ProteinMPNN-Recap}}
}
```

-----

## License & Acknowledgements

This repository remains under the **MIT License** of the original project.

All credit for the model architecture and original implementation goes to **Justas Dauparas & co-authors (Baker lab, 2022)**. My edits are provided “as-is” with no warranty. PRs and issues welcome—please keep the scope focused on reproducibility rather than new feature development.

Happy inverse-folding\! 🧬
