# #!/usr/bin/env python3
"""
score_models.py
"""

import os
import glob
import argparse
import subprocess
import shutil
import numpy as np
import csv
import torch
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    p = argparse.ArgumentParser(description="Score every .pt in a folder (native recovery + perplexity → CSV)")
    p.add_argument("--model_folder", type=str, required=True, help="Directory containing one or more .pt model files")
    p.add_argument("--test_pdb_dir", type=str, required=True, help="Directory of PDB files to pass to `--pdb_path` (e.g. `/data/bio/pdb/test_split/`)")
    p.add_argument("--out_csv", type=str, required=True, help="Path to write output CSV (model_filename,avg_recovery_percent,avg_perplexity)")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for `--score_only` (default: 16)")
    p.add_argument("--max_length", type=int, default=512, help="Max sequence length to allow (default: 512)")
    p.add_argument("--tmp_root", type=str, default="./_tmp_mpn_metrics/", help="Temporary folder (will be created then erased) to hold per‐model `.npz` outputs")
    return p.parse_args()

def compute_metrics_from_npz(npz_folder):
    """
    Given a folder of many .npz files (one per input chain), parse each to compute:
      • avg_recovery_percent
      • avg_perplexity

    Expects each .npz to contain at least:
      - 'pred_idx' (int array of shape (L,))
      - 'true_idx' (int array of shape (L,))
      - 'mask'    (0/1 array of shape (L,))
      - 'score_per_position' (float array of shape (L,)) = -log p(true residue) at each position
    
    If 'pred_idx' or 'true_idx' are missing, recovery= None. If 'score_per_position' missing,
    perplexity = None. This function returns (avg_recovery_percent, avg_perplexity) as floats
    or (None, None) if no valid data.
    """
    import numpy as np
    import glob, os

    npz_files = glob.glob(os.path.join(npz_folder, "*.npz"))
    if len(npz_files) == 0:
        return None, None

    total_correct = 0
    total_positions = 0
    all_perplexities = []

    for fn in npz_files:
        data = np.load(fn)
        if {"pred_idx","true_idx","mask"}.issubset(data.keys()):
            pred = data["pred_idx"]      # shape (L,)
            true = data["true_idx"]      # shape (L,)
            mask = data["mask"].astype(int)  # shape (L,)
            correct = int(((pred == true) * mask).sum())
            positions = int(mask.sum())
            total_correct += correct
            total_positions += positions

            if "score_per_position" in data:
                sc = data["score_per_position"]  # shape (L,)
                length = int(mask.sum())
                if length > 0:
                    # Per‐chain perplexity = exp(sum_i (sc_i * mask_i) / length)
                    ppl_chain = np.exp((sc * mask).sum() / length)
                    all_perplexities.append(ppl_chain)
        else:
            # Cannot compute recovery or perplexity on this single chain
            continue

    if total_positions == 0:
        avg_recovery = None
    else:
        avg_recovery = 100.0 * (total_correct / total_positions)

    if len(all_perplexities) == 0:
        avg_perplexity = None
    else:
        avg_perplexity = float(np.mean(all_perplexities))

    return avg_recovery, avg_perplexity

def main():
    args = parse_args()

    # 1) Prepare the temporary root (delete if it exists, then recreate)
    TMP = args.tmp_root
    if os.path.isdir(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP, exist_ok=True)

    # 2) Find all .pt files under --model_folder
    model_paths = sorted(glob.glob(os.path.join(args.model_folder, "*.pt")))
    if not model_paths:
        raise RuntimeError(f"No .pt files found under `{args.model_folder}`")

    # 3) Define CSV fieldnames (we'll open/close the file for each model)
    fieldnames = ["model_filename", "avg_recovery_percent", "avg_perplexity"]
    
    # Track all results in memory
    all_results = []

    # 4) Loop over each .pt checkpoint with progress bar
    for model_path in tqdm(model_paths, desc="Scoring models", total=len(model_paths)):
        model_name = os.path.basename(model_path)
        print(f"\n>>> Scoring: {model_name}")  # Keep this print for model identification

        # 4a) Create a per‐model tmp folder
        this_tmp = os.path.join(TMP, model_name.replace(".pt",""))
        os.makedirs(this_tmp, exist_ok=True)

        model_weights_folder = os.path.dirname(model_path)            # e.g. "results/noise_spread"
        model_name = os.path.basename(model_path)      # e.g. "v_e150_b10000_n48_noise0.00.pt"
        nm = model_name[:-3]                           # now "v_e150_b10000_n48_noise0.00" (drop the ".pt")


       # Instead of passing a single wildcard, explicitly iterate over each test file:
        test_files = sorted(glob.glob(os.path.join(args.test_pdb_dir, "*.pt")))
        for test_file in test_files:
            cmd = [
                "python", "protein_mpnn_run.py",
                "--score_only", "1",
                "--pdb_path", test_file,
                "--path_to_model_weights", model_weights_folder,
                "--model_name", nm,
                "--batch_size", str(args.batch_size),
                "--max_length", str(args.max_length),
                "--out_folder", this_tmp
            ]
            subprocess.run(cmd, check=True)

        # 4d) Parse metrics from all `.npz` in this_tmp
        rec, ppl = compute_metrics_from_npz(this_tmp)

        # 4e) Store results
        result = {
            "model_filename":         model_name,
            "avg_recovery_percent":   f"{rec:.3f}" if rec is not None else "",
            "avg_perplexity":         f"{ppl:.3f}" if ppl is not None else ""
        }
        all_results.append(result)

        # 4f) Write current state to CSV (overwriting previous file)
        with open(args.out_csv, "w", newline="") as csv_fh:
            writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
            print(f"→ Updated `{args.out_csv}` with {len(all_results)} rows")

        # 4g) Clean up that model's tmp folder
        shutil.rmtree(this_tmp)

    print(f"→ Done. Final results written to `{args.out_csv}` with {len(model_paths)} rows.")

if __name__ == "__main__":
    main()
