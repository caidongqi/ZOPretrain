#!/usr/bin/env python3
import argparse
import os
import glob
from typing import Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt


def find_best_csv_file(experiment_dir: str) -> Optional[str]:
	"""Find the CSV file with the largest max step in the given directory.

	Searches for client*.csv files. If none, searches for *.csv.
	Returns the path of the file with maximum 'step' value; None if not found.
	"""
	patterns = [
		os.path.join(experiment_dir, "client*.csv"),
		os.path.join(experiment_dir, "*.csv"),
	]
	candidate_files = []
	for pattern in patterns:
		candidate_files.extend(glob.glob(pattern))
	# Deduplicate while preserving order
	seen = set()
	unique_files = []
	for f in candidate_files:
		if f not in seen:
			unique_files.append(f)
			seen.add(f)
	if not unique_files:
		return None

	best_file = None
	best_max_step = -1
	for fpath in unique_files:
		try:
			df_head = pd.read_csv(fpath, nrows=1)
			if "step" not in df_head.columns:
				continue
			df = pd.read_csv(fpath, usecols=["step"], low_memory=False)
			max_step = pd.to_numeric(df["step"], errors="coerce").max()
			if pd.isna(max_step):
				continue
			if int(max_step) > best_max_step:
				best_max_step = int(max_step)
				best_file = fpath
		except Exception:
			# Skip unreadable files
			continue
	return best_file


def load_step_loss(csv_path: str) -> Tuple[pd.Series, pd.Series, Optional[str]]:
	"""Load step and loss series from a CSV path.

	If multiple rows per step exist, take the last occurrence per step (by file order).
	Returns (step_series, loss_series, label_lr)
	"""
	df = pd.read_csv(csv_path, low_memory=False)
	if "step" not in df.columns or "loss" not in df.columns:
		raise ValueError(f"CSV {csv_path} must include 'step' and 'loss' columns")
	# Ensure numeric
	df["step"] = pd.to_numeric(df["step"], errors="coerce")
	df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
	df = df.dropna(subset=["step", "loss"])  # drop rows with invalid numbers
	# Keep last occurrence per step in chronological order
	df = df.reset_index(drop=True)
	last_idx_per_step = df.groupby("step", sort=True).tail(1).index
	df_unique = df.loc[last_idx_per_step].sort_values("step")
	label_lr = None
	if "lr" in df.columns:
		# Pick the first non-null lr representation
		lr_values = df["lr"].dropna().astype(str).unique()
		if len(lr_values) > 0:
			label_lr = lr_values[0]
	return df_unique["step"], df_unique["loss"], label_lr


def main():
	parser = argparse.ArgumentParser(description="Plot loss curves for two experiment directories.")
	parser.add_argument("exp_dir_a", type=str, help="Path to experiment directory A")
	parser.add_argument("exp_dir_b", type=str, help="Path to experiment directory B")
	parser.add_argument("--output", type=str, default=None, help="Path to save the output PNG (optional)")
	args = parser.parse_args()

	exp_dirs = [args.exp_dir_a, args.exp_dir_b]
	series = []
	labels = []
	for exp_dir in exp_dirs:
		if not os.path.isdir(exp_dir):
			raise FileNotFoundError(f"Directory not found: {exp_dir}")
		best_csv = find_best_csv_file(exp_dir)
		if best_csv is None:
			raise FileNotFoundError(f"No CSV files with 'step' found in: {exp_dir}")
		steps, losses, label_lr = load_step_loss(best_csv)
		# Build a label using directory name and lr if available
		dir_name = os.path.basename(os.path.normpath(exp_dir))
		label = dir_name
		if label_lr is not None:
			label = f"{dir_name} (lr={label_lr})"
		series.append((steps, losses))
		labels.append(label)

	plt.figure(figsize=(9, 5.5))
	for (steps, losses), label in zip(series, labels):
		plt.plot(steps.to_numpy(), losses.to_numpy(), label=label, linewidth=1.6)
	plt.xlabel("step")
	plt.ylabel("loss")
	plt.title("Loss vs Step")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()

	if args.output is None:
		# Default output next to the first directory
		base_dir = os.path.dirname(os.path.normpath(args.exp_dir_a)) or "."
		out_name = "loss_compare.png"
		args.output = os.path.join(base_dir, out_name)
	os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
	plt.savefig(args.output, dpi=150)
	print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
	main()
