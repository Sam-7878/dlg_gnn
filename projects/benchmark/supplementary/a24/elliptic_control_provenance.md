# Matched canonical Elliptic controls

25fresh runs reuse the exact canonical X/E/y and seed42–46 stratified50/50val/test masks. Both aligned and permuted vectors use the same pretrained features with a fixed row permutation. Zero uses the same20local epochs and then zeros those features,retaining consumed RNG and compute;this differs from the old A03zero control. Base70 controls global epoch budget. Results preserve the higher Permuted than aligned mean;they do not identify a capacity/regularization mechanism. Raw vectors and validation-max F1 vs test F1 are separate and retained locally.
