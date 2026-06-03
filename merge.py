"""
merge_batches.py

Merges multiple independently-generated HDF5 batch files into a single combined file.

Usage:
    python merge_batches.py batch000.h5 batch001.h5 ... --output combined.h5
"""

import argparse
import numpy as np
import h5py

DATASETS = ['x', 'kx', 'ky', 'E', 'z0', 'zf', 'u_perp', 'T', 'g', 'I', 'I_err', 'weight']


def merge(input_files: list[str], output_file: str):
    arrays = {key: [] for key in DATASETS}
    total_original = 0

    for fname in input_files:
        print(f"Reading {fname}...")
        with h5py.File(fname, 'r') as f:
            n_original = f.attrs.get('n_original', len(f['I'][:]))
            for key in DATASETS:
                # Only take the original (non-mirrored) half
                arrays[key].append(f[key][:n_original])
            total_original += n_original

    # Concatenate all batches
    combined = {key: np.concatenate(arrays[key]) for key in DATASETS}

    # # Re-create ky mirror
    # ky_mirror = -combined['ky']
    # points_mirror = {key: combined[key].copy() for key in DATASETS}
    # points_mirror['ky'] = ky_mirror

    print(
        f"Writing {output_file} ({total_original} original + {total_original} mirrored = {2 * total_original} total)...")
    with h5py.File(output_file, 'w') as f:
        for key in DATASETS:
            full = combined[key]
            f.create_dataset(key, data=full)
        f.attrs['n_samples'] = total_original
        f.attrs['n_original'] = total_original
        f.attrs['source_files'] = [str(fn) for fn in input_files]

    print(f"Done. Total samples (no mirror): {total_original}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs='+', help="Input HDF5 batch files")
    parser.add_argument("--output", default="radiation_training_data_combined.h5")
    args = parser.parse_args()

    merge(args.inputs, args.output)