"""
merge_batches.py

Merges multiple independently-generated HDF5 batch files into a single combined file.

Usage:
   python merge_batches.py batch000.h5 batch001.h5 ... --output combined.h5
"""

import argparse
import numpy as np
import h5py


def get_dataset_names(fname: str) -> list[str]:
    """Return sorted list of top-level dataset names in an HDF5 file."""
    with h5py.File(fname, 'r') as f:
        return sorted(key for key in f.keys() if isinstance(f[key], h5py.Dataset))


def merge(input_files: list[str], output_file: str):
    # Discover dataset names from the first file, and make sure every
    # subsequent file has the same set of datasets.
    dataset_names = get_dataset_names(input_files[0])
    if not dataset_names:
        raise ValueError(f"No datasets found in {input_files[0]}")

    for fname in input_files[1:]:
        names = get_dataset_names(fname)
        if set(names) != set(dataset_names):
            raise ValueError(
                f"Dataset mismatch between {input_files[0]} "
                f"({dataset_names}) and {fname} ({names})"
            )

    print(f"Detected datasets: {dataset_names}")

    arrays = {key: [] for key in dataset_names}
    total_original = 0

    for fname in input_files:
        print(f"Reading {fname}...")
        with h5py.File(fname, 'r') as f:
            n_original = f.attrs.get('n_original', len(f[dataset_names[0]][:]))
            for key in dataset_names:
                # Only take the original (non-mirrored) half
                arrays[key].append(f[key][:n_original])
            total_original += n_original

    # Concatenate all batches
    print("Concatenating arrays...")
    combined = {key: np.concatenate(arrays[key]) for key in dataset_names}

    print(
        f"Writing {output_file} ({total_original} original samples)...")
    with h5py.File(output_file, 'w') as f:
        for key in dataset_names:
            full = combined[key]
            f.create_dataset(key, data=full, chunks=(65_536), compression=None)  # Align chunks to batch reads
        f.attrs['n_samples'] = total_original
        f.attrs['n_original'] = total_original
        f.attrs['source_files'] = [str(fn) for fn in input_files]

    print(f"Done. Total samples (no mirror): {total_original}")

    return dataset_names


if __name__ == "__main__":
    # Get command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs='+', help="Input HDF5 batch files")
    parser.add_argument("--output", default="radiation_training_data_combined.h5")
    args = parser.parse_args()

    # Merge all files, saving to requested file
    dataset_names = merge(args.inputs, args.output)

    # Load the saved file and check for infs and nans.
    with h5py.File(args.output, "r") as f:
        for key in dataset_names:
            data = f[key][:]
            n_nan = np.sum(~np.isfinite(data))
            print(f"{key:10s}: {n_nan:,} non-finite values out of {len(data):,}")