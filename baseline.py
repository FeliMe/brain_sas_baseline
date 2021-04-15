import argparse
from multiprocessing import Pool
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data_utils import DatasetHandler, load_mr_scan
from utils import evaluation, utils


class DataPreloader(Dataset):
    def __init__(self, paths, img_size, slices_lower_upper):
        super().__init__()
        self.samples, self.segmentations, self.brain_masks = [], [], []
        self.load_to_ram(paths, img_size, slices_lower_upper)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_batch(paths, img_size, slices_lower_upper):
        samples = []
        segmentations = []
        for p in paths:
            # Samples are shape [1, slices, height, width]
            sample, segmentation = load_mr_scan(
                p, img_size, equalize=True,
                slices_lower_upper=slices_lower_upper
            )
            samples.append(sample)
            segmentations.append(segmentation)

        return {
            'samples': samples,
            'segmentations': segmentations,
        }
    
    def load_to_ram(self, paths, img_size, slices_lower_upper):
        # Set number of cpus used
        num_cpus = os.cpu_count() - 4

        # Split list into batches
        batches = [list(p) for p in np.array_split(
            paths, num_cpus) if len(p) > 0]

        # Start multiprocessing
        with Pool(processes=num_cpus) as pool:
            temp = pool.starmap(
                self.load_batch,
                zip(batches, [img_size for _ in batches], [slices_lower_upper for _ in batches])
            )
        # temp = self.load_batch(paths, img_size, slices_lower_upper)

        # Collect results
        self.samples = [s for t in temp for s in t['samples']]
        self.segmentations = [s for t in temp for s in t['segmentations']]

    def __getitem__(self, idx):
        return self.samples[idx], self.segmentations[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data params
    parser.add_argument("--img_size", type=int, default=128,
                        help="Image resolution")
    parser.add_argument("--test_ds", type=str, required=True,
                        choices=["BraTS", "MSLUB", "WMH", "MSSEG2015"])
    parser.add_argument("--weighting", type=str, default="FLAIR",
                        choices=["T1", "t1", "T2", "t2", "FLAIR", "flair"])
    parser.add_argument("--test_prop", type=float, default=1.0,
                        help="Fraction of data to evaluate on")
    parser.add_argument("--slices_lower_upper", nargs='+', type=int,
                        default=[15, 125],
                        help="Upper and lower bound for MRI slices. Use "
                        "[15, 125] for experiment 1 and [84, 88] for "
                        "experiment 2")
    # Logging params
    parser.add_argument("--n_images_log", type=int, default=30)
    parser.add_argument("--save_dir", type=str, default="./logs/baseline/")
    args = parser.parse_args()

    args.save_dir = f"{args.save_dir}{args.img_size}_" \
                    f"{args.slices_lower_upper[0]}-" \
                    f"{args.slices_lower_upper[1]}/"

    # Get train and test paths
    ds_handler = DatasetHandler()
    paths, _ = ds_handler(args.test_ds, args.weighting)
    paths = paths[-int(len(paths) * args.test_prop):]

    # Load data to RAM
    print("Loading data")
    t_data_start = time()
    ds = DataPreloader(paths, args.img_size, args.slices_lower_upper)
    print(f"Finished loading data in {time() - t_data_start:.2f}s, found {len(ds)} samples.")

    anomaly_maps = torch.cat(ds.samples, 0)
    segmentations = torch.cat(ds.segmentations, 0)

    auroc, aupr, dice, th = evaluation.evaluate(
        predictions=anomaly_maps,
        targets=segmentations,
        # auroc=False,
        # auprc=False,
        # proauc=False,
    )

    # Binarize anomaly_maps
    bin_map = torch.where(anomaly_maps > th, 1., 0.)
    # Connected component filtering
    bin_map = utils.connected_components_3d(bin_map)

    print("Saving some images")
    c = (args.slices_lower_upper[1] - args.slices_lower_upper[0]) // 2
    images = [
        anomaly_maps[:, c][:, None],
        bin_map[:, c][:, None],
        segmentations[:, c][:, None]
    ]
    titles = ['Anomaly map', 'Binarized map', 'Ground truth']
    fig = evaluation.plot_results(images, titles, n_images=args.n_images_log)
    os.makedirs(args.save_dir, exist_ok=True)
    plt.savefig(f"{args.save_dir}{args.test_ds}_{args.weighting}_samples.png")
