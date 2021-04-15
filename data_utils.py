from glob import glob
import os

import nibabel as nib
import numpy as np
import torch
from torchvision.datasets.folder import IMG_EXTENSIONS
from warnings import warn

from utils import CenterCrop3D, ResizeGray, histogram_equalization

DATAROOT = str(os.environ.get('DATAROOT'))

DICOM_EXT = ('.dcm', )
NIFTI_EXT = ('.nii', '.nii.gz')
ALL_EXT = IMG_EXTENSIONS + DICOM_EXT + NIFTI_EXT

RSNA_TRAIN_PATHS = None
RSNA_TRAIN_LABELS = None
RSNA_TEST_PATHS = None
RSNA_TEST_LABELS = None


class DatasetHandler:
    def __init__(self):
        """A class that returns a list of paths and labels for every dataset

        Usage:
        paths, labels = DatasetHandler()('name of dataset')

        Args:
            dataset_name (str): Name of the dataset we want to get files from
            weighting (str): t1 or t2, for MRI
        """
        pass

    def __call__(self, dataset_name, weighting=None):
        switch_ds = {
            'BraTS': self.returnBraTS,
            'MSLUB': self.returnMSLUB,
            'WMH': self.returnWMH,
            'MSSEG2015': self.returnMSSEG,
        }
        paths, labels = switch_ds[dataset_name](weighting=weighting)

        return paths, labels

    @staticmethod
    def returnBraTS(**kwargs):
        root = os.path.join(DATAROOT, 'BraTS/MICCAI_BraTS2020_TrainingData')
        paths = glob(
            f"{root}/*/*{kwargs['weighting'].lower()}_registered.nii.gz")
        labels = [1 for _ in paths]

        # Raise warning if no files are found
        if len(paths) == 0:
            raise RuntimeWarning(f"No files found for BraTS")

        return paths, labels

    @staticmethod
    def returnMSLUB(**kwargs):
        root = os.path.join(DATAROOT, 'MSLUB/lesion')
        w = kwargs['weighting']
        if w != 'FLAIR':
            w += 'W'
        paths = glob(
            f"{root}/*/*{w.upper()}_stripped_registered.nii.gz")
        labels = [1 for _ in paths]

        # Raise warning if no files are found
        if len(paths) == 0:
            raise RuntimeWarning(f"No files found for MSLUB")

        return paths, labels

    @staticmethod
    def returnMSSEG(**kwargs):
        root = os.path.join(DATAROOT, 'MSSEG2015/training')
        w = kwargs['weighting']
        if w.lower() == 't1':
            w = 'mprage'
        paths = glob(
            f"{root}/training*/training*/*{w.lower()}_pp_registered.nii")
        labels = [1 for _ in paths]

        # Raise warning if no files are found
        if len(paths) == 0:
            raise RuntimeWarning(f"No files found for MSSEG2015")

        return paths, labels

    @staticmethod
    def returnWMH(**kwargs):
        root = os.path.join(DATAROOT, 'WMH')
        w = kwargs['weighting']
        paths = glob(
            f"{root}/*/*/orig/{w.upper()}_stripped_registered.nii.gz")
        labels = [1 for _ in paths]

        # Raise warning if no files are found
        if len(paths) == 0:
            raise RuntimeWarning(f"No files found for WMH")

        return paths, labels


def load_mr_scan(path: str, img_size: int = None, equalize: bool = False, slices_lower_upper: tuple = None):
    """Load an MR image in the Nifti format from path and it's corresponding
    segmentation from 'anomaly_segmentation.nii.gz' in the same folder if
    available.

    Args:
        path (str): Path to MR image
        img_size (int): Optional. Size of the loaded image
        n_clahe (bool): Perform N-CLAHE histogram normalization
        slices_lower_upper (tuple): Lower and upper indices for slices

    Returns:
        sample (torch.tensor): Loaded MR image as short, shape [c, h, d]
        segmentation (torch.tensor): Segmentation as short, shape [c, h, d]
    """
    # Load mri scan
    sample = nii_loader(path, dtype='float32', size=img_size)

    # Load segmentation if available else 0s
    sp = f"{path[:path.rfind('/')]}/anomaly_segmentation.nii.gz"
    if os.path.exists(sp):
        segmentation = nii_loader(sp, dtype='float32', size=img_size)
        # Binarize segmentation at a threshold, see discussion in paper
        # Section 3.2 Pre-processing
        segmentation = torch.where(segmentation > 0.9, 1., 0.)
    else:
        segmentation = torch.zeros_like(sample)
        warn(f"No segmentation found at {sp}")

    # Apply histogram equalization
    if equalize:
        sample = histogram_equalization(sample)

    # Samples are shape [1, slices, height, width]
    if slices_lower_upper is not None:
        brain_inds = slice(*slices_lower_upper)
        sample = sample[:, brain_inds]
        segmentation = segmentation[:, brain_inds]

    return sample, segmentation


def nii_loader(path: str, dtype: str = 'float32', size: int = None):
    """Load a neuroimaging file with nibabel
    https://nipy.org/nibabel/reference/nibabel.html

    Args:
        dtype (str): Optional. Datatype of the loaded volume
        size (int): Optional. Output size for h and w. Only supports rectangles

    Returns:
        volume (np.array): Of shape [1, slices, h, w]
    """
    # Load file
    data = nib.load(path, keep_file_open=False)
    volume = data.get_fdata(caching='unchanged',
                            dtype=np.float32).astype(np.dtype(dtype))
    # Convert to tensor and slices first
    if volume.ndim == 4:
        volume = volume.squeeze(-1)
    volume = torch.Tensor(volume).permute(2, 0, 1).unsqueeze(0)
    # Resize if size is given
    if size is not None:
        volume = CenterCrop3D()(volume)  # Has no effect in SRI space since it's already 240x240
        volume = ResizeGray(size=[volume.shape[1], size, size])(volume)
    return volume
