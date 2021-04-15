import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import equalize_hist
from skimage.measure import label, regionprops
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

""" General utilities """


def torch2np_img(img: torch.Tensor):
    """Converts a pytorch image to a cv2 RGB image

    Args:
        img (torch.Tensor): range (-1, 1), dtype torch.float32, shape [C, H, W]

    Returns:
        img (np.array): range(0, 255), dtype np.uint8, shape (H, W, C)
    """
    return (img.permute(1, 2, 0).numpy() * 255.).astype(np.uint8)


def plot_img(img):
    """Plot a torch tensor with shape [n, c, h, w], [c, h, w] or [h, w]"""
    if not torch.is_tensor(img):
        img = torch.from_numpy(img)
    img = img.detach().cpu()
    if img.ndim == 2:
        img = img.unsqueeze(0)
    if img.ndim == 3:
        img = img.unsqueeze(0)
    img_grid = make_grid(img, normalize=False, scale_each=False)
    plt.imshow(img_grid.permute(1, 2, 0))
    # plt.axis('off')
    plt.show()


def save_img(img, f):
    """Save a torch tensor with shape [n, c, h, w], [c, h, w] or [h, w]"""
    if not torch.is_tensor(img):
        img = torch.from_numpy(img)
    img = img.detach().cpu()
    if img.ndim == 2:
        img = img.unsqueeze(0)
    if img.ndim == 3:
        img = img.unsqueeze(0)
    img_grid = make_grid(img, normalize=True, scale_each=True)
    save_image(img_grid, f)


""" Data normalization and augmentation functions """


class CenterCrop3D:
    """Center crop a volume with shape [channels, slices, height, width] to a
    rectangle in height and width
    """
    @staticmethod
    def __call__(volume):
        _, _, h, w = volume.shape
        # If the volume is already a rectangle, just return it
        if h == w:
            return volume
        # Else we need to crop along the longer side
        min_side = min(h, w)
        lower = min_side // 2
        upper = min_side // 2 if min_side % 2 == 0 else min_side // 2 + 1
        if h < w:
            center = w // 2
            bottom = center - lower
            top = center + upper
            volume = volume[:, :, :, bottom:top]
        else:
            center = h // 2
            bottom = center - lower
            top = center + upper
            volume = volume[:, :, bottom:top, :]
        return volume


class ResizeGray:
    def __init__(self, size, mode='nearest', align_corners=None):
        """Resample a tensor of shape [c, slices, h, w], or [c, h, w] to size
        Arguments are the same as in torch.nn.functional.interpolate, but we
        don't need a batch- or channel dimension here.
        The datatype can only be preserved when using nearest neighbor.

        Example:
        volume = torch.randn(1, 189, 197, 197)
        out = ResizeGray()(volume, size=[189, 120, 120])
        out.shape = [1, 189, 120, 120]
        out.dtype = volume.dtype if mode == 'nearest' else torch.float32
        """
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, volume):
        dtype = volume.dtype
        out = F.interpolate(volume[None].float(), size=self.size,
                            mode=self.mode,
                            align_corners=self.align_corners)[0]
        if self.mode == 'nearest':
            out = out.type(dtype)
        return out


def histogram_equalization(img):
    # Take care of torch tensors
    batch_dim = img.ndim == 4
    is_torch = torch.is_tensor(img)
    if batch_dim:
        img = img.squeeze(0)
    if is_torch:
        img = img.numpy()

    # Create equalization mask
    mask = np.zeros_like(img)
    mask[img > 0] = 1

    # Equalize
    img = equalize_hist(img.astype(np.long), nbins=256, mask=mask)

    # Assure that background still is 0
    img *= mask

    # Take care of torch tensors again
    if is_torch:
        img = torch.Tensor(img)
    if batch_dim:
        img = img.unsqueeze(0)

    return img


""" Others """


def connected_components_3d(volume):
    is_batch = True
    is_torch = torch.is_tensor(volume)
    if is_torch:
        volume = volume.numpy()
    if volume.ndim == 3:
        volume = volume.unsqueeze(0)
        is_batch = False

    # shape [b, d, h, w], treat every sample in batch independently
    pbar = tqdm(range(len(volume)), desc="Connected components")
    for i in pbar:
        cc_volume = label(volume[i], connectivity=3)
        props = regionprops(cc_volume)
        for prop in props:
            if prop['filled_area'] <= 20:
                volume[i, cc_volume == prop['label']] = 0

    if not is_batch:
        volume = volume.squeeze(0)
    if is_torch:
        volume = torch.from_numpy(volume)
    return volume
