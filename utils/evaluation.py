import warnings

import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
import torch
from tqdm import tqdm
from torchvision.utils import make_grid

from utils.utils import connected_components_3d, torch2np_img


def plot_roc(fpr, tpr, auroc, title=""):
    """Returns a plot of the reciever operating characteristics (ROC)

    Args:
        fpr (array): false positives per threshold
        tpr (array): true positives per threshold
        auroc (float): area under ROC curve
        title (str): Title of plot

    Returns:
        fig (matplotlib.figure.Figure): Finished plot
    """

    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.title(title)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auroc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return fig


def plot_results(images: list, titles: list, n_images: int = 25):
    """Returns a plot containing the input images, reconstructed images,
    uncertainty maps and anomaly maps"""

    if len(images) != len(titles):
        raise RuntimeError("Not the same number of images and titles")

    # Stack tensors to grid image and transform to numpy for plotting
    img_dict = {}
    for img, title in zip(images, titles):
        img_grid = make_grid(
            img[:n_images].float(), nrow=1, normalize=True, scale_each=True)
        img_grid = torch2np_img(img_grid)
        img_dict[title] = img_grid

    n = len(images)

    # Construct matplotlib figure
    fig = plt.figure(figsize=(3 * n, 1 * n_images))
    plt.axis("off")
    for i, key in enumerate(img_dict.keys()):
        a = fig.add_subplot(1, n, i + 1)
        plt.imshow(img_dict[key])
        a.set_title(key)

    return fig


def compute_auroc(predictions, targets):
    """Compute the area under reciever operating characteristics curve.
    If the label is a scalar value, we measure the detection performance,
    else the segmentation performance.

    Args:
        predictions (torch.tensor): Predicted anomaly map. Shape [b, c, h, w]
        targets (torch.tensor): Target label [b] or segmentation map [b, c, h, w]
    Returns:
        auroc (float)
    """

    # Safety check. Can't compute auroc with no positive targets
    if targets.sum() == 0:
        warnings.warn("Can't compute auroc with only negative target values, "
                      "returning 0.5")
        auroc = 0.5
    else:
        auroc = roc_auc_score(targets.view(-1), predictions.view(-1))
        # auroc = roc_auc_score(targets.flatten(), predictions.flatten())
    return auroc


def compute_roc(predictions, targets):
    """Compute the reciever operating characteristics curve.
    If the label is a scalar value, we measure the detection performance,
    else the segmentation performance.

    Args:
        predictions (torch.tensor): Predicted anomaly map. Shape [b, c, h, w]
        targets (torch.tensor): Target label [b] or segmentation map [b, c, h, w]
    Returns:
        fpr (np.array): False positive rate
        tpd (np.array): True positive rate
        thresholds (np.array)
    """
    fpr, tpr, thresholds = roc_curve(targets.view(-1), predictions.view(-1))
    return fpr, tpr, thresholds


def compute_best_dice(preds, targets, n_thresh=100):
    """Compute the best dice score between an anomaly map and the ground truth
    segmentation using a greedy binary search with depth search_depth

    Args:
        preds (torch.tensor): Predicted binary anomaly map. Shape [b, c, h, w]
        targets (torch.tensor): Target label [b] or segmentation map [b, c, h, w]
        n_thresh (int): Number of thresholds to try
    Returns:
        max_dice (float): Maximal dice score
        max_thresh (float): Threshold corresponding to maximal dice score
    """
    if targets.ndim == 1:
        warnings.warn("Can't compute a meaningful dice score with only"
                      "labels, returning 0.")
        return 0., 0.

    thresholds = np.linspace(preds.min(), preds.max(), n_thresh)
    threshs = []
    scores = []
    pbar = tqdm(thresholds, desc="DICE search")
    for t in pbar:
        dice = compute_dice(torch.where(preds > t, 1., 0.), targets)
        scores.append(dice)
        threshs.append(t)

    scores = torch.stack(scores, 0)
    # max_dice = scores.max()
    max_thresh = threshs[scores.argmax()]

    # Get best dice once again after connected component analysis
    bin_preds = torch.where(preds > max_thresh, 1., 0.)
    bin_preds = connected_components_3d(bin_preds)
    max_dice = compute_dice(bin_preds, targets)
    return max_dice, max_thresh


def compute_dice_fpr(preds, targets, max_fprs=[0.01, 0.05, 0.1]):
    fprs, _, thresholds = compute_roc(preds, targets)
    dices = []
    for max_fpr in max_fprs:
        th = thresholds[fprs < max_fpr][-1]
        bin_preds = torch.where(preds > th, 1., 0.)
        bin_preds = connected_components_3d(bin_preds)
        dice = compute_dice(bin_preds, targets)
        dices.append(dice)
        print(f"DICE{int(max_fpr * 100)}: {dice:.4f}, threshold: {th:.4f}")
    return dices


def compute_dice(predictions, targets) -> float:
    """Compute the DICE score. This only works for segmentations.
    PREDICTIONS NEED TO BE BINARY!

    Args:
        predictions (torch.tensor): Predicted binary anomaly map. Shape [b, c, h, w]
        targets (torch.tensor): Target label [b] or segmentation map [b, c, h, w]
    Returns:
        dice (float)
    """
    if (predictions - predictions.int()).sum() > 0.:
        raise RuntimeError("predictions for DICE score must be binary")
    if (targets - targets.int()).sum() > 0.:
        raise RuntimeError("targets for DICE score must be binary")

    pred_sum = predictions.view(-1).sum()
    targ_sum = targets.view(-1).sum()
    intersection = predictions.view(-1).float() @ targets.view(-1).float()
    # pred_sum = predictions.flatten().sum()
    # targ_sum = targets.flatten().sum()
    # intersection = predictions.flatten().float() @ targets.flatten().float()
    dice = (2 * intersection) / (pred_sum + targ_sum)
    return dice


def compute_pro_auc(predictions, targets, expect_fpr=0.3, max_steps=300):
    """Computes the PRO-score and intersection over union (IOU)
    Code from: https://github.com/YoungGod/DFR/blob/master/DFR-source/anoseg_dfr.py
    """
    if targets.ndim == 1:
        warnings.warn("Can't compute a meaningful pro score with only"
                      "labels, returning 0.")
        return 0.

    def rescale(x):
        return (x - x.min()) / (x.max() - x.min())

    if torch.is_tensor(predictions):
        predictions = predictions.numpy()
    if torch.is_tensor(targets):
        targets = targets.numpy()

    # Squeeze away channel dimension
    predictions = predictions.squeeze(1)
    targets = targets.squeeze(1)

    # Binarize target segmentations
    targets[targets <= 0.5] = 0
    targets[targets > 0.5] = 1
    targets = targets.astype(np.bool)

    # Maximum and minimum thresholdsmax_th = scores.max()
    max_th = predictions.max()
    min_th = predictions.min()
    delta = (max_th - min_th) / max_steps

    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(predictions, dtype=np.bool)
    for step in tqdm(range(max_steps), desc="PRO AUC"):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[predictions <= thred] = 0
        binary_score_maps[predictions > thred] = 1

        # Connected component analysis
        # binary_score_maps = connected_components_3d(binary_score_maps)

        pro = []    # per region overlap
        iou = []    # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):    # for i th image
            # pro (per region level)
            label_map = measure.label(targets[i], connectivity=2)
            props = measure.regionprops(label_map)
            for prop in props:
                # find the bounding box of an anomaly region
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = targets[i][x_min:x_max, y_min:y_max]   # bug!
                cropped_targets = prop.filled_image    # corrected!
                intersection = np.logical_and(
                    cropped_pred_label, cropped_targets).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(
                binary_score_maps[i], targets[i]).astype(np.float32).sum()
            union = np.logical_or(
                binary_score_maps[i], targets[i]).astype(np.float32).sum()
            if targets[i].any() > 0:    # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
#             print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        targets_neg = ~targets
        fpr = np.logical_and(
            targets_neg, binary_score_maps).sum() / targets_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    ious_mean = np.array(ious_mean)
    ious_std = np.array(ious_std)

    # best per image iou
    best_miou = ious_mean.max()
    print(f"Best IOU: {best_miou:.4f}")

    # default 30% fpr vs pro, pro_auc
    # find the indexs of fprs that is less than expect_fpr (default 0.3)
    idx = fprs <= expect_fpr
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)    # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    print(f"pro auc ({int(expect_fpr*100)}% FPR): {pro_auc_score:.4f}")

    return pro_auc_score, best_miou


def compute_aupr(predictions, targets):
    """Compute the area under the precision-recall curve

    Args:
        predictions (torch.tensor): Anomaly scores
        targets (torch.tensor): Segmentation map, must be binary
    """
    precision, recall, _ = precision_recall_curve(targets.view(-1), predictions.view(-1))
    # precision, recall, _ = precision_recall_curve(targets.flatten(), predictions.flatten())
    aupr = auc(recall, precision)
    return aupr


def evaluate(predictions, targets, auroc=True, dice=True, auprc=True, proauc=True, n_thresh_dice=100):
    # compute_dice_fpr(predictions, targets, masks)
    if auroc:
        auroc = compute_auroc(predictions, targets)
        print(f"AUROC: {auroc:.4f}")
    else:
        auroc = 0.0

    if auprc:
        auprc = compute_aupr(predictions, targets)
        print(f"AUPRC: {auprc:.4f}")
    else:
        auprc = 0.0

    if dice:
        dice, th = compute_best_dice(predictions, targets, n_thresh=n_thresh_dice)
        print(f"DICE: {dice:.4f}, best threshold: {th:.4f}")
    else:
        dice = 0.0
        th = None

    if proauc:
        h, w = predictions.shape[-2:]
        compute_pro_auc(
            predictions=predictions.view(-1, 1, h, w),
            targets=targets.view(-1, 1, h, w),
        )

    return auroc, auprc, dice, th
