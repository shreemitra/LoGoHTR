import os
import pandas as pd
import numpy as np
import cv2
import random
from PIL import ImageFilter, ImageOps
import torch
import torch.nn.functional as F
class DataPreperation():
    def __init__(self, root_path):
        self.root_path = root_path
        #self.files = []
        self.image = []
        #self.output = []
        #self.data_image = []
        #self.data_output = []

    def Train_Test_Data(self):
      count = 0
      print("Data Preperation Initiated!!!")

      for i in os.listdir(self.root_path):
        count+=1
        self.image.append(i)
        if (count % 10000 == 0 ):
          print("Number of Data Prepared =====>{}".format(count))

      self.df_train = pd.DataFrame(list(zip(self.image)),columns =['Image'])

      return self.df_train
    
class ScaleToLimitRange:
    def __init__(self, w_lo: int, w_hi: int, h_lo: int, h_hi: int) -> None:
        assert w_lo <= w_hi and h_lo <= h_hi
        self.w_lo = w_lo
        self.w_hi = w_hi
        self.h_lo = h_lo
        self.h_hi = h_hi

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        r = h / w
        lo_r = self.h_lo / self.w_hi
        hi_r = self.h_hi / self.w_lo
        assert lo_r <= h / w <= hi_r, f"img ratio h:w {r} not in range [{lo_r}, {hi_r}]"

        scale_r = min(self.h_hi / h, self.w_hi / w)
        if scale_r < 1.0:
            img = cv2.resize(img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_LINEAR)
            return img

        scale_r = max(self.h_lo / h, self.w_lo / w)
        if scale_r > 1.0:
            img = cv2.resize(img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_LINEAR)
            return img

        assert self.h_lo <= h <= self.h_hi and self.w_lo <= w <= self.w_hi
        return img


class ScaleAugmentation:
    def __init__(self, lo: float, hi: float) -> None:
        assert lo <= hi
        self.lo = lo
        self.hi = hi

    def __call__(self, img: np.ndarray) -> np.ndarray:
        k = np.random.uniform(self.lo, self.hi)
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
        return img
    
class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        
def LLocal(x_aug1, x_aug2, temperature=0.5, K=5, overlap=0.5):
    """
    Computes the local loss for patch-based self-supervised learning under the Barlow Twins setup.
    
    Args:
        x_aug1: Feature map from the encoder for the first augmented view. Shape: [B, C, H, W]
        x_aug2: Feature map from the encoder for the second augmented view. Shape: [B, C, H, W]
        temperature: Scaling factor for logits.
        K: Number of patches along one dimension.
        overlap: Fractional overlap between patches (0 for non-overlapping).
        
    Returns:
        Local loss as a scalar tensor.
    """
    B, C, H, W = x_aug1.shape
    stride = int((1 - overlap) * (H // K))  # Stride for overlapping patches
    patch_size = H // K  # Size of each patch

    # Extract patches from feature maps
    def extract_patches(feature_map):
        patches = []
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                patch = F.adaptive_avg_pool2d(
                    feature_map[:, :, i:i+patch_size, j:j+patch_size], 
                    1
                ).squeeze(-1).squeeze(-1)  # Global pooling to [B, C]
                patches.append(patch)
        return torch.stack(patches, dim=1)  # Shape: [B, num_patches, C]

    patches_1 = extract_patches(x_aug1)  # Patches from the first augmentation
    patches_2 = extract_patches(x_aug2)  # Patches from the second augmentation

    # Normalize patches for cosine similarity
    patches_1 = F.normalize(patches_1, dim=-1)
    patches_2 = F.normalize(patches_2, dim=-1)

    # Compute positive pairs (same spatial positions)
    pos_pairs = torch.sum(patches_1 * patches_2, dim=-1)  # Shape: [B, num_patches]

    # Compute negative pairs (different spatial positions within the same view)
    neg_pairs_1 = torch.matmul(patches_1, patches_1.transpose(1, 2))  # [B, num_patches, num_patches]
    neg_pairs_2 = torch.matmul(patches_2, patches_2.transpose(1, 2))  # [B, num_patches, num_patches]

    # Remove self-similarities (diagonal entries)
    neg_pairs_1 = neg_pairs_1 - torch.eye(neg_pairs_1.size(1), device=neg_pairs_1.device).unsqueeze(0)
    neg_pairs_2 = neg_pairs_2 - torch.eye(neg_pairs_2.size(1), device=neg_pairs_2.device).unsqueeze(0)

    # Concatenate positive and negative pairs
    logits = torch.cat([pos_pairs.unsqueeze(-1), neg_pairs_1, neg_pairs_2], dim=-1) / temperature  # Shape: [B, num_patches, 1 + num_negatives]
    
    # Labels: Positive class is index 0 for cross-entropy
    labels = torch.zeros(B * logits.size(1), dtype=torch.long, device=logits.device)

    # Compute cross-entropy loss
    logits = logits.view(-1, logits.size(-1))  # Flatten for cross-entropy
    loss = F.cross_entropy(logits, labels)

    return loss
