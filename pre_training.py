import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import torch
import torch.nn as nn # Added for DataParallel
from torchvision import transforms
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from utils import DataPreperation, ScaleAugmentation, ScaleToLimitRange, LLocal
from DataModule import DataModule, collate_fn
from Model import Model
from BarlowTwins import BarlowTwins
from tqdm import tqdm
import torch.optim as optim
import logging
from datetime import datetime

# ---------- Logging Setup ----------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Create a file handler with better flushing behavior
file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# ---------- H-Params ----------
K_MIN = 0.7
K_MAX = 1.4
H_LO, H_HI = 16, 256
W_LO, W_HI = 16, 1024
batch_size = 16
total_steps = 500000      # Total number of training iterations
log_interval = 1000      # Log and save every N iterations

# ---------- Transformations ----------
cv2_transforms = transforms.Compose([
    ScaleAugmentation(K_MIN, K_MAX),
    ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI)
])

pil_transforms = transforms.Compose([
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.2)
    ], p=0.5),
    transforms.ToTensor()
])

# ---------- Dataset and Path Preparation ----------
root_path = "/ssd_scratch/shree14/Train_Images_Lines/"
model_path = "/home2/shree.mitra23m/WACV_Reb/"
os.makedirs(model_path, exist_ok=True)

# --- MODIFICATION: Create a dedicated directory for checkpoints ---
checkpoint_dir = os.path.join(model_path, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
# --- END MODIFICATION ---

logging.info("Preparing dataset...")
data = DataPreperation(root_path=root_path)
train_df = data.Train_Test_Data()
dm = DataModule(bs=batch_size, root_path=root_path, df=train_df,
                 cv2_transforms=cv2_transforms, pil_transforms=pil_transforms)
train_loader = DataLoader(dm, batch_size=batch_size, drop_last=True, shuffle=True,
                          collate_fn=collate_fn, num_workers=4)

# ---------- Model Setup ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model(growth_rate=24, num_layers=16, reduction=0.5)

# Add DataParallel for multi-GPU support
if torch.cuda.device_count() > 1:
  logging.info(f"Using {torch.cuda.device_count()} GPUs!")
  model = nn.DataParallel(model)

model.to(device)

criterion = BarlowTwins(batch_size=batch_size)
optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)

# ---------- Iteration-Based Training Loop ----------
step = 0
train_losses = []
save_name_pre = 'BarlowTwins_HTR'

logging.info("Starting iteration-based training...")

while step < total_steps:
    for x_1, x_2, mask in train_loader:
        model.train()
        x_1, x_2, mask = x_1.to(device, non_blocking=True), x_2.to(device, non_blocking=True), mask.to(device, non_blocking=True)

        _, z1, z1_first, _ = model(x_1, mask)
        _, z2, z2_first, _ = model(x_2, mask)

        global_loss = criterion(z1, z2)
        local_loss = LLocal(z1_first, z2_first)
        loss = 0.6 * local_loss + 0.4 * global_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        step += 1

        if step % 100 == 0:
            log_message = f"Step {step}/{total_steps} - Loss: {loss.item():.4f}"
            tqdm.write(log_message)
            logging.info(log_message)

        if step % log_interval == 0 or step == total_steps:
            # --- MODIFICATION: Save checkpoints to the new directory ---
            backbone_path = os.path.join(checkpoint_dir, f"{save_name_pre}_backbone_step{step}.pth")
            full_model_path = os.path.join(checkpoint_dir, f"{save_name_pre}_full_step{step}.pth")
            # --- END MODIFICATION ---

            # Handle saving for DataParallel wrapper
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.backbone.state_dict(), backbone_path)
                torch.save(model.module.state_dict(), full_model_path)
            else:
                torch.save(model.backbone.state_dict(), backbone_path)
                torch.save(model.state_dict(), full_model_path)
            
            logging.info(f"Saved checkpoint at step {step} to {checkpoint_dir}")
            scheduler.step(loss.item())

        if step >= total_steps:
            break

# ---------- Final Save ----------
# --- MODIFICATION: Save final checkpoints to the new directory ---
final_backbone_path = os.path.join(checkpoint_dir, f"{save_name_pre}_final_backbone.pth")
final_full_model_path = os.path.join(checkpoint_dir, f"{save_name_pre}_final_full_model.pth")
# --- END MODIFICATION ---

# Handle saving for DataParallel wrapper
if isinstance(model, nn.DataParallel):
    torch.save(model.module.backbone.state_dict(), final_backbone_path)
    torch.save(model.module.state_dict(), final_full_model_path)
else:
    torch.save(model.backbone.state_dict(), final_backbone_path)
    torch.save(model.state_dict(), final_full_model_path)

logging.info("Final model and backbone saved.")

# ---------- Save Losses ----------
loss_file_path = os.path.join(model_path, "loss.txt")
with open(loss_file_path, "w") as f:
    for l in train_losses:
        f.write(f"{l}\n")
logging.info(f"Losses saved to {loss_file_path}")