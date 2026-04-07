###########################
# Import Python Packages
###########################
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms_v2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from model import UNET
from dataset import CustomDataset
from dl_utils import train_one_epoch, test, dice_score, accuracy_score

####################
# Hyperparameters
####################
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

BATCH_SIZE = 32         # TODO: Change this as you see fit
NUM_EPOCHS = 10         # TODO: Change this as you see fit
LEARNING_RATE = 1e-3    # TODO: Change this as you see fit
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
DATA_DIR = "./oxford-pet-dataset"
NUM_CLASSES = 3  # Number of segmentation classes
CHECKPOINT_PATH = "model_checkpoint.pth"

####################
# Dataset
####################
# Define Augmentations for Training
train_tran = transforms_v2.Compose([
    transforms_v2.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms_v2.RandomHorizontalFlip(p=0.5),
    transforms_v2.RandomRotation(degrees=15),
    transforms_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms_v2.RandomResizedCrop(size=(IMAGE_HEIGHT, IMAGE_WIDTH), scale=(0.8, 1.0)),
    transforms_v2.ToImage(),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define Transformations for Validation & Test (No Augmentation)
test_tran = transforms_v2.Compose([
    transforms_v2.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms_v2.ToImage(),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
# TODO: Load dataset
train_ds = CustomDataset(os.path.join(DATA_DIR, 'train_img'),os.path.join(DATA_DIR, 'train_mask'), transform=test_tran)
valid_ds = CustomDataset(os.path.join(DATA_DIR, 'valid_img'),os.path.join(DATA_DIR, 'valid_mask'), transform=test_tran)
test_ds = CustomDataset(os.path.join(DATA_DIR, 'test_img'),os.path.join(DATA_DIR, 'test_mask'), transform=test_tran)

# TODO: Define DataLoaders
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train set: {len(train_ds)} samples")
print(f"Validation set: {len(valid_ds)} samples")
print(f"Test set: {len(test_ds)} samples")


####################
# Model
####################
device = "cuda" if torch.cuda.is_available() else "cpu"
# TODO: Create a UNet model
model = UNET(out_channels=NUM_CLASSES).to(device)
print(model)

# TODO: Define loss function
loss_fn = nn.CrossEntropyLoss()

# TODO: Define optimizermodel = UNET().to(device)
# model.load_state_dict(torch.load("model_best.pth"))
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Setup TensorBoard
writer = SummaryWriter(f'./runs/trainer_{model._get_name()}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

####################
# Load Checkpoint (if exists)
####################
start_epoch = 0
best_dice_score = 0.0  # Higher is better

if os.path.exists(CHECKPOINT_PATH):
    print(f"\nLoading checkpoint from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_dice_score = checkpoint["best_dice_score"]
    print(f"Resuming training from epoch {start_epoch} (Best Dice Score: {best_dice_score:.4f})")

####################
# Model Training
####################
for epoch in range(start_epoch, NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    
    # Train for one epoch
    train_one_epoch(train_dl, model, loss_fn, optimizer, epoch, DEVICE, writer, log_step_interval=5)
    
    # Evaluate on training set
    train_loss, train_preds, train_targets = test(valid_dl, model, loss_fn, DEVICE)

    # Compute training performance metrics
    train_dice = dice_score(train_preds, train_targets, num_classes=NUM_CLASSES)
    train_acc = accuracy_score(train_preds, train_targets)

    # Evaluate on validation set
    val_loss, val_preds, val_targets = test(valid_dl, model, loss_fn, DEVICE)

    # Compute validation performance metrics
    val_dice = dice_score(val_preds, val_targets, num_classes=NUM_CLASSES)
    val_acc = accuracy_score(val_preds, val_targets)

    print(f"Train Dice Score: {train_dice:.4f} | Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Dice Score: {val_dice:.4f} | Validation Accuracy: {val_acc:.2f}%")

    # Log metrics to TensorBoard (Track Train vs Validation)
    writer.add_scalars('Train vs. Valid/loss', {'train': train_loss, 'valid': val_loss}, epoch)
    writer.add_scalars('Train vs. Valid/dice', {'train': train_dice, 'valid': val_dice}, epoch)
    writer.add_scalars('Train vs. Valid/acc', {'train': train_acc, 'valid': val_acc}, epoch)

    # Save model checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_dice_score": best_dice_score
    }
    torch.save(checkpoint, CHECKPOINT_PATH)

    # Save the best model based on Dice Score
    if val_dice > best_dice_score:
        best_dice_score = val_dice
        torch.save(model.state_dict(), 'model_best.pth')
        print("Saved best model to model_best.pth")

    ####################
    # Visualize Predictions
    ####################
    model.eval()
    imgs, masks = next(iter(valid_dl))
    imgs = imgs.to(DEVICE)
    logits = model(imgs)
    pred_masks = torch.argmax(logits, dim=1)

    # Save example segmentation output
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].imshow(imgs[0].cpu().numpy().transpose(1, 2, 0))
    axs[1].imshow(masks[0].cpu().numpy())
    axs[2].imshow(pred_masks[0].cpu().numpy())
    plt.savefig(f'pred_semantic_seg_epoch_{epoch}.jpg')
    plt.close()

print("Training Complete!")


####################
# Model Evaluation
####################

# TODO: Load the best model
model.load_state_dict(torch.load("model_best.pth"))
model.to(device)

# Evaluate on test set
test_loss, test_preds, test_targets = test(test_dl, model, loss_fn, DEVICE)

# Compute test performance metrics
test_dice = dice_score(test_preds, test_targets, num_classes=NUM_CLASSES)
test_acc = accuracy_score(test_preds, test_targets)

print(f"Test Dice Score: {test_dice:.4f} | Accuracy: {test_acc:.2f}%")


####################
# Visualize Predictions
####################
model.eval()
imgs, masks = next(iter(valid_dl))
imgs = imgs.to(DEVICE)
logits = model(imgs)
pred_masks = torch.argmax(logits, dim=1)

# Save example segmentation output
fig, axs = plt.subplots(1, 3, figsize=(12, 6))
axs[0].imshow(imgs[0].cpu().numpy().transpose(1, 2, 0))
axs[1].imshow(masks[0].cpu().numpy())
axs[2].imshow(pred_masks[0].cpu().numpy())
plt.savefig('pred_semantic_seg.jpg')
plt.close()
