import torch
from dataset import BlurDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split


def save_checkpoint(state, filename="my_checkpount.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
        train_dir,
        train_blur_dir,
        batch_size,
        train_transform,
        num_workers=2,
        pin_memory=True,
):
    train_ds = BlurDataset(train_dir, train_blur_dir, train_transform)
    lengths = [int(len(train_ds)*0.9), int(len(train_ds)*0.1)]
    train_ds, val_ds = random_split(train_ds, lengths)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = preds.float()
            num_correct = (preds == y).sum()
            num_pixels += torch.numel(preds)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    model.train()

