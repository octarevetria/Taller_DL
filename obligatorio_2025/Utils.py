import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def visualizar_predicciones(model, val_loader, device, checkpoint_path="best_unet.pth", n_show=4):
    """
    Permite visualizar imagenes, mascaras y predicciones, en ese orden
    """

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    imgs, masks = next(iter(val_loader))
    imgs, masks = imgs.to(device), masks.to(device)

    with torch.no_grad():
        outputs = model(imgs)
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

    # Pasamos a CPU para mostrar
    imgs_np  = imgs.cpu().permute(0, 2, 3, 1).numpy()
    masks_np = masks.cpu().squeeze(1).numpy()
    preds_np = preds.cpu().squeeze(1).numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    def unnormalize(img):
        return np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)

    n_show = min(n_show, imgs_np.shape[0])
    plt.figure(figsize=(12, 3 * n_show))

    for i in range(n_show):
        img_show = unnormalize(imgs_np[i])

        plt.subplot(n_show, 3, 3*i + 1)
        plt.imshow(img_show)
        plt.title("Imagen")
        plt.axis("off")

        plt.subplot(n_show, 3, 3*i + 2)
        plt.imshow(masks_np[i], cmap="gray")
        plt.title("Mascara real")
        plt.axis("off")

        plt.subplot(n_show, 3, 3*i + 3)
        plt.imshow(preds_np[i], cmap="gray")
        plt.title("Prediccion")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def train(model,
          train_loader,
          val_loader,
          optimizer,
          criterion,
          device,
          num_epochs=20,
          best_model_path=None):
    """
    Return:
        history: dict con listas:
                 - 'train_loss', 'val_loss'
                 - 'train_dice', 'val_dice'
    """

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": []
    }

    best_val_dice = -1.0

    for epoch in tqdm(range(1, num_epochs + 1)):
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_dice = validate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Nuevo mejor modelo guardado con Val Dice: {best_val_dice:.4f}")



    return history

def dice_coef(preds, targets, eps=1e-7):
    """
    preds: logits (salida del modelo, sin sigmoide)
    targets: mascaras binarias [0,1]
    """
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    n_batches = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_dice += dice_coef(outputs.detach(), masks.detach())
        n_batches += 1

    return running_loss / n_batches, running_dice / n_batches


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    n_batches = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            running_loss += loss.item()
            running_dice += dice_coef(outputs, masks)
            n_batches += 1

    return running_loss / n_batches, running_dice / n_batches

def train_full(model, loader, optimizer, criterion, device, num_epochs=5):
    """
    Reentrena el modelo usando todo el conjunto de entrenamiento.
    Parte de los pesos cargados previamente.
    No captura metricas de validacion
    """
    history = {
        "train_loss": [],
        "train_dice": []
    }

    for epoch in tqdm(range(1, num_epochs + 1)):
        train_loss, train_dice = train_one_epoch(
            model, loader, optimizer, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)

        print(f"[FULL TRAIN] Epoch {epoch}/{num_epochs} | "
              f"Loss: {train_loss:.4f} | Dice: {train_dice:.4f}")

    return history

def mask_to_rle(mask: np.ndarray) -> str:

    if mask.ndim == 3:
        mask = mask.squeeze()

    mask = mask.astype(np.uint8)

    pixels = mask.T.flatten() # Orden Fortran como pide la letra

    pixels = np.concatenate([[0], pixels, [0]])

    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1 
    """
    Se desfasan para comparar cambios de valores y luego se ajusta con el +1
    Queda un array bool con True donde hay cambios de valor
    """
    starts = changes[::2]
    ends   = changes[1::2]
    lengths = ends - starts

    rle = " ".join(str(x) for pair in zip(starts, lengths) for x in pair)
    return rle

import os
import pandas as pd

def save_history_to_csv(history, metrics_dir, run_name):

    os.makedirs(metrics_dir, exist_ok=True)

    filename = f"metrics_{run_name}.csv"
    metrics_path = os.path.join(metrics_dir, filename)

    num_epochs = len(history["train_loss"])

    metrics_df = pd.DataFrame({
        "epoch": list(range(1, num_epochs + 1)),
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "train_dice": history["train_dice"],
        "val_dice": history["val_dice"],
    })

    metrics_df.to_csv(metrics_path, index=False)

    print("Archivo guardado en:", metrics_path)
    return metrics_df

def collect_val_examples_with_dice(model, device, loader, max_batches=None):
    model.eval()
    images_list = []
    masks_list = []
    preds_list = []
    dices_list = []

    with torch.no_grad():
        for b_idx, (imgs, masks) in enumerate(loader):
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            probs = torch.sigmoid(preds)
            preds_bin = (probs > 0.5).float()

            intersection = (preds_bin * masks).sum(dim=(2,3))
            union = preds_bin.sum(dim=(2,3)) + masks.sum(dim=(2,3))
            batch_dice = (2 * intersection + 1e-7) / (union + 1e-7)

            for i in range(imgs.size(0)):
                images_list.append(imgs[i].cpu())
                masks_list.append(masks[i].cpu())
                preds_list.append(preds_bin[i].cpu())
                dices_list.append(batch_dice[i].item())

            if max_batches is not None and (b_idx + 1) >= max_batches:
                break

    return images_list, masks_list, preds_list, np.array(dices_list)
