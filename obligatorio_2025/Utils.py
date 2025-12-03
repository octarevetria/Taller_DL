import torch
import numpy as np
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

        if best_model_path is not None and val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
        )

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

    for epoch in range(1, num_epochs + 1):
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
