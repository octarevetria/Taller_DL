import copy
import torch
from sklearn.metrics import accuracy_score, f1_score

def train_w_early_stopping(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, device, wandb_run=None):
    best_f1 = 0.0
    epochs_no_improve = 0
    best_val_f1 = 0.0
    best_model_state = None
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")

        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f} - Val Acc: {acc:.4f} - Val F1 (w): {val_f1:.4f}")

        if wandb_run is not None:
            wandb_run.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": acc,
            "val_f1": val_f1
            })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, "best_model_f1.pth") 
            print(f"New best F1: {best_val_f1:.4f} (model saved)")
        else:
            epochs_no_improve += 1
            print(f" No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best model loaded (best F1 = {best_val_f1:.4f})")
    else:
        print("No best model state found.")
    return model