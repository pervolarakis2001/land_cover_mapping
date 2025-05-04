import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import os


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_model_state = None
        self.verbose = verbose

    def __call__(self, val_loss, model=None):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None:
                self.best_model_state = model.state_dict()
            if self.verbose:
                print(f"New best loss: {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f" No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")

    def restore_best_weights(self, model):
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)


def compute_dice_iou(y_true, y_pred, num_classes=8):
    dice_scores = {}
    iou_scores = {}

    for c in range(num_classes):
        pred_c = y_pred == c
        true_c = y_true == c
        tp = np.logical_and(pred_c, true_c).sum()
        fp = np.logical_and(pred_c, ~true_c).sum()
        fn = np.logical_and(~pred_c, true_c).sum()

        dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)

        dice_scores[c] = dice
        iou_scores[c] = iou

    return dice_scores, iou_scores


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=50,
    num_classes=8,
    save_name="best_model.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_losses = []
    val_losses = []
    train_iou = []
    val_iou = []
    train_dice = []
    val_dice = []

    early_stopping = EarlyStopping(patience=5, min_delta=0.001, verbose=True)

    for epoch in range(num_epochs):

        # Training phase
        model.train()
        running_loss = 0.0
        train_all_preds = []
        train_all_labels = []

        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
        ):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)

            # Mask out ignore index 255
            preds = preds.cpu().numpy().ravel()
            labels = labels.cpu().numpy().ravel()
            mask = labels != 255

            train_all_preds.append(preds[mask])
            train_all_labels.append(labels[mask])

        y_pred = np.concatenate(train_all_preds)
        y_true = np.concatenate(train_all_labels)

        dice_per_class, iou_per_class = compute_dice_iou(
            y_true, y_pred, num_classes=num_classes
        )

        epoch_dice = np.mean(list(dice_per_class.values()))
        epoch_iou = np.mean(list(iou_per_class.values()))

        epoch_loss = running_loss / len(train_loader.dataset)

        train_losses.append(epoch_loss)
        train_dice.append(epoch_dice)
        train_iou.append(epoch_iou)

        # Validation phase
        model.eval()
        running_loss = 0.0
        val_all_preds = []
        val_all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"
            ):
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)

                # Mask out ignore index 255
                preds = preds.cpu().numpy().ravel()
                labels = labels.cpu().numpy().ravel()
                mask = labels != 255

                val_all_preds.append(preds[mask])
                val_all_labels.append(labels[mask])

        y_pred = np.concatenate(val_all_preds)
        y_true = np.concatenate(val_all_labels)

        dice_per_class, iou_per_class = compute_dice_iou(
            y_true, y_pred, num_classes=num_classes
        )

        epoch_val_dice = np.mean(list(dice_per_class.values()))
        epoch_val_iou = np.mean(list(iou_per_class.values()))

        epoch_val_loss = running_loss / len(val_loader.dataset)

        val_losses.append(epoch_val_loss)
        val_dice.append(epoch_val_dice)
        val_iou.append(epoch_val_iou)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Train Dice: {epoch_dice:.4f} || "
            f"Train IoU: {epoch_iou:.4f} || "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Dice: {epoch_val_dice:.4f} |"
            f"Val IoU: {epoch_val_iou:.4f}  "
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)  #
            else:
                scheduler.step()

        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            break

    early_stopping.restore_best_weights(model)
    torch.save(model.state_dict(), save_name)

    return (
        train_losses,
        val_losses,
        train_dice,
        val_dice,
        train_iou,
        val_iou,
    )


def plot_history(train_losses, val_losses, train_dice, val_dice, train_iou, val_iou):
    plt.figure(figsize=(18, 5))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    # Plot Dice Score
    plt.subplot(1, 3, 2)  # <-- changed from (1,2,3) to (1,3,2)
    plt.plot(train_dice, label="Training Dice")
    plt.plot(val_dice, label="Validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Dice Score")
    plt.legend()

    # Plot IoU Score
    plt.subplot(1, 3, 3)
    plt.plot(train_iou, label="Training IoU")
    plt.plot(val_iou, label="Validation IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU Score")
    plt.title("IoU Score")
    plt.legend()

    plt.tight_layout()
    plt.show()


def segmentation_report(model, loader, num_classes=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)  # [B, C, H, W]
            preds = torch.argmax(logits, dim=1)  # [B, H, W]
            all_preds.append(preds.cpu().numpy().ravel())
            all_trues.append(masks.cpu().numpy().ravel())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_trues)

    # 1) Standard classification report
    cls_report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=[f"Class {c}" for c in range(num_classes)],
        digits=4,
        zero_division=0,
    )

    dice_scores, iou_scores = compute_dice_iou(y_true, y_pred)
    print("=== Classification Report ===")
    print(cls_report)
    print("=== Dice per Class ===")
    for cls, score in dice_scores.items():
        print(f"{cls}: {score:.4f}")

    print("\n=== IoU per Class ===")
    for cls, score in iou_scores.items():
        print(f"{cls}: {score:.4f}")

    # after you fill dice_scores and iou_scores dicts:
    macro_dice = np.mean(list(dice_scores.values()))
    macro_iou = np.mean(list(iou_scores.values()))
    print("\n === Overall Performance === ")
    print(f"Macro-Dice: {macro_dice:.4f}")
    print(f"Macro-IoU : {macro_iou:.4f}")


def save_predictions_patches(model, loader, output_dir, device="cuda"):
    """
    Applies a trained model to a DataLoader and saves each predicted mask as a .npy file.

    Parameters:
        model: Trained segmentation model (e.g., U-Net).
        loader: PyTorch DataLoader yielding (image, label) or (image,) pairs.
        output_dir: Directory to save predicted mask patches.
        device: 'cpu' or 'cuda'.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        idx = 0
        for batch in loader:
            if len(batch) == 2:
                imgs, _ = batch
            else:
                imgs = batch[0]

            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            for pred in preds:
                pred_np = pred.cpu().numpy().astype(np.uint8)
                file_path = os.path.join(output_dir, f"pred_{idx:05d}.npy")
                np.save(file_path, pred_np)
                idx += 1
