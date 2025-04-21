import torch
from tqdm import tqdm
from torchmetrics.segmentation import DiceScore
import torch.nn.functional as F


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


def pixel_accuracy(preds, targets):
    correct = (preds == targets).float()
    return correct.sum() / correct.numel()


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=50,
    num_classes=8,
    save_name="best_model.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_dice = []
    val_dice = []

    early_stopping = EarlyStopping(patience=5, min_delta=0.001, verbose=True)
    # Initialize metrics
    dice_score = DiceScore(
        num_classes=num_classes, average="weighted", input_format="index"
    ).to(device)

    for epoch in range(num_epochs):

        # Training phase
        model.train()
        running_loss = 0.0
        total_acc = 0.0
        dice_score.reset()
        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
        ):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            dice_score.update(preds, labels)
            acc = pixel_accuracy(preds, labels)
            total_acc += acc.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_dice = dice_score.compute().item()
        epoch_train_acc = total_acc / len(train_loader.dataset)

        train_losses.append(epoch_train_loss)
        train_dice.append(epoch_train_dice)
        train_accuracies.append(epoch_train_acc)
        dice_score.reset()

        # Validation phase
        model.eval()
        running_loss = 0.0
        total_acc = 0.0
        dice_score.reset()

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
                dice_score.update(preds, labels)
                acc = pixel_accuracy(preds, labels)
                total_acc += acc.item() * inputs.size(0)

        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_dice = dice_score.compute().item()
        epoch_val_acc = total_acc / len(val_loader.dataset)

        val_losses.append(epoch_val_loss)
        val_dice.append(epoch_val_dice)
        val_accuracies.append(epoch_val_acc)
        dice_score.reset()

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.4f} | "
            f"Train Dice: {epoch_train_dice:.4f} || "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.4f} | "
            f"Val Dice: {epoch_val_dice:.4f}"
        )

        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            break

    early_stopping.restore_best_weights(model)
    torch.save(model.state_dict(), save_name)

    return (
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        train_dice,
        val_dice,
    )


def evaluate_model(model, test_loader, criterion, num_classes=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    running_loss = 0.0
    total_acc = 0.0

    # Initialize Dice metric
    dice_score = DiceScore(
        num_classes=num_classes, average="weighted", input_format="index"
    ).to(device)

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Predictions and metrics
            preds = torch.argmax(outputs, dim=1)
            acc = pixel_accuracy(preds, labels)

            # Accumulate
            dice_score.update(preds, labels)
            running_loss += loss.item() * inputs.size(0)
            total_acc += acc.item() * inputs.size(0)

    avg_loss = running_loss / len(test_loader.dataset)
    avg_dice = dice_score.compute().item()
    avg_acc = total_acc / len(test_loader.dataset)
    dice_score.reset()

    print(f"\n Evaluation Results:")
    print(
        f"Loss: {avg_loss:.4f} | Pixel Accuracy: {avg_acc * 100:.2f}% | Dice Score: {avg_dice:.4f}"
    )

    return avg_loss, avg_acc, avg_dice


import matplotlib.pyplot as plt


def plot_history(train_losses, val_losses, train_accs, val_accs, train_dice, val_dice):
    """
    Plot training & validation loss, accuracy, and Dice score over epochs.

    Args:
        train_losses (list): Training loss per epoch
        val_losses (list): Validation loss per epoch
        train_accs (list): Training pixel accuracy per epoch
        val_accs (list): Validation pixel accuracy per epoch
        train_dice (list): Training Dice score per epoch
        val_dice (list): Validation Dice score per epoch
    """
    plt.figure(figsize=(18, 5))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label="Training Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Pixel Accuracy")
    plt.legend()

    # Plot Dice Score
    plt.subplot(1, 3, 3)
    plt.plot(train_dice, label="Training Dice")
    plt.plot(val_dice, label="Validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Dice Score")
    plt.legend()

    plt.tight_layout()
    plt.show()
