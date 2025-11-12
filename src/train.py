import os
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from .config import (
    CSV_PATH,
    MODEL_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LR,
    WEIGHT_DECAY,
    NUM_WORKERS,
    RANDOM_SEED,
    DEVICE,
)
from .data import load_data
from .model import SignalNet


class SignalsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # (10000,)
        x = torch.from_numpy(x).unsqueeze(0)  # (1, 10000)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

            running_loss += loss.item() * X.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate_with_details(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

            running_loss += loss.item() * X.size(0)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            all_targets.append(y.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    return epoch_loss, epoch_acc, all_targets, all_preds


def plot_history(history, out_prefix="signal_classifier"):
    epochs = history["epoch"]

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["test_loss"], label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["test_acc"], label="test_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_accuracy.png")
    plt.close()


def plot_confusion_matrix(cm, class_names, normalize=False,
                          title="Confusion matrix", out_path="confusion_matrix.png"):
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main_train():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device(
        "cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
    )
    print(f"Использую устройство: {device}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_DIR, "best_model.pth")

    print("Загружаю данные...")
    X_train, y_train, X_test, y_test = load_data(CSV_PATH)

    print(f"Train: {X_train.shape[0]} сигналов")
    print(f"Test:  {X_test.shape[0]} сигналов")

    train_dataset = SignalsDataset(X_train, y_train)
    test_dataset = SignalsDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = SignalNet(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_test_acc = 0.0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )

    print(f"Лучшая тестовая точность за обучение: {best_test_acc:.4f}")
    print(f"Весы лучшей модели сохранены в '{best_model_path}'")

    hist_df = pd.DataFrame(history)
    hist_df.to_csv("metrics_history.csv", index=False)
    print("История метрик сохранена в metrics_history.csv")

    plot_history(history, out_prefix="signal_classifier")
    print("Графики сохранены в signal_classifier_loss.png и signal_classifier_accuracy.png")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Загружены веса лучшей модели для финальной оценки.")

    test_loss, test_acc, y_true, y_pred = evaluate_with_details(
        model, test_loader, criterion, device
    )

    print(f"\nФинальная оценка на тесте (по лучшей модели): loss={test_loss:.4f}, acc={test_acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    class_names = ["sinusoid (1)", "sawtooth (2)", "square (3)"]

    print("\nConfusion matrix (сырые значения):")
    print(cm)

    plot_confusion_matrix(cm, class_names, normalize=False,
                          title="Confusion matrix",
                          out_path="confusion_matrix_raw.png")
    plot_confusion_matrix(cm, class_names, normalize=True,
                          title="Normalized confusion matrix",
                          out_path="confusion_matrix_norm.png")
    print("Матрицы ошибок сохранены в confusion_matrix_raw.png и confusion_matrix_norm.png")

    print("\nКлассификационный отчёт:")
    print(classification_report(y_true, y_pred, target_names=class_names))