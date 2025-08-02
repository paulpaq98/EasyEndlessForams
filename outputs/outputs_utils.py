import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os


def plot_loss_curve(train_loss, val_loss, save_path='loss_curve.png'):
    """
    Affiche et sauvegarde la courbe de perte (loss) pour l'entraînement et la validation.

    Args:
        train_loss (list): Liste des pertes d'entraînement par époque.
        val_loss (list): Liste des pertes de validation par époque.
        save_path (str): Chemin de sauvegarde de l'image.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_curve(train_acc, val_acc, save_path='accuracy_curve.png'):
    """
    Affiche et sauvegarde la courbe de précision (accuracy) pour l'entraînement et la validation.

    Args:
        train_acc (list): Liste des précisions d'entraînement par époque.
        val_acc (list): Liste des précisions de validation par époque.
        save_path (str): Chemin de sauvegarde de l'image.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, class_names, save_path=None):
    """
    Plots and optionally saves a normalized confusion matrix.

    Args:
        y_true (list): True class labels.
        y_pred (list): Predicted class labels.
        labels (list): List of class indices (e.g., [0,1,2]).
        class_names (list): List of class names (str), same order as `labels`.
        save_path (str, optional): If provided, saves the figure to this path.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Normalize by true class
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaNs (e.g. divide by 0)

    # Plot
    plt.figure(figsize=(22, 18))
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix (Normalized by True Class)')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

