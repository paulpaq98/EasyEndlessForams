import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import torchvision.utils as vutils
import numpy as np

# # ================================================
# # Classification outputs
# # ================================================

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

def plot_f1_score_curve(train_f1, val_f1, save_path='f1_score_curve.png'):
    """
    Affiche et sauvegarde la courbe du F1-score pour l'entraînement et la validation.

    Args:
        train_f1 (list): Liste des F1-scores d'entraînement par époque.
        val_f1 (list): Liste des F1-scores de validation par époque.
        save_path (str): Chemin de sauvegarde de l'image.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_f1, label='Train F1-score')
    plt.plot(val_f1, label='Validation F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score (macro)')
    plt.title('F1-score over Epochs')
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

# # ================================================
# # GAN outputs
# # ================================================

def save_image_grid(images, save_path=None, nrow=8, normalize=True, title=None):
    """
    Save or display a grid of images (typically GAN outputs).

    Args:
        images (Tensor): Tensor of shape (B, C, H, W).
        save_path (str): Path to save the image (e.g., 'output/epoch_001.png').
        nrow (int): Number of images per row.
        normalize (bool): Normalize images to [0, 1] before display.
        title (str): Optional title for the plot.

    Returns:
        None
    """
    # Ensure directory exists
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create grid
    grid = vutils.make_grid(images, nrow=nrow, normalize=normalize, pad_value=1)

    # Convert to numpy for display
    np_grid = grid.cpu().numpy().transpose((1, 2, 0))

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.imshow(np_grid)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()