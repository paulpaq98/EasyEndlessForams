import matplotlib.pyplot as plt


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