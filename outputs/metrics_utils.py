import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm


from pytorch_fid.fid_score import calculate_fid_given_paths


def apply_tta(model, inputs, tta_transforms=None):
    """
    Applies Test-Time Augmentation (TTA) to the inputs and returns averaged predictions.

    Args:
        model (torch.nn.Module): The model to evaluate.
        inputs (torch): Input batch tensor (B, C, H, W).
        tta_transforms (list of callable): List of transforms to apply to inputs. Each must return a tensor of same shape.

    Returns:
        torch: Averaged output predictions over TTA variants.
    """
    if tta_transforms is None:
        tta_transforms = [lambda x: x]  # Identity

    outputs = []
    for transform in tta_transforms:
        augmented_inputs = transform(inputs)
        output = model(augmented_inputs)
        outputs.append(output)

    # Average over TTA outputs
    return torch.mean(torch.stack(outputs), dim=0)


def evaluate_model(model, dataloader, device):
    """
    Evaluate a classification model on a dataloader.

    Returns:
        accuracy (float): overall accuracy
        y_true (list): true labels
        y_pred (list): predicted labels
        labels (list): list of unique class labels
    """
    model.eval()
    y_true = []
    y_pred = []


    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            ## TTA
            tta_transforms = [
                lambda x: x,
                lambda x: TF.hflip(x),
                lambda x: TF.vflip(x),
                lambda x: TF.rotate(x, 15),
                lambda x: TF.rotate(x, -15),
            ]

            outputs = apply_tta(model, inputs, tta_transforms)

            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"\nEvaluation Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    labels = sorted(set(y_true))
    return accuracy, y_true, y_pred, labels


def compute_fid(path_real, path_fake, batch_size=32, device=None, dims=2048, verbose=True):
    """
    Compute the Frechet Inception Distance (FID) between two image folders.

    Args:
        path_real (str): Path to the folder with real images.
        path_fake (str): Path to the folder with generated images.
        batch_size (int): Batch size for the Inception model.
        device (str): "cuda" or "cpu". If None, auto-detect.
        dims (int): Dimensionality of Inception features to use.
        verbose (bool): Whether to print the result.

    Returns:
        float: FID score
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert os.path.exists(path_real), f"Path does not exist: {path_real}"
    assert os.path.exists(path_fake), f"Path does not exist: {path_fake}"

    fid_value = calculate_fid_given_paths(
        [path_real, path_fake],
        batch_size=batch_size,
        device=device,
        dims=dims
    )

    if verbose:
        print(f"\nFID Score (real vs. generated): {fid_value:.4f}")

    return fid_value