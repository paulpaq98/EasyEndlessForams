#!/usr/bin/env python
# coding: utf-8

# ================================================
# EfficientNet Multiclass Image Classifier - TEST
# ================================================

import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from data.data_utils import prepare_data
from models.classification.classification_utils import get_transforms, prepare_model
from utils.utils_training import prepare_trainings
from outputs.metrics_utils import evaluate_model
from outputs.outputs_utils import plot_confusion_matrix


def summarize_args(args):
    print("\n========== Configuration Summary ==========")

    print("\n# DATA PARAMETERS")
    print(f"Data root path      : {args.data_root_path}")

    print("\n# MODEL PARAMETERS")
    print(f"loading path     : {args.path_loading}")

    print("\n# TEST PARAMETERS")
    print(f"Batch size          : {args.batch_size}")
    print(f"Number of workers   : {args.num_workers}")

    print("===========================================\n")

def summarize_model_path(path_loading):
    """
    Extracts and prints model properties from the model filename.

    Args:
        path_loading (str): Path to the model file (e.g., "outputs/best_fold_0_total_fold_5_raw_effNet_small.pth")
    """
    if path_loading is None:
        raise ValueError("You must provide a model path using --path_loading")

    filename = os.path.basename(path_loading).split(".")[0]  # Remove .pth
    parts = filename.split("_")

    if "fold" in filename and "total" in filename:
        # Format: best_fold_XX_total_fold_XX_DATA_NAME.pth
        try:
            fold_id = int(parts[2])
            num_fold = int(parts[5])
            data_type = parts[-3]
            name_model = "_".join(parts[-2:])
        except (IndexError, ValueError):
            raise ValueError("Failed to parse K-Fold model filename: check format.")
    else:
        # Format: best_DATA_NAME.pth
        try:
            fold_id = 0
            num_fold = 0
            data_type = parts[-3]
            name_model = "_".join(parts[-2:])
        except (IndexError, ValueError):
            raise ValueError("Failed to parse standard model filename: check format.")

    print("\n========== Model Parameters Summary ==========")
    print(f"Model path      : {path_loading}")
    print(f"Model name      : {name_model}")
    print(f"Data type       : {data_type}")
    print(f"Fold ID         : {fold_id}")
    print(f"Total folds     : {num_fold}")
    print("===================================\n")

    return name_model, data_type, fold_id, num_fold

def clear_cache():
    torch.cuda.empty_cache()

def main(args):

    name_model, data_type, fold_id, num_fold = summarize_model_path(args.path_loading)

    try:
        assert num_fold != 1
    except AssertionError:
        print("Can't create a one-partition K-fold. Use num_fold = 0 for a regular split.")
        return

    print("## Preparing Dataset ##")
    df_split = prepare_data(args.data_root_path, num_fold)

    print("## Preparing Test Loaders ##")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device)

    evaluation_dict = prepare_trainings(df_split, num_fold, data_type, args.batch_size, get_transforms(data_type), args.num_workers)

    print("evaluation_dict :",evaluation_dict)

    for training_dict in evaluation_dict.keys():
        print(f"#### Testing configuration {training_dict} ####")
        Kfold_dict = evaluation_dict[training_dict]

        for evaluation_fold in Kfold_dict.keys():
            fold_data = Kfold_dict[evaluation_fold]
            temp_fold_id = fold_data["fold_id"]

            if temp_fold_id == num_fold-1 or num_fold == 0 : # Evaluation only on the untrained part of the K-fold # Might be updated to allow testing on the trained part for model exploration

                print(f"###### Evaluating on fold {evaluation_fold} ######")

                num_classes = fold_data['num_classes']
                class_names = fold_data['class_names']
                val_loader = fold_data['val_loader']

                model = prepare_model(name_model, num_classes, device)
                model.load_state_dict(torch.load(args.path_loading, map_location=device))
                model.eval()

                print("## Evaluating model on validation set ##")
                accuracy, y_true, y_pred, labels = evaluate_model(model, val_loader, device)

                print(f"Validation Accuracy: {accuracy:.4f}")

                # Save confusion matrix
                root_output = "outputs"
                if num_fold != 0:
                    cm_path = os.path.join(root_output, "classification", "confusion_matrix", f"cm_fold_{temp_fold_id}_total_fold_{num_fold}_{data_type}_{name_model}.png")
                else:
                    cm_path = os.path.join(root_output, "classification", "confusion_matrix", f"cm_{data_type}_{name_model}.png")

                plot_confusion_matrix(y_true, y_pred, labels=labels, class_names=class_names, save_path=cm_path)

                clear_cache()

if __name__ == '__main__':
    parser = ArgumentParser()

    # DATA PARAMETERS #############################################
    parser.add_argument("--data_root_path", type=str, default=".\\data", help="Path to the root folder containing the path to the data")

    # MODEL PARAMETERS ############################################
    parser.add_argument("--path_loading", type=str, required=True, help="Path to load model weights (.pt or .pth)")

    # TEST PARAMETERS #############################################
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading")

    args = parser.parse_args()

    summarize_args(args)
    main(args)
