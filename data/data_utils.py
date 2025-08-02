import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

def prepare_data(data_root_path, num_fold, reset_fold=False):
    dataset_csv_path = os.path.join(data_root_path, "dataset.csv")

    # Create the DataFrame if it does not already exist
    if os.path.exists(dataset_csv_path):
        df = pd.read_csv(dataset_csv_path, sep=";")
    else:
        image_dir = os.path.join(data_root_path, "img")
        image_paths = []
        classes = []

        # Retrieves all images and their class
        for class_name in os.listdir(image_dir):
            class_dir = os.path.join(image_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        relative_path = os.path.join("img", class_name, filename)
                        image_paths.append(os.path.join(data_root_path, relative_path))
                        classes.append(class_name)

        df = pd.DataFrame({"image_path": image_paths, "class": classes})

        # Stratification for split train/val: 1 = train, 0 = val
        df["split"] = -1
        stratifier = StratifiedKFold(n_splits=int(1 / 0.15), shuffle=True, random_state=42)
        train_idx, val_idx = next(stratifier.split(df["image_path"], df["class"]))
        df.loc[train_idx, "split"] = 1
        df.loc[val_idx, "split"] = 0

        # Backup
        df.to_csv(dataset_csv_path, sep=";", index=False)


    # Add stratified Kfold columns if num_fold > 0
    if num_fold > 1:
        kfold_column = f"Kfold_{num_fold:02d}"
        if kfold_column not in df.columns or reset_fold:
            df[kfold_column] = -1
            strat_kfold = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=42)
            for fold_index, (_, val_indices) in enumerate(strat_kfold.split(df["image_path"], df["class"])):
                df.loc[val_indices, kfold_column] = fold_index # from 0 to num_fold-1

            df.to_csv(dataset_csv_path, sep=";", index=False)

    return df
