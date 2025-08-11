import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import Counter
from tqdm import tqdm

def prepare_data(data_root_path, num_fold, reset_fold=False, debug=False):
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

    # Add debug mode column
    if debug == True: # and "debug" and not in df.columns:
        prct = 3
        print(f"⚙️  Debug mode activated: generating {prct}% debug subset with stratified sampling...")
        df["debug"] = 0

        debug_df = []
        for split_val in [0, 1]:  # 0 = val, 1 = train
            split_df = df[df["split"] == split_val]
            grouped = split_df.groupby("class")

            debug_subset = []
            for class_label, group in grouped:
                n = len(group)
                keep_n = max(1, int((prct/100) * n))  # keep at least one per class
                sampled = group.sample(n=keep_n, random_state=42)
                debug_subset.append(sampled)

            debug_subset_df = pd.concat(debug_subset)
            df.loc[debug_subset_df.index, "debug"] = 1
        
        df = df[df["debug"] == 1]

        print(f"✅ Debug subset created with {df['debug'].sum()} images.")

    return df

def get_class_counts(dataset, num_classes):
    labels = [label for _, label in dataset]
    counts = [0] * num_classes
    for label in labels:
        counts[label] += 1
    return counts