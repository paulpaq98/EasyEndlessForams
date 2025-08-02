from data.foram_dataloader import ForamDataset
from torch.utils.data import DataLoader
import torch.nn as nn

def prepare_trainings(df, num_fold, data_type, batch_size, transforms_dict, num_workers):

    training_dict = {}
    training_dict[f"Kfold_{num_fold:02d}"] = {}

    max_fold = 1 if num_fold == 0 else num_fold

    for fold_number in range(max_fold):

        training_dict[f"Kfold_{num_fold:02d}"][f"Kfold_{fold_number:02d}"] = {}
        
        # Create datasets and loaders
        train_dataset = ForamDataset(df, split="train", transform_img=transforms_dict["raw"]["train"], transform_mask=transforms_dict["masked"]["train"], data_type=data_type, fold_id=fold_number, num_fold=num_fold)
        val_dataset = ForamDataset(df, split="val", transform_img=transforms_dict["raw"]["val"], transform_mask=transforms_dict["masked"]["val"], data_type=data_type, fold_id=fold_number, num_fold=num_fold)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        # Get class names and number
        class_names = list(train_dataset.class_to_idx.keys())
        num_classes = len(class_names)

        criterion = nn.CrossEntropyLoss()

        # Dictionnaire
        training_dict[f"Kfold_{num_fold:02d}"][f"Kfold_{fold_number:02d}"]['train_loader'] = train_loader
        training_dict[f"Kfold_{num_fold:02d}"][f"Kfold_{fold_number:02d}"]['val_loader'] = val_loader
        training_dict[f"Kfold_{num_fold:02d}"][f"Kfold_{fold_number:02d}"]['class_names'] = class_names
        training_dict[f"Kfold_{num_fold:02d}"][f"Kfold_{fold_number:02d}"]['num_classes'] = num_classes
        training_dict[f"Kfold_{num_fold:02d}"][f"Kfold_{fold_number:02d}"]['criterion'] = criterion
        training_dict[f"Kfold_{num_fold:02d}"][f"Kfold_{fold_number:02d}"]['fold_id'] = fold_number
        training_dict[f"Kfold_{num_fold:02d}"][f"Kfold_{fold_number:02d}"]['num_fold'] = num_fold

    return training_dict