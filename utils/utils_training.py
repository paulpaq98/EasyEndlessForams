from data.foram_dataloader import ForamDataset
from models.classification.classification_utils import  FocalLoss, LabelSmoothingLoss, ClassBalancedLoss, ClassBalancedFocalLoss
from data.data_utils import get_class_counts

from torch.utils.data import DataLoader
from torchvision import  transforms
import torch

def prepare_trainings(df, num_fold, data_type, batch_size, transforms_dict, num_workers, loss_type):

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

        train_class_counts = get_class_counts(train_dataset, num_classes)

        if loss_type == 'focal':
            criterion = FocalLoss(gamma=2.0)
        elif loss_type == 'smoothing':
            criterion = LabelSmoothingLoss(smoothing=0.1)
        elif loss_type == 'cb':
            criterion = ClassBalancedLoss(samples_per_cls=train_class_counts, beta=0.999, num_classes=num_classes)
        elif loss_type == 'cbfocal':
            criterion = ClassBalancedFocalLoss(samples_per_cls=train_class_counts, beta=0.999, gamma=2.0, num_classes=num_classes)
        elif loss_type == 'gan':
            criterion = torch.nn.BCELoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()

        # Dictionnaire
        training_dict[f"Kfold_{num_fold:02d}"][f"Kfold_{fold_number:02d}"]['train_loader'] = train_loader
        training_dict[f"Kfold_{num_fold:02d}"][f"Kfold_{fold_number:02d}"]['val_loader'] = val_loader
        training_dict[f"Kfold_{num_fold:02d}"][f"Kfold_{fold_number:02d}"]['class_names'] = class_names
        training_dict[f"Kfold_{num_fold:02d}"][f"Kfold_{fold_number:02d}"]['num_classes'] = num_classes
        training_dict[f"Kfold_{num_fold:02d}"][f"Kfold_{fold_number:02d}"]['criterion'] = criterion
        training_dict[f"Kfold_{num_fold:02d}"][f"Kfold_{fold_number:02d}"]['fold_id'] = fold_number
        training_dict[f"Kfold_{num_fold:02d}"][f"Kfold_{fold_number:02d}"]['num_fold'] = num_fold
        training_dict[f"Kfold_{num_fold:02d}"][f"Kfold_{fold_number:02d}"]['loss_type'] = loss_type

    return training_dict


# # ================================================
# # Transforms
# # ================================================

# get transforms for trainings and evaluation (raw and masked)

def get_transforms(data_type):

    transforms_dict = {}
    train_img_transforms, val_img_transforms, train_mask_transforms, val_mask_transforms = None, None, None, None

    if data_type == "raw":

        train_img_transforms = transforms.Compose([
            transforms.Resize(384),
            transforms.RandomResizedCrop(300),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])


        val_img_transforms = transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])


    elif data_type == "masked":

        train_mask_transforms = transforms.Compose([
            transforms.Resize(384),
            transforms.RandomResizedCrop(300),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor()
        ])

        val_mask_transforms = transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(300),
            transforms.ToTensor()
        ])

    transforms_dict["raw"] = {}
    transforms_dict["masked"] = {}

    transforms_dict["raw"]["train"] = train_img_transforms
    transforms_dict["raw"]["val"] = val_img_transforms
    transforms_dict["masked"]["train"] = train_mask_transforms
    transforms_dict["masked"]["val"] = val_mask_transforms

    return transforms_dict