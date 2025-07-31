
# # Imports
# # ================================================
import os

import torch
from torchvision import  transforms, models
from tqdm import tqdm
import torch.nn as nn

# # Functions
# Loading base pre-trained models

def loading_corresponding_model(model_name):

    if model_name == "effNet_small":

        # Charger EfficientNetV2 - small
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

    elif model_name == "effNet_medium":

        # Charger EfficientNetV2 - medium
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)

    elif model_name == "effNet_large":

        # Charger EfficientNetV2 - large
        model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)

    return model

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

# prepare model

def prepare_model(model_name, num_classes, device):

    # Load pre-trained EfficientNetV2
    model = loading_corresponding_model(model_name)

    # Replace the classifier to match the number of classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # Send the template to the correct device
    model = model.to(device)

    return model


# Train module

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10,
                early_stopping=True, 
                patience=5, min_delta=1e-4, load_weights_path=None,
                data_type="raw", model_name="effNet_small", fold_id = 0, num_fold=0, device="cpu"):

    # Loading weights if provided
    if load_weights_path is not None and os.path.exists(load_weights_path):
        model.load_state_dict(torch.load(load_weights_path, map_location=device))
        print(f"🔄 Loaded model weights from {load_weights_path}")
    elif load_weights_path:
        print(f"⚠️ Warning: '{load_weights_path}' not found. Starting from scratch.")

    model = model.to(device)  # Send model to the correct device (GPU or CPU)

    # Initialize metrics
    best_acc = 0.0
    early_stop_counter = 0

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    # Define the scheduler (Cosine Annealing by default here)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Main training loop
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')

        ### --- TRAINING PHASE ---
        model.train()  # Training mode
        running_loss, running_corrects = 0.0, 0

        for data in tqdm(train_loader):

            if len(data)==3:
                inputs, masks, labels = data[0], data[1], data[2]

                # Ensure mask has shape (B, 1, H, W)
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)  # Add channel dimension if missing

                # Normalize mask to be binary (optional, if not already 0 or 1)
                masks = (masks > 0.5).float()

                # Apply the mask to each image: keep only the masked region
                inputs = inputs * masks
                    
            elif len(data)==2:
                inputs, labels = data[0], data[1]

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()           # Reset gradients
            outputs = model(inputs)         # Forward pass / prediction
            _, preds = torch.max(outputs, 1)  # Predicted class
            loss = criterion(outputs, labels)

            loss.backward()  # Backpropagation
            optimizer.step() # Weight update

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

        # Average metrics over the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.item())

        print(f'Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

        ### --- VALIDATION PHASE ---
        model.eval()  # Evaluation mode
        val_running_loss, val_corrects = 0.0, 0

        with torch.no_grad():  # No gradient for validation

            for data in tqdm(val_loader):

                if len(data)==3:
                    inputs, masks, labels = data[0], data[1], data[2]

                    # Ensure mask has shape (B, 1, H, W)
                    if masks.dim() == 3:
                        masks = masks.unsqueeze(1)  # Add channel dimension if missing

                    # Normalize mask to be binary (optional, if not already 0 or 1)
                    masks = (masks > 0.5).float()

                    # Apply the mask to each image: keep only the masked region
                    inputs = inputs * masks

                elif len(data)==2:
                    inputs, labels = data[0], data[1]

                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels)

        # Average metrics over validation
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc.item())

        print(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}')

        # --- Scheduler (cosine annealing) ---
        scheduler.step()

        # --- Save the best model ---
        if val_acc > best_acc + min_delta:
            best_acc = val_acc
            best_loss = val_loss
            if num_fold !=0:
                save_path = "models\\classification\\best_fold_"+str(fold_id)+"_total_fold_"+str(num_fold)+"_"+data_type+"_"+model_name+".pth"
            else :
                save_path = "models\\classification\\best_"+data_type+"_"+model_name+".pth"
            torch.save(model.state_dict(), save_path)
            print("✔ Best model saved")
            early_stop_counter = 0  # Reset patience
        else:
            early_stop_counter += 1
            if early_stopping:
                print(f"⏳ Early stopping patience: {early_stop_counter}/{patience}")
                if early_stop_counter >= patience:
                    print("⛔ Early stopping triggered")
                    break  # Stop training

    return train_loss_history, val_loss_history, train_acc_history, val_acc_history
