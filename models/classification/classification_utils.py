
# # Imports
# # ================================================
import os

import torch
from torchvision import models
from tqdm import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score


# # Functions

# # ================================================
# # Models
# # ================================================
# Loading base pre-trained models

def loading_corresponding_model(model_name, num_classes, pretrained_path=None):

    ## GETTING BASE MODELS

    # # ================================================
    # # EfficientNet
    # # ================================================

    if pretrained_path!=None:

        print("pretrained_path : ",pretrained_path)

    #  EfficientNetV2 - small
    if model_name == "effNet_small":

        #  EfficientNetV2 - small
        if pretrained_path == "default":
            print("loading default weights")
            model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        else:
            model = models.efficientnet_v2_s(weights=None)

    #  EfficientNetV2 - medium
    if model_name == "effNet_medium":

        #  EfficientNetV2 - small
        if pretrained_path == "default":
            model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        else:
            model = models.efficientnet_v2_m(weights=None)

    #  EfficientNetV2 - large
    if model_name == "effNet_large":

        #  EfficientNetV2 - small
        if pretrained_path == "default":
            model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
        else:
            model = models.efficientnet_v2_l(weights=None)

    ## ADAPTATING MODELS FOR THE PROBLEM AT HAND

    if "effNet" in model_name:
        # Replace the classifier to match the number of classes
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # # ================================================
    # # SwinTransformer
    # # ================================================

    #  SwinTransformer - tiny
    if model_name == "swin_tiny":

        #  EfficientNetV2 - tiny
        if pretrained_path == "default":
            print("loading default weights")
            model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)

            """
            for param in model.parameters():
            param.requires_grad = False

            for param in model.head.parameters():
                param.requires_grad = True            
            """

        else:
            model = models.swin_t(weights=None)

        in_features = model.head.in_features
        model.head = nn.Sequential(
                    nn.Dropout(p=0.3),
                    nn.Linear(in_features, num_classes)
                )
        
    # # ================================================
    # # ConvNeXt
    # # ================================================

    #  ConvNeXt - tiny
    if model_name == "ConvNeXt_tiny":

        #  ConvNeXt - tiny
        if pretrained_path == "default":
            print("loading default weights")
            model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

            """
            for param in models():
                param.requires_grad = False          
            """

        else:
            model = models.convnext_tiny(weights=None)

        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)         


    #  ConvNeXt - base
    if model_name == "convNeXt_base":

        #  ConvNeXt - tiny
        if pretrained_path == "default":
            print("loading default weights")
            model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)

            """
            for param in models():
                param.requires_grad = False          
            """

        else:
            model = models.convnext_base(weights=None)

        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)     


    return model

# prepare model

def prepare_model(model_name, num_classes, device, pretrained_path):

    # Load pre-trained EfficientNetV2
    model = loading_corresponding_model(model_name, num_classes, pretrained_path)

    # Load pretrained custom model
    if pretrained_path is not None and os.path.exists(pretrained_path):
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
            print(f"🔄 Loaded model weights from {pretrained_path}")
        except:
            print(f"Could not load weights")

    # Send the template to the correct device
    model = model.to(device)

    return model

# # ================================================
# # Loss
# # ================================================

# Focal Loss 
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()
    
# Class Balenced SoftMax Cross Entropy

class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Softmax Cross-Entropy Loss.
    Source: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019.
    """

    def __init__(self, samples_per_cls, beta, num_classes):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.samples_per_cls = samples_per_cls
        self.num_classes = num_classes

        # Compute class weights from effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        self.weights = (1.0 - beta) / np.array(effective_num)
        self.weights = self.weights / np.sum(self.weights) * num_classes  # Normalize

        self.weights = torch.tensor(self.weights, dtype=torch.float32)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch): (B, num_classes)
            targets (torch): (B,)
        """
        if logits.device != self.weights.device:
            self.weights = self.weights.to(logits.device)

        weights_per_sample = self.weights[targets]  # (B,)
        loss = F.cross_entropy(logits, targets, weight=None, reduction='none')  # (B,)
        loss = weights_per_sample * loss
        return loss.mean()

# Class Balenced SoftMax Cross Entropy + Focal Loss, based on https://arxiv.org/abs/1901.05555

class ClassBalancedFocalLoss(nn.Module):
    """
    Combines Class-Balanced Loss and Focal Loss.
    Based on: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019.
    """

    def __init__(self, samples_per_cls, beta, gamma, num_classes):
        super(ClassBalancedFocalLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes

        # Compute effective number
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes  # Normalize
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, num_classes)
            targets: (B,)
        """
        if logits.device != self.weights.device:
            self.weights = self.weights.to(logits.device)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Softmax + log
        pred = F.softmax(logits, dim=1)
        pred_log = torch.log(pred + 1e-12)

        # Focal factor
        focal_factor = torch.pow(1 - pred, self.gamma)

        # Weight by class-balanced weights
        weights = self.weights.unsqueeze(0)  # shape (1, C)
        cb_weights = weights * focal_factor  # shape (B, C)

        loss = -targets_one_hot * cb_weights * pred_log
        return loss.sum(dim=1).mean()
    
# Label Smoothing
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        n_classes = inputs.size(1)
        log_preds = F.log_softmax(inputs, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_preds, dim=1))

# # ================================================
# # Trainning
# # ================================================

# Warmup + CosineAnnealing
def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    return LambdaLR(optimizer, lr_lambda)


# Train module

def train_model(model, 
                train_loader, val_loader, 
                criterion,
                loss_type,
                optimizer,
                num_epochs=10,
                early_stopping=True, 
                patience=5, 
                min_delta=1e-4, 
                data_type="raw", 
                model_name="effNet_small",
                fold_id = 0,
                num_fold=0,
                device="cpu"):

    model = model.to(device)  # Send model to the correct device (GPU or CPU)

    # Initialize metrics
    best_acc = 0.0
    best_f1 = 0.0
    early_stop_counter = 0

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []
    train_f1_history, val_f1_history = [], []

    # Define the scheduler (Cosine Annealing by default here)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    #scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=3, total_epochs=num_epochs)
    
    # Main training loop
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')

        ### --- TRAINING PHASE ---
        model.train()  # Training mode
        running_loss, running_corrects = 0.0, 0

        # Accumulate all predictions and true labels
        all_train_preds = []
        all_train_labels = []

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

            all_train_preds.append(preds.cpu())
            all_train_labels.append(labels.cpu())

        # Average metrics over the epoch
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc.item())

        # Flatten predictions and labels
        all_train_preds = torch.cat(all_train_preds)
        all_train_labels = torch.cat(all_train_labels)

        # Compute macro-F1
        train_f1 = f1_score(all_train_labels.numpy(), all_train_preds.numpy(), average='macro')
        train_f1_history.append(train_f1)

        print(f'Train Loss: {loss:.4f} | Acc: {train_acc:.4f} | Macro-F1: {train_f1:.4f}')

        ### --- VALIDATION PHASE ---
        model.eval()  # Evaluation mode
        val_running_loss, val_corrects = 0.0, 0

        # Accumulate all predictions and true labels
        all_val_preds = []
        all_val_labels = []

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
                
                all_val_preds.append(preds.cpu())
                all_val_labels.append(labels.cpu())

        # Average metrics over validation
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc.item())

        # Flatten predictions and labels
        all_val_preds = torch.cat(all_val_preds)
        all_val_labels = torch.cat(all_val_labels)

        # Compute macro-F1
        val_f1 = f1_score(all_val_labels.numpy(), all_val_preds.numpy(), average='macro')
        val_f1_history.append(val_f1)

        print(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Macro-F1: {val_f1:.4f}')

        # Scheduler
        scheduler.step()

        # Save the best model based on macro-F1
        if val_f1 > best_f1 + min_delta:
            best_acc = val_acc
            best_f1 = val_f1
            best_loss = val_loss
            if num_fold != 0:
                save_path = f"models\\classification\\{loss_type}_fold_{fold_id}_total_fold_{num_fold}_{data_type}_{model_name}.pth"
            else:
                save_path = f"models\\classification\\{loss_type}_{data_type}_{model_name}.pth"
            torch.save(model.state_dict(), save_path)
            print("✔ Best model saved (based on Macro-F1)")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stopping:
                print(f"⏳ Early stopping patience: {early_stop_counter}/{patience}")
                if early_stop_counter >= patience:
                    print("⛔ Early stopping triggered")
                    break
        

    return train_loss_history, val_loss_history, train_acc_history, val_acc_history, train_f1_history, val_f1_history


