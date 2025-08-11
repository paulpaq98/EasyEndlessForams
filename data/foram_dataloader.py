import torch
from PIL import Image
from torch.utils.data import Dataset

class ForamDataset(Dataset):
    def __init__(self, df, split="train", transform_img=None, transform_mask=None, data_type = "raw", fold_id=0, num_fold=0):
        self.data = df

        # Build class-to-index map
        classes = sorted(self.data['class'].unique())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        kfold_column = f"Kfold_{num_fold:02d}" if num_fold !=0 else "split"

        if split == "train":
            self.data = self.data[self.data[kfold_column] != fold_id]
        elif split == "val":
            self.data = self.data[self.data[kfold_column] == fold_id]
            
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.data_type = data_type

        # Paths and labels
        self.img_paths = self.data['image_path'].tolist()
        try:
            self.mask_paths = self.data['mask_path'].tolist()
        except:
            self.mask_paths = None
        self.labels = [self.class_to_idx[label] for label in self.data['class']]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image and mask
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.data_type == "masked":
            mask = Image.open(self.mask_paths[idx]).convert("L")  # Assuming masks are grayscale

        # Apply transforms
        if self.transform_img and self.transform_mask:
            seed = torch.seed()  # Ensure same transformation for both
            torch.manual_seed(seed)
            img = self.transform_img(img)
            if self.data_type == "masked":  # Assuming masks are grayscale
                torch.manual_seed(seed)
                mask = self.transform_mask(mask)
        else:
            img = self.transform_img(img) if self.transform_img else img
            if self.data_type == "masked":
                mask = self.transform_mask(mask) if self.transform_mask else mask

        label = self.labels[idx]

        if self.data_type == "raw":
            return img, label  

        elif self.data_type == "masked":
            return img, mask, label  