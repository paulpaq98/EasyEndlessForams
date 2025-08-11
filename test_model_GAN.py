#!/usr/bin/env python
# coding: utf-8

# # GAN (PyTorch)
# # ================================================

# # Imports
# # ================================================

from argparse import ArgumentParser

import torch
import torch.optim as optim

from data.data_utils import prepare_data
from models.generation.generation_utils import train_gan_model, Generator, Discriminator
from utils.utils_training import prepare_trainings, get_transforms
from outputs.metrics_utils import compute_fid


# # ================================================

def summarize_args(args):
    print("\n========== GAN Training Configuration ==========")
    print(f"Data root path      : {args.data_root_path}")
    print(f"Data type           : {args.data_type}")

    print(f"Epochs              : {args.num_epochs}")
    print(f"Batch size          : {args.batch_size}")
    print(f"Learning rate       : {args.learning_rate}")
    print(f"Latent dim          : {args.latent_dim}")
    print(f"Early stopping      : {args.early_stopping}")
    print(f"Patience            : {args.patience}")
    print("===============================================\n")

def clear_cache():
    torch.cuda.empty_cache()

def main(args):

    print("## Preparing Dataset ##")
    df_split = prepare_data(args.data_root_path, num_fold=0, reset_fold=False, debug=False)

    print("## Preparing Trainings ##")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    trainings_dict = prepare_trainings(
        df = df_split,
        num_fold = 0,
        data_type = args.data_type,
        batch_size = args.batch_size, 
        transforms_dict = get_transforms(args.data_type),
        num_workers = args.num_workers, 
        loss_type="gan"
    )

    for training_name, Kfold_dict in trainings_dict.items():
        print(f"#### Launching GAN training ####")
        for fold_name, train_dict in Kfold_dict.items():
            
            train_loader = train_dict["train_loader"]
            val_loader = train_dict["val_loader"]

            loss_fn = train_dict["criterion"]

            G = Generator(z_dim=args.latent_dim).to(device)
            D = Discriminator().to(device)

            optimizer_G = optim.Adam(G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
            optimizer_D = optim.Adam(D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    # Exemple : en fin d'entraînement GAN
    class_name = "Candeina_nitida" # a changer
    real_dir = f"data/img/{class_name}"
    fake_dir = "outputs/generation/generated_images_epoch50"

    fid = compute_fid(real_dir, fake_dir)


if __name__ == '__main__':
    parser = ArgumentParser()

    # DATA PARAMETERS
    parser.add_argument("--data_root_path", type=str, default="./data")
    parser.add_argument("--data_type", type=str, default="raw")

    # GAN PARAMETERS
    parser.add_argument("--latent_dim", type=int, default=100)

    # TRAIN PARAMETERS
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--early_stopping", type=bool, default=False)
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()

    summarize_args(args)

    main(args)


