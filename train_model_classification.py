#!/usr/bin/env python
# coding: utf-8

# # EfficientNet Multiclass Image Classifier (PyTorch)
# # ================================================

# # Imports
# # ================================================

from argparse import ArgumentParser

import torch
import torch.optim as optim

from data.data_utils import prepare_data

from models.classification.classification_utils import prepare_model, train_model
from outputs.outputs_utils import plot_accuracy_curve, plot_loss_curve, plot_f1_score_curve
from utils.utils_training import prepare_trainings, get_transforms


# # ================================================

def summarize_args(args):

    print("\n========== Configuration Summary ==========")

    print(f"debug mode          : {args.debug}")
    print("===========================================\n")

    print("\n# DATA PARAMETERS")
    print(f"Data root path      : {args.data_root_path}")
    print(f"Data type           : {args.data_type}")
    print(f"Number of folds     : {args.num_fold}")
    print(f"reset fold          : {args.reset_fold}")

    print("\n# MODEL PARAMETERS")
    print(f"Model name          : {args.name_model}")
    print(f"Pretrained path     : {args.path_pretrain}")

    print("\n# TRAIN PARAMETERS")
    print(f"Learning rate       : {args.learning_rate}")
    print(f"Number of epochs    : {args.num_epochs}")
    print(f"Batch size          : {args.batch_size}")
    print(f"Number of workers   : {args.num_workers}")
    print(f"early stopping      : {args.early_stopping}")
    print(f"patience            : {args.patience}")
    print(f"loss type           : {args.loss_type}")

    print("===========================================\n")

def clear_cache():
    torch.cuda.empty_cache()

def main(args):

    try:
        assert args.num_fold !=1
    except AssertionError:
        print("Can't create a one partition K-fold need at least 2 parts. Use num_fold = 0 for a regular training with  85% train / 15% validation repartition")
        return 

    # def fonction to print a clean resume of the args
    print("## Preparing Dataset ##")

    df_split = prepare_data(args.data_root_path, args.num_fold, args.reset_fold, args.debug)

    print("## Preparing Trainings ##")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device : ",device)

    trainings_dict = prepare_trainings(df_split, args.num_fold, args.data_type, args.batch_size, get_transforms(args.data_type), args.num_workers, args.loss_type)

    #print("training_dict :",trainings_dict)

    for training_dict in trainings_dict.keys():

        print(f"#### Launching training {training_dict} ####")
        Kfold_dict = trainings_dict[training_dict]

        for training_fold in Kfold_dict.keys():

            print(f"###### training fold {training_fold} ##")

            # Prepare model
            model = prepare_model(args.name_model,  Kfold_dict[training_fold]['num_classes'], device, pretrained_path=args.path_pretrain)

            # set optimizer
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            
            #getting dict
            train_dict = Kfold_dict[training_fold]

            ## TRAINNING train_loss_history, val_loss_history, train_acc_history, val_acc_history, train_f1_history, val_f1_history
            train_loss, val_loss, train_acc, val_acc, train_f1, val_f1 = train_model(
                model, 
                train_dict["train_loader"], 
                train_dict["val_loader"], 
                train_dict["criterion"],
                train_dict["loss_type"],  
                optimizer, 
                args.num_epochs, 
                early_stopping=args.early_stopping,
                patience=args.patience,
                data_type = args.data_type,
                model_name = args.name_model,
                num_fold = args.num_fold,
                fold_id = train_dict["fold_id"],
                device=device)
            
            root_output = "outputs"

            if args.num_fold !=0:

                save_path_loss_curve = root_output + "\\" + "classification\\loss_curve\\loss_curve"+str(train_dict["loss_type"])+"_fold_"+str(train_dict["fold_id"])+"_total_fold_"+str(args.num_fold)+"_"+args.data_type+"_"+args.name_model+".png"
                save_path_accuracy_curve = root_output + "\\" + "classification\\accuracy_curve\\accuracy_curve"+str(train_dict["loss_type"])+"_fold_"+str(train_dict["fold_id"])+"_total_fold_"+str(args.num_fold)+"_"+args.data_type+"_"+args.name_model+".png"
                save_path_f1_curve = root_output + "\\" + "classification\\f1_curve\\f1_curve"+str(train_dict["loss_type"])+"_fold_"+str(train_dict["fold_id"])+"_total_fold_"+str(args.num_fold)+"_"+args.data_type+"_"+args.name_model+".png"


            else:

                save_path_loss_curve = root_output + "\\" + "classification\\loss_curve\\loss_curve_"+str(train_dict["loss_type"])+"_"+args.data_type+"_"+args.name_model+".png"
                save_path_accuracy_curve = root_output + "\\" + "classification\\accuracy_curve\\accuracy_curve_"+str(train_dict["loss_type"])+"_"+args.data_type+"_"+args.name_model+".png"
                save_path_f1_curve = root_output + "\\" + "classification\\f1_curve\\f1_curve_"+str(train_dict["loss_type"])+"_"+args.data_type+"_"+args.name_model+".png"


            plot_loss_curve(train_loss, val_loss, save_path=save_path_loss_curve)
            plot_accuracy_curve(train_acc, val_acc, save_path=save_path_accuracy_curve)
            plot_f1_score_curve(train_f1, val_f1, save_path=save_path_f1_curve)

            clear_cache()


if __name__ == '__main__':
    parser = ArgumentParser()

    # DATA PARAMETERS #############################################

    parser.add_argument("--data_root_path", help="the type path to the root folder containing the data ", type=str, default=".\\data")

    parser.add_argument("--data_type", help="the type of data used for training : - with the background (raw), - with a mask (masked) ", type=str, default="raw")

    parser.add_argument("--num_fold", help="number of folds for a K-fold trainning", type=int, default=0)

    parser.add_argument("--reset_fold", help="force the reset on the stratified Kfold", type=bool, default=False)

    # MODEL PARAMETERS #############################################

    parser.add_argument("--name_model", help="name of the model being trained", type=str, default="effNet_small") # effNet_medium # effNet_large

    parser.add_argument("--path_pretrain", 
                        help="relative path to an already trained model for finetuning - use 'default' to load the pretrained torch default weights ", type=str, default=None)


    # TRAIN PARAMETERS #############################################

    parser.add_argument("--learning_rate", help="initial learning rate", type=float, default=1e-4)

    parser.add_argument("--num_epochs", help="maximum number of epochs", type=int, default=10)

    parser.add_argument("--batch_size", help="batch size for training", type=int, default=4)

    parser.add_argument("--num_workers", help="maximum number of workers for loading with the cpu", type=int, default=0)

    parser.add_argument("--early_stopping", help="activate the early stopping of the trainning phase", type=bool, default=False, choices=[True])

    parser.add_argument("--patience", help="number of epochs before the early stopping is activated", type=int, default=5)

    parser.add_argument("--loss_type", type=str, default="ce", choices=["ce", "cb", "focal", "cbfocal", "smoothing"],
                    help="Type of loss function to use: ce | cb | focal | cbfocal |smoothing")

    ########################################################################

    parser.add_argument("--debug", type=bool, default=False, choices=[True],
                help="Activate debug mode ")   
    
    args = parser.parse_args()

    summarize_args(args)

    main(args)


# python .\train_model_classification.py --name_model effNet_small --debug True --num_epoch 2 --loss_type cbfocal --num_workers 2 --early_stopping True --batch_size 8