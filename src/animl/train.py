'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    Original script from
    2022 Benjamin Kellenberger

    Modiefied by Peter van Lunteren 2024
'''
import argparse
import yaml
import os
from tqdm import trange
import pandas as pd
import random
import torch.nn as nn
import torch
from torch.backends import cudnn
from torch.optim import SGD, AdamW
from sklearn.metrics import precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, CosineAnnealingLR
from torch.amp import autocast, GradScaler

from animl.generator import train_dataloader
from animl.classify import save_classifier, load_classifier
from animl.utils.general import NUM_THREADS

# # log values using comet ml (comet.com)
# from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model
# experiment = Experiment()


def init_seed(seed):
    '''
        Initalizes the seed for all random number generators used. This is
        important to be able to reproduce results and experiment with different
        random setups of the same code and experiments.

        Args:
            - seed (int): seed for RNG
    '''
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = True
        cudnn.deterministic = True


def train_func(data_loader, model, optimizer,  scheduler, device='cpu'):
    '''
        Training Function

        Args:
            - data_loader: dataloader object
            - model: loaded model object
            - optimizer: optimizer object
            - device (str): device to load model and data to

        Returns:
            - loss_total: loss for epoch
            - oa_total: overall accuracy for epoch
    '''
    model.to(device)
    model.train()  # put the model into training mode

    # loss function
    criterion = nn.CrossEntropyLoss()

    # log the loss and overall accuracy (OA)
    loss_total, oa_total = 0.0, 0.0

    progressBar = trange(len(data_loader))

    if cfg.get('mixed_precision', False) and device != 'cpu' and torch.cuda.is_available():
        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler('cuda', enabled=True)

    for idx, batch in enumerate(data_loader):
        # put data and labels on device
        data = batch[0]
        labels = batch[1]
        data, labels = data.to(device), labels.to(device)
        # reset gradients to zero
        optimizer.zero_grad()

        # mixed precision training if GPU is available
        if cfg.get('mixed_precision', False) and device != 'cpu' and torch.cuda.is_available():
            # Scales the loss, and calls backward() on the scaled loss to create
            # backward gradients. This is a more efficient way to calculate gradients.
            with autocast(device_type='cuda', dtype=torch.float16):
                prediction = model(data)
                loss = criterion(prediction, labels)
            scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            # Calls the step function on the optimizer
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()
        else:
            # forward pass
            prediction = model(data)
            # loss
            loss = criterion(prediction, labels)
            # calculate gradients of current batch
            loss.backward()
            # apply gradients to model parameters
            optimizer.step()

        loss_total += loss.item()

        pred_label = torch.argmax(prediction, dim=1)

        oa = torch.mean((pred_label == labels).float())
        oa_total += oa.item()

        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)

    # end of epoch
    scheduler.step()
    progressBar.close()
    loss_total /= len(data_loader)
    oa_total /= len(data_loader)

    return loss_total, oa_total


def validate(data_loader, model, device="cpu"):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.

    Args:
        - data_loader: dataloader object
        - model: loaded model object
        - device (str): device to load model and data to

    Returns:
        - loss_total: loss for validation set
        - oa_total: accuracy for validation set
        - precision: precision for validation set
        - recall: recall for validation set
    '''
    model.to(device)
    model.eval()  # put the model into evaluation mode

    criterion = nn.CrossEntropyLoss()

    # log the loss and overall accuracy (OA)
    loss_total, oa_total = 0.0, 0.0

    # create empty lists for true and predicted labels
    true_labels = []
    pred_labels = []

    progressBar = trange(len(data_loader))
    with torch.no_grad():  # gradients not necessary for validation
        for idx, batch in enumerate(data_loader):
            data = batch[0]
            labels = batch[1]
            data, labels = data.to(device), labels.to(device)

            # add true labels to the true labels list
            labels_np = labels.cpu().detach().numpy()
            true_labels.extend(labels_np)

            # forward pass
            prediction = model(data)

            # loss
            loss = criterion(prediction, labels)

            # log statistics
            loss_total += loss.item()

            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()

            # add predicted labels to the predicted labels list
            pred_label_np = pred_label.cpu().detach().numpy()
            pred_labels.extend(pred_label_np)

            progressBar.set_description(
                '[Val  ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)

    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(data_loader)
    oa_total /= len(data_loader)

    # calculate precision and recall
    precision = precision_score(true_labels, pred_labels, average="weighted")
    recall = recall_score(true_labels, pred_labels, average="weighted")

    return loss_total, oa_total, precision, recall


def load_checkpoint(model_path, model, optimizer, scheduler, device):
    model_states = []
    for file in os.listdir(model_path):
        if os.path.splitext(file)[1] == ".pt":
            model_states.append(file)

    if len(model_states):
        # at least one save state found; get latest
        model_epochs = [int(m.replace(model_path, '').replace('.pt', '')) for m in model_states]
        start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        checkpoint = torch.load(open(f'{model_path}/{start_epoch}.pt', 'rb'), map_location=device)
        model.load_state_dict(checkpoint['model'])
        # Model is assumed to be on the correct device already (moved in main before optimizer creation)

        # load optimzier state if available
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            # Ensure optimizer's state tensors are on the correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.device != device:
                        state[k] = v.to(device)

        # load scheduler state if available
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

        # get last epoch from model if avialble
        if 'epoch' in checkpoint:
            return checkpoint['epoch']
        else:
            return start_epoch
    else:
        # no save state found; stasrt anew
        print('No model state found, starting new model')
        return 0


def main(cfg):
    '''
    Command line function

    Example usage:
    > python train.py --config configs/exp_resnet18.yaml
    '''
    # load cfg file
    cfg = yaml.safe_load(open(cfg, 'r'))
    
    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))
    crop = cfg.get('crop', True)
    file_col = cfg.get('file_col', 'FilePath')
    label_col = cfg.get('label_col', 'species')

    # check if GPU is available
    device = cfg.get('device', 'cpu')
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        device = 'cpu'

    # initialize model and get class list
    classes = pd.read_csv(cfg['class_file'])
    # model will be on CPU after this call if cfg['experiment_folder'] is a directory
    model, current_epoch = load_classifier(cfg['experiment_folder'], len(classes), device=device, architecture=cfg['architecture'])

    # Move model to the target device BEFORE optimizer initialization
    model.to(device)
    print(f"Model moved to {device}")

    categories = dict([[x[cfg.get('class_list_label', 'class')], x[cfg.get('class_list_index', 'id')]] for _, x in classes.iterrows()])

    # load datasets
    train_dataset = pd.read_csv(cfg['training_set']).reset_index(drop=True)
    validate_dataset = pd.read_csv(cfg['validate_set']).reset_index(drop=True)

    # Initialize data loaders for training and validation set
    dl_train = train_dataloader(train_dataset, categories, batch_size=cfg['batch_size'], num_workers=cfg.get('num_workers', NUM_THREADS),
                                file_col=file_col, label_col=label_col, crop=crop, augment=cfg.get('augment', True),
                                cache_dir=cfg.get('cache_folder', None))
    dl_val = train_dataloader(validate_dataset, categories, batch_size=cfg.get('val_batch_size', 16), num_workers=cfg.get('num_workers', NUM_THREADS),
                              file_col=file_col, label_col=label_col, crop=crop, augment=False, cache_dir=cfg.get('cache_folder', None))

    # set up model optimizer
    if cfg.get("optimizer", "AdamW") == 'AdamW':
        optim = AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'], amsgrad=False)
    else:
        optim = SGD(model.parameters(), lr=cfg['learning_rate'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])

    # initialize scheduler
    if cfg.get("scheduler", True):
        # scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=cfg['patience'])
        scheduler = CosineAnnealingLR(optim, T_max=cfg.get('t_max', 100), eta_min=0)
    else:  # do nothing scheduler
        scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1)

    # Load checkpoint for model weights, optimizer state, scheduler state, and actual current_epoch
    current_epoch = load_checkpoint(cfg['experiment_folder'], model, optim, scheduler, device)

    # initialize training arguments
    numEpochs = cfg['num_epochs']
    frozen_epochs = cfg.get('frozen_epochs', 1)
    if 'patience' in cfg:
        patience = cfg['patience']
        early_stopping = True
        print(f"Early stopping enabled with a patience of {patience} epochs")
    else:
        early_stopping = False

    best_val_loss = float('inf')
    epochs_no_improve = 0

    log_file = cfg.get('log_file', None)
    if log_file is not None and current_epoch == 0:
        with open(log_file, 'a') as f:
            f.write("Epoch,LearningRate,Train_Loss,Train_Accuracy,Val_Loss,Val_Accuracy,Precision,Recall\n")

    # training loop
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')
        print(f"Using learning rate : {scheduler.get_last_lr()[0]}")

        if current_epoch > frozen_epochs:
            for param in model.parameters():
                param.requires_grad = True

        loss_train, oa_train = train_func(dl_train, model, optim, device)
        loss_val, oa_val, precision, recall = validate(dl_val, model, device)

        # combine stats and save
        stats = {
            'num_classes': len(classes),
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val,
            'precision': precision,
            'recall': recall
        }

        # Log epoch stats to file
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"{current_epoch},{scheduler.get_last_lr()[0]:.5f},{loss_train:.4f},{oa_train:.4f},"
                        f"{loss_val:.4f},{oa_val:.4f},"
                        f"{precision:.4f},{recall:.4f}\n")

        # <current_epoch>.pt checkpoint saving every *checkpoint_frequency* epochs
        checkpoint = cfg.get('checkpoint_frequency', 10)
        # experiment.log_metrics(stats, step=current_epoch)
        if current_epoch % checkpoint == 0:
            save_classifier(cfg['experiment_folder'], current_epoch, model, stats, optim, scheduler)

        # best.pt saving
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            epochs_no_improve = 0
            save_classifier(cfg['experiment_folder'], 'best', model, stats)
            print(f"Current best model saved at epoch {current_epoch} with ...")
            print(f"     val loss : {best_val_loss:.5f}")
            print(f"       val OA : {oa_val:.5f}")
            print(f"val precision : {precision:.5f}")
            print(f"   val recall : {recall:.5f}\n")
        else:
            epochs_no_improve += 1

        # last.pt saving
        save_classifier(cfg['experiment_folder'], 'last', model, stats)

        # if user specified early stopping
        if early_stopping:
            # check patience
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='exp_resnet18.yaml')
    args = parser.parse_args()

    print(f'Using config "{args.config}"')
    main(args.config)
