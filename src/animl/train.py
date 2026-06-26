'''
Classifier Training Script

Original script from
2022 Benjamin Kellenberger

Modified by Peter van Lunteren 2024
'''
import argparse
from pathlib import Path
from tqdm import trange

from animl import file_management

# mlops
try:
    import comet_ml
except ImportError:
    comet_ml = None

import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from sklearn.metrics import precision_score, recall_score
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR  # , ReduceLROnPlateau
from torch.amp import autocast, GradScaler

from animl.generator import train_dataloader
from animl.classification import load_classifier
from animl.utils.general import NUM_THREADS, init_seed


def save_classifier(model,
                    out_dir: str,
                    epoch: int,
                    stats: dict,
                    optimizer=None,
                    scheduler=None,
                    scaler=None):
    '''
    Saves model state weights.

    Args:
        model: pytorch model
        out_dir (str): directory to save model to
        epoch (int): current training epoch
        stats (dict): performance metrics of current epoch
        optimizer: pytorch optimizer (optional)
        scheduler: pytorch scheduler (optional)
        scaler: pytorch GradScaler (optional)

    Returns:
        None
    '''
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if model.__class__.__name__=="BioClip":
        # save only parameters that are changeable: lora and classifier
        trainable_state_dict = {
            k: v for k, v in model.state_dict().items()
            if "lora" in k or "classifier" in k
        }
        checkpoint = {
            'model': trainable_state_dict,
            'stats': stats
        }
    else:
        # get model parameters and add to stats
        checkpoint = {'model': model.state_dict(),
                    'stats': stats}
    # save optimizer, scheduler, and scaler state dicts if they are provided
    if optimizer is not None or scheduler is not None:
        checkpoint['epoch'] = epoch
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    if scaler is not None:
        checkpoint['scaler'] = scaler.state_dict()

    torch.save(checkpoint, open(f'{out_dir}/{epoch}.pt', 'wb'))


def load_classifier_checkpoint(model_path, model, optimizer, scheduler, scaler, device):
    '''
    Load checkpoint model weights to resume training.

    Args:
        model_path: path to saved weights
        model: loaded model object
        optimizer: optimizer object
        scheduler: learning rate scheduler
        scaler: GradScaler object or None if not using GradScaler
        device (str): device to load model and data to

    Returns:
        starting epoch (int)
    '''
    model_states = []
    for file in Path.iterdir(Path(model_path)):
        if Path(file).suffix.lower() == ".pt":
            model_states.append(file)

    if len(model_states):
        # at least one save state found; get latest
        savepoints = [m.stem for m in model_states]
        model_epochs = [int(sp) for sp in savepoints if sp.isdigit()]
        start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        checkpoint = torch.load(open(f'{model_path}/{start_epoch}.pt', 'rb'), map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
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

        if 'scaler' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])

        # get last epoch from model if available
        if 'epoch' in checkpoint:
            return checkpoint['epoch']
        else:
            return start_epoch
    else:
        # no save state found; start anew
        print('No model state found, starting new model')
        return 0


def _train_classifier_helper(data_loader, model, optimizer, scheduler, scaler=None, device='cpu',
                             mixed_precision=False, progress=True):
    '''
    Main training loop.

    Args:
        data_loader: dataloader object
        model: loaded model object
        optimizer: optimizer object
        scheduler: learning rate scheduler
        scaler: GradScaler object
        device (str): device to load model and data to
        mixed_precision (bool): flag to enable mixed precision for GPU
        progress (bool): flag to enable/disable progress bar

    Returns:
        loss_total: loss for epoch
        oa_total: overall accuracy for epoch
    '''
    model.to(device)
    model.train()  # put the model into training mode

    # loss function
    criterion = nn.CrossEntropyLoss()

    # log the loss and overall accuracy (OA)
    loss_total, oa_total = 0.0, 0.0

    if progress:
        progressBar = trange(len(data_loader))

    for idx, batch in enumerate(data_loader):
        collated, failed = batch
        if collated is None:  # entire batch was bad
            continue
        # put data and labels on device
        data = collated[0]
        labels = collated[1]
        data, labels = data.to(device), labels.to(device)
        # reset gradients to zero
        optimizer.zero_grad()

        # mixed precision training if GPU is available
        if mixed_precision and device != 'cpu' and torch.cuda.is_available() and scaler is not None:
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

        if progress:
            progressBar.set_description(
                '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)

    # end of epoch
    scheduler.step()
    if progress:
        progressBar.close()
    loss_total /= len(data_loader)
    oa_total /= len(data_loader)

    return loss_total, oa_total


def _validate_classifier_helper(data_loader, model, device="cpu", progress=True):
    '''
    Model validation function for each epoch.

    Note that this looks almost the same as the training
    function, except that we don't use any optimizer or gradient steps.

    Args:
        data_loader: dataloader object
        model: loaded model object
        device (str): device to load model and data to
        progress (bool): flag to enable/disable progress bar

    Returns:
        loss_total: loss for validation set
        oa_total: accuracy for validation set
        precision: precision for validation set
        recall: recall for validation set
    '''
    model.to(device)
    model.eval()  # put the model into evaluation mode

    criterion = nn.CrossEntropyLoss()

    # log the loss and overall accuracy (OA)
    loss_total, oa_total = 0.0, 0.0

    # create empty lists for true and predicted labels
    true_labels = []
    pred_labels = []

    if progress:
        progressBar = trange(len(data_loader))
    with torch.no_grad():  # gradients not necessary for validation
        for idx, batch in enumerate(data_loader):
            collated, failed = batch
            if collated is None:  # entire batch was bad
                continue
            data = collated[0]
            labels = collated[1]
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

            if progress:
                progressBar.set_description(
                    '[Val  ] Loss: {:.2f}; OA: {:.2f}%'.format(
                        loss_total/(idx+1),
                        100*oa_total/(idx+1)
                    )
                )
                progressBar.update(1)

    # end of epoch; finalize
    if progress:
        progressBar.close()
    loss_total /= len(data_loader)
    oa_total /= len(data_loader)

    # calculate precision and recall
    precision = precision_score(true_labels, pred_labels, average="weighted")
    recall = recall_score(true_labels, pred_labels, average="weighted")

    return loss_total, oa_total, precision, recall


def train_classifier(cfg):
    '''
    Command line function

    Args:
        cfg: path to config file

    Example usage:
    > python train.py --config configs/exp_resnet18.yaml
    '''
    # load cfg file
    cfg = file_management.load_yaml(cfg)

    if comet_ml:
        api_key = cfg.get('comet_api_key', None)
        if api_key:
            experiment = comet_ml.start(api_key=api_key,
                                        project_name=cfg.get('comet_project_name', None),
                                        workspace=cfg.get('comet_workspace', None))
            print("Comet ML experiment initialized.")
        else:
            experiment = None
            print("Comet ML not configured; skipping experiment logging.")
    else:
        experiment = None
        print("Comet ML not installed; skipping experiment logging.")

    progress = cfg.get('progress', True)
    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))
    crop = cfg.get('crop', True)
    file_col = cfg.get('file_col', 'filepath')
    label_col = cfg.get('label_col', 'species')
    resize_width, resize_height = cfg.get('image_size', [480,480])
    architecture=cfg['architecture']

    # check if GPU is available
    device = cfg.get('device', 'cpu')
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        device = 'cpu'
    # get mixed precision flag
    mixed_precision = cfg.get('mixed_precision', False)

    # model will be on CPU after this call if cfg['experiment_folder'] is a directory
    model, classes, current_epoch = load_classifier(cfg['experiment_folder'], cfg['class_file'],
                                                    device=device, architecture=cfg['architecture'])

    # Move model to the target device BEFORE optimizer initialization
    model.to(device)
    print(f"Model moved to {device}")

    categories = file_management.class_list_to_dict(classes, id_col=cfg.get('class_list_index', 'id'),
                                                    class_col=cfg.get('class_list_label', 'class'))

    # load datasets
    train_dataset = file_management.load_data(cfg['training_set'])
    validate_dataset = file_management.load_data(cfg['validate_set'])

    # Initialize data loaders for training and validation set
    dl_train = train_dataloader(train_dataset, categories,
                                batch_size=cfg['batch_size'],
                                num_workers=cfg.get('num_workers', NUM_THREADS),
                                file_col=file_col, label_col=label_col,
                                crop=crop, augment=cfg.get('augment', True),
                                resize_height=resize_height, resize_width=resize_width,
                                cache_dir=cfg.get('cache_folder', None))
    dl_val = train_dataloader(validate_dataset, categories,
                              batch_size=cfg.get('val_batch_size', 16),
                              num_workers=cfg.get('num_workers', NUM_THREADS),
                              file_col=file_col, label_col=label_col,
                              crop=crop, augment=False,
                              resize_height=resize_height, resize_width=resize_width,
                              cache_dir=cfg.get('cache_folder', None))

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

    if mixed_precision and device != 'cpu' and torch.cuda.is_available():
        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler('cuda', enabled=True)
    else:
        scaler = None

    # Load checkpoint for model weights, optimizer state, scheduler state, and actual current_epoch
    current_epoch = load_classifier_checkpoint(cfg['experiment_folder'], model, optim, scheduler, scaler, device=device)

    # initialize training arguments
    numEpochs = cfg['num_epochs']
    frozen_epochs = cfg.get('frozen_epochs', 1)
    print(f"Training for a total of {numEpochs} epochs, with {frozen_epochs} frozen epochs.")
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
            for name, param in model.named_parameters():
                if architecture != "bioclip_2" or "lora" in name:
                    #for bioclip, we only want to unfreeze the lora parameters
                    param.requires_grad = True

        loss_train, oa_train = _train_classifier_helper(dl_train, model, optim, scheduler, scaler=scaler, device=device,
                                                        mixed_precision=mixed_precision, progress=progress)
        loss_val, oa_val, precision, recall = _validate_classifier_helper(dl_val, model, device, progress=progress)

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

        if experiment:
            experiment.log_metrics(stats, step=current_epoch)

        if current_epoch % checkpoint == 0:
            save_classifier(model, cfg['experiment_folder'], current_epoch, stats, optim, scheduler, scaler)

        # best.pt saving
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            epochs_no_improve = 0
            save_classifier(model, cfg['experiment_folder'], 'best',  stats)
            print(f"Current best model saved at epoch {current_epoch} with ...")
            print(f"     val loss : {best_val_loss:.5f}")
            print(f"       val OA : {oa_val:.5f}")
            print(f"val precision : {precision:.5f}")
            print(f"   val recall : {recall:.5f}\n")
        else:
            epochs_no_improve += 1

        # last.pt saving
        save_classifier(model, cfg['experiment_folder'], 'last', stats)

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
    train_classifier(args.config)
