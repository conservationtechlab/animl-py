'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    Original script from
    2022 Benjamin Kellenberger
'''
import argparse
import yaml
from tqdm import trange
import pandas as pd
import random
import torch.nn as nn
import torch
from torch.backends import cudnn
from torch.optim import SGD
from sklearn.metrics import precision_score, recall_score

from .generator import train_dataloader
from .classifiers import save_model, load_model

# # log values using comet ml (comet.com)
# from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model
# experiment = Experiment()


def init_seed(seed):
    '''
        Initalizes the seed for all random number generators used. This is
        important to be able to reproduce results and experiment with different
        random setups of the same code and experiments.
    '''
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = True
        cudnn.deterministic = True


def train(data_loader, model, optimizer, device='cpu'):
    '''
        Our actual training function.
    '''
    model.to(device)
    model.train()  # put the model into training mode

    # loss function
    criterion = nn.CrossEntropyLoss()

    # log the loss and overall accuracy (OA)
    loss_total, oa_total = 0.0, 0.0

    progressBar = trange(len(data_loader))
    for idx, batch in enumerate(data_loader):
        # put data and labels on device
        data = batch[0]
        labels = batch[1]
        data, labels = data.to(device), labels.to(device)
        # forward pass
        prediction = model(data)
        # reset gradients to zero
        optimizer.zero_grad()

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
    progressBar.close()
    loss_total /= len(data_loader)
    oa_total /= len(data_loader)

    return loss_total, oa_total


def validate(data_loader, model, device="cpu"):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
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
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
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


def main():
    '''
    Command line function

    Example usage :
    > python train.py --config configs/exp_resnet18.yaml
    '''
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='exp_resnet18.yaml')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))
    crop = cfg.get('crop', False)

    # check if GPU is available
    device = cfg.get('device', 'cpu')
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        device = 'cpu'

    # initialize model and get class list
    model, classes, current_epoch = load_model(cfg['experiment_folder'], cfg['class_file'], device=device, architecture=cfg['architecture'])

    categories = dict([[x["species"], x["id"]] for _, x in classes.iterrows()])

    # initialize data loaders for training and validation set
    train_dataset = pd.read_csv(cfg['training_set']).reset_index(drop=True)
    validate_dataset = pd.read_csv(cfg['validate_set']).reset_index(drop=True)
    dl_train = train_dataloader(train_dataset, categories, batch_size=cfg['batch_size'], workers=cfg['num_workers'], crop=crop)
    dl_val = train_dataloader(validate_dataset, categories, batch_size=cfg['batch_size'], workers=cfg['num_workers'], crop=crop)

    # set up model optimizer
    optim = SGD(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

    numEpochs = cfg['num_epochs']
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train = train(dl_train, model, optim, device)
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

        checkpoint = cfg.get('checkpoint_frequency', 10)
        # experiment.log_metrics(stats, step=current_epoch)
        if current_epoch % checkpoint == 0:
            save_model(cfg['experiment_folder'], current_epoch, model, stats)


if __name__ == '__main__':
    main()
