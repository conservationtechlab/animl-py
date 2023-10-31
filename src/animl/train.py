'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    2022 Benjamin Kellenberger
'''

import os
import argparse
import yaml
import glob
from tqdm import trange
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score

# let's import our own classes and functions!
from util import init_seed, setup_optimizer
from dataset import create_dataloader
from model import load_model, save_model


# # log values using comet ml (comet.com)
# experiment = Experiment()

def train(cfg, dataLoader, model, optimizer):
    '''
        Our actual training function.
    '''
    # put model on device
    device = cfg['device']
    model.to(device)

    # put the model into training mode
    # this is required for some layers that behave differently during training
    # and validation (examples: Batch Normalization, Dropout, etc.)
    model.train()

    # loss function
    criterion = nn.CrossEntropyLoss()

    # running averages
    loss_total, oa_total = 0.0, 0.0                         # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    for idx, (data, labels, paths) in enumerate(dataLoader): 
        # see the last line of file "dataset.py" where we return the image tensor (data) and label
        # put data and labels on device
        data, labels = data.to(device), labels.to(device)
        # import pdb; pdb.set_trace() # DEBUGGER

        # forward pass
        prediction = model(data)

        # reset gradients to zero
        optimizer.zero_grad()

        # loss
        loss = criterion(prediction, labels)

        # backward pass (calculate gradients of current batch)
        loss.backward()

        # apply gradients to model parameters
        optimizer.step()

        # log statistics
        loss_total += loss.item()                       
        # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor

        pred_label = torch.argmax(prediction, dim=1)    
        # the predicted label is the one at position (class index) with highest predicted value
        
        oa = torch.mean((pred_label == labels).float()) 
        # OA: number of correct predictions divided by batch size (i.e., average/mean)
        oa_total += oa.item()

        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)

    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)           # shorthand notation for: loss_total = loss_total / len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total


def validate(cfg, dataLoader, model):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    
    criterion = nn.CrossEntropyLoss()   # we still need a criterion to calculate the validation loss

    # running averages
    loss_total, oa_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)    

    # create empty lists for true and predicted labels
    true_labels = []
    pred_labels = []

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    
    with torch.no_grad():               
        # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels) in enumerate(dataLoader):

            # put data and labels on device
            data, labels = data.to(device), labels.to(device)

            # add true labels to the true labels list
            # import pdb; pdb.set_trace() # DEBUGGER
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
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    # calculate precision and recall
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)

    return loss_total, oa_total, precision, recall


def main():
    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='exp_resnet18.yaml')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    # initialize data loaders for training and validation set
    train_dataset = pd.read_csv(cfg['training_set'])
    train_dataset = pd.read_csv(cfg['test_set'])
    dl_train = create_dataloader(train_dataset)
    dl_val = create_dataloader(test_dataset)

    # initialize model
    model, current_epoch = load_model(cfg)

    # set up model optimizer
    optim = setup_optimizer(cfg, model)

    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    numEpochs = cfg['num_epochs']
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train = train(cfg, dl_train, model, optim)
        loss_val, oa_val, precision, recall = validate(cfg, dl_val, model)
        
        # combine stats and save
        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val,
            'precision' : precision,
            'recall' : recall
        }
        
        experiment.log_metrics(stats, step=current_epoch)

        if current_epoch % 10 == 0:
            save_model(cfg, current_epoch, model, stats)
    

if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()