'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    Original script from
    2022 Benjamin Kellenberger
'''
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import argparse
import yaml
from tqdm import trange
import pandas as pd
import glob
import torch.nn as nn
import torch
import os
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score #fix

# let's import our own classes and functions!
from .utils import train_utils
from .generator import train_dataloader
from .classifiers import CTLClassifier, EfficientNet


# # log values using comet ml (comet.com)
experiment = Experiment(
  api_key="z3XHB9d67yOgZ2B5reqfuDLfZ",
  project_name="Cougar-Binary",
  workspace="tkswanson"
)


def save_model(cfg, epoch, model, stats):
    '''
        Saves model state weights.
    '''
    # make sure save directory exists; create if not
    exp_folder = cfg['experiment_folder']
    os.makedirs(exp_folder, exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

    # ...and save
    torch.save(stats, open(f'{exp_folder}/{epoch}.pt', 'wb'))

    # also save config file if not present
    cfpath = exp_folder + '/config.yaml'
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)


def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    if (cfg['architecture']=="CTL"):
        model_instance = CTLClassifier(cfg['num_classes'])
    elif (cfg['architecture']=="efficientnet_v2_m"):
        model_instance = EfficientNet(cfg['num_classes'])        
    else:
        raise AssertionError('Please provide the correct model')
    overwrite = cfg['overwrite']
    exp_folder = cfg['experiment_folder']

    # load latest model state
    model_states = glob.glob(exp_folder + '*.pt')

    if len(model_states) and overwrite==False:
        # at least one save state found; get latest
        model_epochs = [int(m.replace(exp_folder,'').replace('.pt','')) for m in model_states]
        start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(open(f'{exp_folder}/{start_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

    else:
        # no save state found; start anew
        print('Starting new model')
        start_epoch = 0

    return model_instance, start_epoch

def train(cfg, dataLoader, model, optimizer):
    '''
        Our actual training function.
    '''
    device = cfg['device']
    model.to(device)
    model.train() # put the model into training mode

    # loss function
    criterion = nn.CrossEntropyLoss()

    # log the loss and overall accuracy (OA)
    loss_total, oa_total = 0.0, 0.0                         

    progressBar = trange(len(dataLoader))
    for idx, (data, labels, _) in enumerate(dataLoader): 
        # put data and labels on device
        data, labels = data.to(device), labels.to(device)
        # forward pass
        prediction = model(data)
        # reset gradients to zero
        optimizer.zero_grad()

        loss = criterion(prediction, labels)
        loss.backward() # backward pass (calculate gradients of current batch)
        # apply gradients to model parameters
        optimizer.step()
        # log statistics
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
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total


def validate(cfg, dataLoader, model):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    device = cfg['device']
    model.to(device)
    model.eval() # put the model into evaluation mode
    
    criterion = nn.CrossEntropyLoss() # we still need a criterion to calculate the validation loss

    # log the loss and overall accuracy (OA)
    loss_total, oa_total = 0.0, 0.0

    # create empty lists for true and predicted labels
    true_labels = []
    pred_labels = []

    progressBar = trange(len(dataLoader))
    with torch.no_grad():               
        # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels, _) in enumerate(dataLoader):
            # put data and labels on device
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
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    # calculate precision and recall
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)

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
    train_utils.init_seed(cfg.get('seed', None))

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    # initialize data loaders for training and validation set
    train_dataset = pd.read_csv(cfg['training_set'])
    validate_dataset = pd.read_csv(cfg['validate_set'])

    classes = pd.read_csv(cfg['class_file'])
    categories = dict([[x["species"], x["id"]] for _,x in classes.iterrows()])
    
    dl_train = train_dataloader(train_dataset, categories, batch_size=cfg['batch_size'], workers=cfg['num_workers'])
    dl_val = train_dataloader(validate_dataset, categories, batch_size=cfg['batch_size'], workers=cfg['num_workers'])

    # initialize model
    model, current_epoch = load_model(cfg)

    # set up model optimizer
    optim = train_utils.setup_optimizer(cfg, model)

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
    main()
