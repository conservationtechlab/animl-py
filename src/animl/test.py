'''
Test script for evaluating trained model on holdout test set.

Original script from
2022 Benjamin Kellenberger
'''
import argparse
from tqdm import trange
import pandas as pd
import torch
import numpy as np
from typing import Union
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from torch.utils.data import DataLoader

from animl import file_management
from animl.generator import train_dataloader
from animl.classification import load_classifier
from animl.utils.general import NUM_THREADS


def _test_classifer_helper(data_loader: DataLoader,
                           model: torch.nn.Module,
                           device: Union[str, torch.device] = 'cpu') -> float:
    '''
    Run trained model on test split

    Args:
        data_loader: test set dataloader
        model: trained model object
        device: run model on gpu or cpu, defaults to cpu
    '''
    model.eval()  # put the model into evaluation mode

    pred_labels = []
    true_labels = []
    filepaths = []

    progressBar = trange(len(data_loader))
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            collated, failed = batch
            if collated is None:  # entire batch was bad
                continue
            # forward pass
            data = collated[0]
            labels = collated[1]
            paths = collated[2]
            data = data.to(device)
            prediction = model(data)
            # add predicted labels to the predicted labels list
            pred_label = torch.argmax(prediction, dim=1)
            pred_label_np = pred_label.cpu().detach().numpy()
            pred_labels.extend(pred_label_np)
            # get ground truth labels
            labels_np = labels.numpy()
            true_labels.extend(labels_np)
            # get file paths
            filepaths.extend(paths)

            progressBar.update(1)

    return pred_labels, true_labels, filepaths


def test_classifier(cfg):
    '''
    Command line function

    Args:
        cfg: path to config file

    Example usage:
    > python test.py --config configs/exp_resnet18.yaml
    '''
    # load cfg file
    cfg = file_management.load_yaml(cfg)

    crop = cfg.get('crop', False)

    # check if GPU is available
    device = cfg.get('device', 'cpu')
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        device = 'cpu'

    # initialize model and get class list
    model, classes = load_classifier(cfg['active_model'],
                                     cfg['class_file'],
                                     device=device,
                                     architecture=cfg['architecture'])

    class_list_label = cfg.get('class_list_label', 'class')
    class_list_index = cfg.get('class_list_index', 'id')

    categories = file_management.class_list_to_dict(classes, id_col=class_list_index, class_col=class_list_label)

    # initialize data loaders for training and validation set
    test_dataset = file_management.load_data(cfg['test_set'])
    dl_test = train_dataloader(test_dataset, categories,
                               batch_size=cfg['batch_size'],
                               num_workers=cfg.get('num_workers', NUM_THREADS),
                               file_col=cfg.get('file_col', 'filepath'),
                               label_col=cfg.get('label_col', 'species'),
                               crop=crop, augment=False,
                               cache_dir=cfg.get('cache_folder', None))
    # get predictions
    pred, true, paths = _test_classifer_helper(dl_test, model, device)
    # calculate precision and recall
    prec = precision_score(true, pred, average='weighted')
    recall = recall_score(true, pred, average='weighted')

    pred = np.asarray(pred)
    true = np.asarray(true)

    oa = np.mean((pred == true))
    print(f"Test accuracy: {oa}")

    results = pd.DataFrame({'FilePath': paths,
                            'Ground Truth': true,
                            'Predicted': pred,
                            'Accuracy': oa,
                            'Precision': prec,
                            'Recall': recall})
    file_management.save_data(results, cfg['experiment_folder'] + "/test_results.csv")

    cm = confusion_matrix(true, pred)
    confuse = pd.DataFrame(cm, columns=classes[class_list_label], index=classes[class_list_label])
    file_management.save_data(confuse, cfg['experiment_folder'] + "/confusion_matrix.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test species classifier model.')
    parser.add_argument('--config', help='Path to config file')
    args = parser.parse_args()

    print(f'Using config "{args.config}"')
    test_classifier(args.config)
