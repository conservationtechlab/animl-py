'''
Test Module for verifying a newly trained classifier models

'''
import pandas as pd
import argparse
import yaml
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from animl import classifiers, generator


def test(data_loader, model, device='cpu'):
    '''
        Function for getting predicted labels for a test dataset

        Args
            - data_loader: Train Dataloader
            - model: loaded model object
            - device: specify to run inference on cpu or gpu

        Returns
            - file_paths: list of file paths of tested images
            - pred_labels: list of predicted labels
            - true_labels: list of ground truth
    '''
    model.to(device)
    model.eval()

    # create empty lists for true and predicted labels
    file_paths = []
    pred_labels = []
    true_labels = []

    with torch.no_grad():
        for _, batch in enumerate(data_loader):

            data = batch[0]
            labels = batch[1]
            paths = batch[2]
            file_paths.extend(paths)

            # get ground truth
            labels_np = labels.numpy()
            true_labels.extend(labels_np)

            # forward pass
            data = data.to(device)
            prediction = model(data)

            # add predicted labels to the predicted labels list
            pred_label = torch.argmax(prediction, dim=1)
            pred_label_np = pred_label.cpu().detach().numpy()
            pred_labels.extend(pred_label_np)

    return file_paths, pred_labels, true_labels


def main():
    '''
    Command line function

    Example usage :
    > python test.py --config configs/exp_resnet18.yaml
    '''
    parser = argparse.ArgumentParser(description='Test a deep learning model.')
    parser.add_argument('--config', help='Path to config file')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'rcfg, '))

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        device = 'cpu'

    # get model class list
    classes = pd.read_csv(cfg['class_file'])
    categories = dict([[x["species"], x["id"]] for _, x in classes.iterrows()])

    # initialize data loader
    test_dataset = pd.read_csv(cfg['test_set']).reset_index(drop=True)
    dl_test = generator.train_dataloader(test_dataset, categories, batch_size=cfg['batch_size'], workers=cfg['num_workers'])

    # initialize model
    model, _ = classifiers.load_model(cfg['active_model'], cfg['num_classes'], architecture=cfg['architecture'])

    paths, pred, true = test(dl_test, model, device)

    oa = np.mean((pred == true))
    print("Test accuracy:", oa)

    cm = confusion_matrix(true, pred)
    confuse = pd.DataFrame(cm, index=classes['species'], columns=classes['species'])

    confuse.to_csv(cfg['experiment_folder'] + "/confusion_matrx.csv")


if __name__ == '__main__':
    main()
