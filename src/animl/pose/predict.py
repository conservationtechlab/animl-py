#get sequence of images from each capture, minimum one image and max 6
#divide into even (enough) parts
#run the model on the first 3 but before taking the max class, we sum up the probabilities from both left and right classes across the three images
#take the max across the 3
#if both sides have the same predicted viewpoint, use the one with the higher likelihood and set the other to be the other one
import argparse
import yaml
from tqdm import trange
import pandas as pd
import torch
import numpy as np
from typing import Union
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from animl.generator import manifest_dataloader
from animl.classification import load_model


def predict_viewpoints(dataset: pd.DataFrame,
                       cfg,
                       crop: bool,
                       model: torch.nn.Module, 
                       device: Union[str, torch.device] = 'cpu'):
    '''
        Run trained model on test split

        Args:
            - data_loader: test set dataloader
            - model: trained model object
            - device: run model on gpu or cpu, defaults to cpu
    '''
    model.to(device)
    model.eval()  # put the model into training mode

    pred_labels = []
    filepaths = []

    progressBar = trange(len(dataset['capture_group'].value_counts()))
    with torch.no_grad():
        for group in dataset: #sequence group
            df = group[1]
            half1 = df.loc[df.groupby(['Station', 'StationID']).ngroup() == 0].reset_index()
            half2 = df.loc[df.groupby(['Station', 'StationID']).ngroup() == 1].reset_index()
            if len(half2) > 0: # if there are 2 cameras
                group1 = manifest_dataloader(half1, batch_size=len(half1), workers=cfg['num_workers'], 
                                file_col=cfg.get('file_col', 'FilePath'), crop=crop)
                for batch in enumerate(group1):
                    g1batch = batch[1]
                    data = g1batch[0]
                    data = data.to(device)
                    prediction1 = model(data) # list of predictions
                    g1_sums = torch.sum(prediction1, dim=0) # sum predictions for each viewpoint
                    g1_pred = torch.argmax(g1_sums).item() # return column with the max sum for viewpoint prediction
                    
                    # get filepath
                    g1_paths = g1batch[1]

                group2 = manifest_dataloader(half2, batch_size=len(half2), workers=cfg['num_workers'], 
                                file_col=cfg.get('file_col', 'FilePath'), crop=crop)
                for batch in enumerate(group2):
                    g2batch = batch[1]
                    data = g2batch[0]
                    data = data.to(device)
                    prediction2 = model(data) # list of predictions
                    g2_sums = torch.sum(prediction2, dim=0) # sum predictions for each viewpoint
                    g2_pred = torch.argmax(g2_sums).item() # return column with the max sum for viewpoint prediction

                    # get ground truth label
                    g2_paths = g2batch[1]

                if g1_pred == g2_pred: # if viewpoint predictions are the same, whichever group has the higher summed probability will get that viewpoint
                    if g1_sums[g1_pred] > g2_sums[g2_pred]:
                        g2_pred = 0 if g1_pred == 1 else 1 # change g2 to be the opposite viewpoint of g1 if g1 sum is greater
                    elif g2_sums[g2_pred] > g1_sums[g1_pred]:
                        g1_pred = 0 if g2_pred == 1 else 1

                g1 = [g1_pred] * len(group1) # make list of the prediction labels to append to pred_labels, group prediction applies to whole group
                g2 = [g2_pred] * len(group2)
                pred_labels.extend(g1)
                pred_labels.extend(g2)                

                #get file paths for each group
                filepaths.extend(g1_paths)
                filepaths.extend(g2_paths)
            else: # entire group is from the same station/stationID
                dl_group = manifest_dataloader(pd.DataFrame(df), batch_size=len(df), workers=cfg['num_workers'], 
                                file_col=cfg.get('file_col', 'FilePath'), crop=crop)

                for batch in enumerate(dl_group):
                    g1batch = batch[1]
                    data = g1batch[0]
                    data = data.to(device)
                    prediction = model(data) # list of predictions

                    sums = torch.sum(prediction, dim=0) # sum predictions for each viewpoint
                    pred = torch.argmax(sums) # return column with the max sum for viewpoint prediction
                    pred_labels.append(pred.item())
                    
                    paths = batch[1]
                    filepaths.extend(paths)

            progressBar.update(1)
        
    return pred_labels, filepaths


def main():
    '''
    Command line function

    Example usage:
    > python predict.py --config configs/exp_resnet18.yaml
    '''
    parser = argparse.ArgumentParser(description='Test viewpoint prediction model.')
    parser.add_argument('--config', help='Path to config file')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
    crop = cfg.get('crop', False)

    # check if GPU is available
    device = cfg.get('device', 'cpu')
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        device = 'cpu'

    # initialize model and get class list
    classes = pd.read_csv(cfg['class_file'])
    model = load_model(cfg['active_model'], len(classes), device=device, architecture=cfg['architecture'])

    class_list_label = cfg.get('class_list_label', 'class')
    #class_list_index = cfg.get('class_list_index', 'id')

    #categories = dict([[x[class_list_label], x[class_list_index]] for _, x in classes.iterrows()])

    # initialize dataset with datapoints grouped by CaptureID
    dataset = pd.read_csv(cfg['test_set'])
    dataset['capture_group'] = dataset['CaptureID'].diff().gt(6).cumsum().add(1)
    grouped = dataset.groupby('capture_group')
    truth = dataset[['FilePath', 'viewpoint']].copy() #get ground truth viewpoint for each datapoint
    # get predictions
    pred, paths = predict_viewpoints(grouped, cfg, crop, model, device)
    print(len(paths))
    print(len(pred))
    predictions = pd.DataFrame({'FilePath': paths,
                                'prediction': pred})

    merged = truth.merge(predictions, left_on='FilePath', right_on='FilePath')

    print(merged)

    #pred = np.asarray(pred)
    #true = np.asarray([truth['viewpoint']]) # get ground truth viewpoints of each file
    #for path in paths:
    #    file = truth.loc[truth['FilePath'] == path]
    #    true.append(file['viewpoint'])
    #paths = np.asarray(paths)
    #true = np.asarray(true)


    oa = (merged['prediction'] == merged['viewpoint']).mean()
    print(f"Test accuracy: {oa}")

    #results = pd.DataFrame({'FilePath': paths,
    #                        'Ground Truth': true,
    #                        'Predicted': pred})
    merged.to_csv(cfg['experiment_folder'] + "/grouptest_results.csv")

    cm = confusion_matrix(merged['viewpoint'], merged['prediction'])
    confuse = pd.DataFrame(cm, columns=classes[class_list_label], index=classes[class_list_label])
    confuse.to_csv(cfg['experiment_folder'] + "/groupconfusion_matrix.csv")

if __name__ == '__main__':
    main()
