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

from animl.generator import train_dataloader
from animl.classification import load_model


def predict_viewpoints(dataset: pd.DataFrame, 
                       categories: dict, 
                       cfg,
                       crop: bool,
                       model: torch.nn.Module, 
                       device: Union[str, torch.device] = 'cpu') -> float:
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
    true_labels = []
    probs = []
    filepaths = []
    

    

    progressBar = dataset['group'].nunique()
    with torch.no_grad():
        for group in dataset: #sequence group
            half1 = group.iloc[:len(group) // 2, :]
            half2 = group.iloc[len(group) // 2:, :]
            group1 = train_dataloader(half1, categories, batch_size=len(half1), workers=cfg['num_workers'], 
                               file_col=cfg.get('file_col', 'FilePath'), label_col=cfg.get('label_col', 'viewpoint'), 
                               crop=crop, augment=False, cache_dir=cfg.get('cache_folder', None), crop_coord=cfg['crop_coord'])
            #for batch in group1: #camera side group, treating idx as camera id for now?
                # forward pass
            g1_data = group1[0]
            g1_data = g1_data.to(device)
            prediction1 = model(g1_data) #list of tensors
            prediction1stack = torch.stack(prediction1)
            #columns = [0, 1] # index 0 = left prediction, 1 = right prediction
            g1_sums = torch.sum(prediction1stack, dim=0) # sum predictions for each viewpoint
            g1_pred = torch.argmax(g1_sums) # return column with the max sum for prediction
            #g1_right = torch.sum(prediction[:, 1]) #sum right predictions (index 1)

            group2 = train_dataloader(half2, categories, batch_size=len(half2), workers=cfg['num_workers'], 
                           file_col=cfg.get('file_col', 'FilePath'), label_col=cfg.get('label_col', 'viewpoint'), 
                           crop=crop, augment=False, cache_dir=cfg.get('cache_folder', None), crop_coord=cfg['crop_coord'])
            #for batch in group2: #camera side group
                # forward pass
            g2_data = group2[0]
            g2_data = g2_data.to(device)
            prediction2 = model(g2_data)
            prediction2stack = torch.stack(prediction2)
            g2_sums = torch.sum(prediction2stack, dim=0) #list of sums of each column
            g2_pred = torch.argmax(g2_sums)

            if g1_pred == g2_pred: # if viewpoint predictions are the same, whichever group has the higher summed probability will get that viewpoint
                if g1_sums[g1_pred] > g2_sums[g2_pred]:
                    g2_pred = 0 if g1_pred == 1 else 1 # change g2 to be the opposite viewpoint of g1 if g1 sum is greater
                elif g2_sums[g2_pred] > g1_sums[g1_pred]:
                    g1_pred = 0 if g2_pred == 1 else 1
            g1 = [g1_pred] * len(group1) # make list of the prediction labels to append to pred_labels, group prediction applies to whole group
            g2 = [g2_pred] * len(group2)
            pred_labels.extend(g1, g2)

            # get ground truth labels for each group
            g1_labels = group1[1]
            labels1_np = g1_labels.numpy()

            g2_labels = group2[1]
            labels2_np = g2_labels.numpy()
            true_labels.extend(labels1_np, labels2_np)

            #get probabilities for each label in each group
            #g1_probs = torch.gather(input=prediction1,dim=1,index=g1_pred)
            #g2_probs = torch.gather(input=prediction2,dim=1,index=g2_pred)
            probs.extend(prediction1[:,g1_pred], prediction2[:,g2_pred])

            #get file paths for each group
            g1_paths = group1[2]
            g2_paths = group2[2]
            filepaths.extend(g1_paths, g2_paths)
            

            progressBar.update(1)
        

    return pred_labels, true_labels, filepaths


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
    class_list_index = cfg.get('class_list_index', 'id')

    categories = dict([[x[class_list_label], x[class_list_index]] for _, x in classes.iterrows()])

    # initialize dataset with datapoints grouped by CaptureID
    dataset = pd.read_csv(cfg['test_set'])
    #df = dataset.sort_values('CaptureID').reset_index(drop=True)
    #df['group'] = (df['CaptureID'].diff().abs() > 6).cumsum() #camera capture sequences are in groups of 6 maximum
    dataset['group'] = dataset['CaptureID'].diff().gt(6).cumsum().add(1)
    grouped = dataset.groupby('group')

    # get predictions
    pred, true, paths = predict_viewpoints(grouped, categories, cfg, crop, model, device)
    pred = np.asarray(pred)
    true = np.asarray(true)

    oa = np.mean((pred == true))
    print(f"Test accuracy: {oa}")

    results = pd.DataFrame({'FilePath': paths,
                            'Ground Truth': true,
                            'Predicted': pred})
    results.to_csv(cfg['experiment_folder'] + "/grouptest_results.csv")

    cm = confusion_matrix(true, pred)
    confuse = pd.DataFrame(cm, columns=classes[class_list_label], index=classes[class_list_label])
    confuse.to_csv(cfg['experiment_folder'] + "/groupconfusion_matrix.csv")


if __name__ == '__main__':
    main()
