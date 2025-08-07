
import argparse
import yaml
from tqdm import trange
import pandas as pd
import torch
from typing import Union
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from animl.classification import load_classifier
from animl.generator import manifest_dataloader
from animl.api import matchypatchy


def predict_viewpoints(dataset: pd.DataFrame,
                       model: torch.nn.Module, 
                       device: Union[str, torch.device] = 'cpu'):
    '''
        Run trained model on test split

        Args:
            - data_loader: test set dataloader
            - model: trained model object
            - device: run model on gpu or cpu, defaults to cpu

        Return:
            - pred_labels
            - filepaths
    '''
    model.to(device)
    model.eval()  # put the model into training mode

    pred_labels = []
    filepaths = []

    dataset['sequence_group'] = dataset.sort_values("sequence")['sequence'].diff().gt(6).cumsum().add(1)
    grouped = dataset.groupby('sequence_group')
    progressBar = trange(grouped.ngroups)
    with torch.no_grad():
        for group in grouped: #sequence group
            df = group[1]
            half1 = df.loc[df.groupby(['camera']).ngroup() == 0].reset_index() # split by left/right camera
            half2 = df.loc[df.groupby(['camera']).ngroup() == 1].reset_index()
            if len(half2) > 0: # if there are 2 cameras
                group1 = manifest_dataloader(half1, 'FilePath', normalize=True, batch_size=len(half1), num_workers=8)
                g1_pred, g1_paths, g1_sums = enumerate_dl(half1, group1, device, model)

                group2 = manifest_dataloader(half2, 'FilePath', normalize=True, batch_size=len(half2), num_workers=8)
                g2_pred, g2_paths, g2_sums = enumerate_dl(half2, group2, device, model)

                if g1_pred == g2_pred: # if viewpoint predictions are the same, whichever group has the higher summed probability will get that viewpoint
                    if g1_sums[g1_pred] > g2_sums[g2_pred]:
                        g2_pred = 0 if g1_pred == 1 else 1 # change g2 to be the opposite viewpoint of g1 if g1 sum is greater
                    elif g2_sums[g2_pred] > g1_sums[g1_pred]:
                        g1_pred = 0 if g2_pred == 1 else 1
                
                g1 = [g1_pred] * len(half1) # make list of the prediction labels to append to pred_labels, group prediction applies to whole group
                g2 = [g2_pred] * len(half2)
                pred_labels.extend(g1)
                pred_labels.extend(g2)                
                # get file paths for each group
                filepaths.extend(g1_paths)
                filepaths.extend(g2_paths)
            else: # entire group is from the same station/stationID
                dl_group = manifest_dataloader(df, 'FilePath', normalize=False, batch_size=len(df), num_workers=8)
                for batch in enumerate(dl_group):
                    g1batch = batch[1]
                    data = g1batch[0]
                    data = data.to(device)
                    prediction = model(data) # list of predictions
                    sums = torch.sum(prediction, dim=0) # sum predictions for each viewpoint
                    pred = torch.argmax(sums).item()    # return column with the max sum for viewpoint prediction

                    paths = batch[1][1]
                pred_labels.extend([pred] * len(df))
                filepaths.extend(paths)
            progressBar.update(1)
        
    #return pred_labels, filepaths # output roi id
    
    return pred_labels, filepaths

def enumerate_dl(df, dataloader, device, model):
    for batches in enumerate(dataloader):
        batch = batches[1]
        data = batch[0]
        data = data.to(device)
        prediction = model(data) # list of predictions
        sums = torch.sum(prediction, dim=0) # sum predictions for each viewpoint
        pred = torch.argmax(sums).item()    # return column with the max sum for viewpoint prediction

        paths = batch[1]            
                    
    return pred, paths, sums

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
    model = load_classifier(cfg['experiment_folder'], len(classes), device=device, architecture=cfg['architecture'])
    model = model[0]
    class_list_label = cfg.get('class_list_label', 'class')

    dataset = pd.read_csv(cfg['test_set'])

    truth = dataset[['FilePath', 'viewpoint']].copy() #get ground truth viewpoint for each datapoint
    # get predictions
    pred, paths = predict_viewpoints(dataset, model, device)
    predictions = pd.DataFrame({'FilePath': paths,
                                'prediction': pred})
    merged = pd.merge(predictions, truth, on=['FilePath'])    
    merged = merged.drop_duplicates(subset=['FilePath'])

    oa = (merged['prediction'] == merged['viewpoint']).mean()
    prec = precision_score(merged['viewpoint'], merged['prediction'], average='binary')
    recall = recall_score(merged['viewpoint'], merged['prediction'], average='binary')
    print(f"Test accuracy: {oa}")
    print(f"Precision: {prec}")
    print(f"Recall: {recall}")

    results = pd.DataFrame({'FilePath': paths,
                            #'Ground Truth': true,
                            'Predicted': pred})
    results.to_csv(cfg['experiment_folder'] + "/grouptest_results.csv")

    cm = confusion_matrix(merged['viewpoint'], merged['prediction'])
    confuse = pd.DataFrame(cm, columns=classes[class_list_label], index=classes[class_list_label])
    confuse.to_csv(cfg['experiment_folder'] + "/groupconfusion_matrix.csv")

if __name__ == '__main__':
    main()
