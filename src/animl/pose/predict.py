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
from typing import Union
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from animl.generator import manifest_dataloader
from animl.classification import load_model


def predict_viewpoints(dataset: pd.DataFrame,
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

    dataset['sequence'] = dataset['sequence_id'].diff().gt(6).cumsum().add(1)
    grouped = dataset.groupby('sequence')
    progressBar = trange(len(dataset['sequence'].value_counts()))
    with torch.no_grad():
        for group in grouped: #sequence group
            df = group[1]
            half1 = df.loc[df.groupby(['camera']).ngroup() == 0].reset_index()
            half2 = df.loc[df.groupby(['camera']).ngroup() == 1].reset_index()
            if len(half2) > 0: # if there are 2 cameras
                group1 = manifest_dataloader(half1, batch_size=len(half1), workers=1, 
                                file_col=half1['filepath'], crop=True)
                for batch in enumerate(group1):
                    g1batch = batch[1]
                    data = g1batch[0]
                    data = data.to(device)
                    prediction1 = model(data) # list of predictions
                    g1_sums = torch.sum(prediction1, dim=0) # sum predictions for each viewpoint
                    g1_pred = torch.argmax(g1_sums).item() # return column with the max sum for viewpoint prediction
                    
                    # get filepath
                    g1_paths = g1batch[1]

                group2 = manifest_dataloader(half2, batch_size=len(half2), workers=1, 
                                file_col=half2['filepath'], crop=True)
                for batch in enumerate(group2):
                    g2batch = batch[1]
                    data = g2batch[0]
                    data = data.to(device)
                    prediction2 = model(data) # list of predictions
                    g2_sums = torch.sum(prediction2, dim=0) # sum predictions for each viewpoint
                    g2_pred = torch.argmax(g2_sums).item() # return column with the max sum for viewpoint prediction

                    # get ground truth label
                    g2_paths = g2batch[1]

                """ #this is currently lowering the overall accuracy
                if g1_pred == g2_pred: # if viewpoint predictions are the same, whichever group has the higher summed probability will get that viewpoint
                    if g1_sums[g1_pred] > g2_sums[g2_pred]:
                        g2_pred = 0 if g1_pred == 1 else 1 # change g2 to be the opposite viewpoint of g1 if g1 sum is greater
                    elif g2_sums[g2_pred] > g1_sums[g1_pred]:
                        g1_pred = 0 if g2_pred == 1 else 1
                """
                g1 = [g1_pred] * len(g1_paths) # make list of the prediction labels to append to pred_labels, group prediction applies to whole group
                g2 = [g2_pred] * len(g2_paths)
                pred_labels.extend(g1)
                pred_labels.extend(g2)                

                # get file paths for each group
                filepaths.extend(g1_paths)
                filepaths.extend(g2_paths)
            else: # entire group is from the same station/stationID
                dl_group = manifest_dataloader(df, batch_size=len(df), workers=1, 
                                file_col=df['filepath'], crop=True)

                for batch in enumerate(dl_group):
                    g1batch = batch[1]
                    data = g1batch[0]
                    data = data.to(device)
                    prediction = model(data) # list of predictions

                    sums = torch.sum(prediction, dim=0) # sum predictions for each viewpoint
                    pred = torch.argmax(sums) # return column with the max sum for viewpoint prediction

                    paths = batch[1][1]
                    pred_labels.extend([pred.item()] * len(paths))
                    
                    filepaths.extend(paths)

            progressBar.update(1)
        
    return pred_labels, filepaths # output roi id


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
    pred, paths = predict_viewpoints(grouped, model, device)

    predictions = pd.DataFrame({'FilePath': paths,
                                'prediction': pred})
   
    merged = pd.merge(predictions, truth[truth.duplicated(subset=['FilePath'], keep='first') == False],  on=['FilePath'])
    merged = merged.drop_duplicates()
    print(merged)

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
