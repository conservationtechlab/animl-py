"""
Predict viewpoints from a sequence of images using a viewpoint model

"""
from tqdm import tqdm
import torch

from animl.utils.general import get_device
from animl.generator import manifest_dataloader


def predict_by_camera(model, dataloader, device=None):
    """
    Run model on a sequence from a single camera

    Args:
        model: viewpoint model object
        dataloader: dataloader for the camera
        device: the device to run the model on (default is determined by get_device)

    Returns:
        pred: the predicted viewpoint class
        paths: the file paths of the images in the sequence
        sums: the summed probabilities for each viewpoint class
    """
    if device is None:
        device = get_device()

    # batch size is len of half, should have one batch per camera
    for batches in enumerate(dataloader):
        batch = batches[1]
        data = batch[0]
        data = data.to(device)
        prediction = model(data)  # list of predictions
        sums = torch.sum(prediction, dim=0)  # sum predictions for each viewpoint
        pred = torch.argmax(sums).item()  # return column with the max sum for viewpoint prediction
        paths = batch[1]
    return pred, paths, sums


def predict_viewpoints(model, dataset, device=None):
    '''
    Run viewpoint model on two-camera sequences to improve prediction accuracy

    Args:
        model: viewpoint model object
        data_loader: pd.DataFrame with columns 'FilePath', 'camera', 'sequence'
        device: run model on gpu or cpu, defaults to cpu

    Return:
        pred_labels
        filepaths
    '''
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()  # put the model into training mode

    pred_labels = []
    filepaths = []

    # sort dataset by sequence, then group by sequence
    dataset['sequence_group'] = dataset.sort_values("sequence")['sequence'].diff().gt(6).cumsum().add(1)
    sequences = dataset.groupby('sequence_group')

    with torch.no_grad():
        for group in tqdm(sequences):
            df = group[1]
            # split by left/right camera
            half1 = df.loc[df.groupby(['camera']).ngroup() == 0].reset_index()
            half2 = df.loc[df.groupby(['camera']).ngroup() == 1].reset_index()

            # if there are 2 cameras
            if len(half2) > 0:
                group1 = manifest_dataloader(half1, 'FilePath', normalize=True, batch_size=len(half1), num_workers=8)
                g1_pred, g1_paths, g1_sums = predict_by_camera(model, group1, device)

                group2 = manifest_dataloader(half2, 'FilePath', normalize=True, batch_size=len(half2), num_workers=8)
                g2_pred, g2_paths, g2_sums = predict_by_camera(model, group2, device)

                # if viewpoint predictions are the same, whichever group has the higher summed probability will get that viewpoint
                if g1_pred == g2_pred:
                    # change g2 to be the opposite viewpoint of g1 if g1 sum is greater
                    if g1_sums[g1_pred] > g2_sums[g2_pred]:
                        g2_pred = 0 if g1_pred == 1 else 1
                    elif g2_sums[g2_pred] > g1_sums[g1_pred]:
                        g1_pred = 0 if g2_pred == 1 else 1

                # make list of the prediction labels to append to pred_labels, group prediction applies to whole group
                g1 = [g1_pred] * len(half1)
                g2 = [g2_pred] * len(half2)
                pred_labels.extend(g1)
                pred_labels.extend(g2)
                # get file paths for each group
                filepaths.extend(g1_paths)
                filepaths.extend(g2_paths)

            # entire group is from the same camera
            else:
                dl_group = manifest_dataloader(df, 'FilePath', normalize=False, batch_size=len(df), num_workers=8)
                pred, paths, sums = predict_by_camera(dl_group, device, model)
                pred_labels.extend([pred] * len(df))
                filepaths.extend(paths)

    return pred_labels, filepaths
