'''
    Tools for Saving, Loading, and Building Species Classifiers

    @ Kyra Swanson 2023
'''
import argparse
import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from time import time
from tqdm import tqdm

import torch
import torch.onnx
import onnxruntime

from animl import generator, file_management, split
from animl.model_architecture import EfficientNet, ConvNeXtBase
from animl.utils.torch_utils import get_device


def softmax(x):
    '''
    Helper function to softmax
    '''
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)


def save_model(out_dir, epoch, model, stats):
    '''
    Saves model state weights.

    Args:
        - out_dir (str): directory to save model to
        - epoch (int): current training epoch
        - model: pytorch model
        - stats (dict): performance metrics of current epoch

    Returns:
        None
    '''
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # get model parameters and add to stats
    stats['model'] = model.state_dict()

    torch.save(stats, open(f'{out_dir}/{epoch}.pt', 'wb'))


def load_model(model_path, classes, device=None, architecture="CTL"):
    '''
    Creates a model instance and loads the latest model state weights.

    Args:
        - model_path (str): file or directory path to model weights
        - class_file (str): path to associated class list
        - device (str): specify to run on cpu or gpu
        - architecture (str): expected model architecture

    Returns:
        - model: model object of given architecture with loaded weights
        - classes: associated species class list
        - start_epoch (int, optional): current epoch, 0 if not resuming training
    '''
    # read class file
    model_path = Path(model_path)

    # check to make sure GPU is available if chosen
    if device is None:
        device = get_device()
    print('Device set to', device)

    # load latest model state from given folder
    if model_path.is_dir():
        model_path = str(model_path)
        start_epoch = 0
        if (architecture == "CTL") or (architecture == "efficientnet_v2_m"):
            model = EfficientNet(classes)
        elif architecture == "convnext_base":
            model = ConvNeXtBase(classes)
        else:  # can only resume models from a directory at this time
            raise AssertionError('Please provide the correct model')

        model_states = []
        for file in os.listdir(model_path):
            if os.path.splitext(file)[1] == ".pt":
                model_states.append(file)

        if len(model_states):
            # at least one save state found; get latest
            model_epochs = [int(m.replace(model_path, '').replace('.pt', '')) for m in model_states]
            start_epoch = max(model_epochs)

            # load state dict and apply weights to model
            print(f'Resuming from epoch {start_epoch}')
            state = torch.load(open(f'{model_path}/{start_epoch}.pt', 'rb'))
            model.load_state_dict(state['model'])
        else:
            # no save state found; start anew
            print('No model state found, starting new model')

        return model, start_epoch

    # load a specific model file
    elif model_path.is_file():
        print(f'Loading model at {model_path}')
        start_time = time()
        # TensorFlow
        # if model_path.endswith('.h5'):
        #    model = keras.models.load_model(model_path)
        # PyTorch dict
        if model_path.suffix == '.pt':
            if (architecture == "CTL") or (architecture == "efficientnet_v2_m"):
                model = EfficientNet(classes, tune=False)
                # TODO: torch 2.6 defaults to weights_only = True
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model'])
                model.to(device)
                model.eval()
                model.framework = "EfficientNet"
            elif architecture == "convnext_base":
                model = ConvNeXtBase(classes, tune=False)
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model'])
                model.to(device)
                model.eval()
                model.framework = "ConvNeXt-Base"
        # PyTorch full model
        elif model_path.suffix == '.pth':
            model = torch.load(model_path, map_location=device)
            model.to(device)
            model.eval()
            model.framework = "pytorch"
        elif model_path.suffix == '.onnx':
            if device == "cpu":
                model = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            else:
                model = onnxruntime.InferenceSession(model_path, providers=["CUDAExecutionProvider", 'CPUExecutionProvider'])
            model.framework = "onnx"
        else:
            raise ValueError('Unrecognized model format: {}'.format(model_path))
        elapsed = time() - start_time
        print('Loaded model in %.2f seconds' % elapsed)

        # no need to return epoch
        return model

    # no dir or file found
    else:
        raise ValueError("Model not found at given path")


def tensor_to_onnx(tensor, channel_last=True):
    '''
    Helper function for onnx, shifts dims to BxHxWxC
    '''
    if channel_last:
        tensor = tensor.permute(0, 2, 3, 1)  # reorder BxCxHxW to BxHxWxC

    tensor = tensor.numpy()

    return tensor


def predict_species(detections, model,
                    device=None, out_file=None,
                    file_col='Frame', crop=True,
                    resize_width=299, resize_height=299,
                    normalize=True, batch_size=1, workers=1):
    """
    Predict species using classifier model

    Args
        - detections (pd.DataFrame): dataframe of (animal) detections
        - model: preloaded classifier model
        - classes: preloaded class list, pd Series
        - device (str): specify to run model on cpu or gpu, default to cpu
        - out_file (str): path to save prediction results to
        - raw (bool): return raw logits instead of applying labels
        - file_col (str): column name containing file paths
        - crop (bool): use bbox to crop images before feeding into model
        - resize_width (int): image width input size
        - resize_height (int): image height input size
        - normalize (bool): normalize the tensor before inference
        - batch_size (int): data generator batch size
        - workers (int): number of cores

    Returns
        - detections (pd.DataFrame): MD detections with classifier prediction and confidence
    """
    if file_management.check_file(out_file):
        return file_management.load_data(out_file)

    if not torch.cuda.is_available():
        device = 'cpu'
    elif torch.cuda.is_available() and device is None:
        device = 'cuda:0'
    else:
        device = device

    if isinstance(detections, pd.DataFrame):
        # initialize lists
        raw_output = []

        dataset = generator.manifest_dataloader(detections, file_col=file_col, crop=crop,
                                                resize_width=resize_width, resize_height=resize_height,
                                                normalize=normalize, batch_size=batch_size, workers=workers)

        with torch.no_grad():
            for _, batch in tqdm(enumerate(dataset)):
                # pytorch
                if model.framework == "pytorch" or model.framework == "EfficientNet":
                    data = batch[0]
                    data = data.to(device)
                    output = model(data)
                    raw_output.extend(torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy())

                # onnx
                elif model.framework == "onnx":
                    data = batch[0]
                    data = tensor_to_onnx(data)
                    output = model.run(None, {model.get_inputs()[0].name: data})[0]
                    raw_output.extend(softmax(output))

                else:
                    raise AssertionError("Model architechture not supported.")
    else:
        raise AssertionError("Input must be a data frame of crops or vector of file names.")

    raw_output = np.vstack(raw_output)

    if out_file:
        file_management.save_data(pd.DataFrame(raw_output), out_file)

    return raw_output
    

def single_classification(animals, predictions_raw, class_list):   
    """
    Get maximum likelihood prediction from softmaxed logits
    """
    class_list = pd.Series(class_list)
    animals["prediction"] = list(class_list[np.argmax(predictions_raw, axis=1)])
    animals["confidence"] = animals["conf"].mul(np.max(predictions_raw, axis=1))
    return animals



def sequence_classification(animals, empty, predictions_raw, class_list, station_col,
                            empty_class="empty", sort_columns=None,
                            file_col="FilePath", maxdiff=60):
    """
    This function applies image classifications at a sequence level by leveraging information from
    multiple images. A sequence is defined as all images at the same camera and station where the
    time between consecutive images is <=maxdiff. This can improve classification accuracy, but
    assumes that only one species is present in each sequence. If you regularly expect multiple
    species to occur in an image or sequence don't use this function.

    Parameters:
    - animals (Pandas DataFrame): Sub-selection of all images that contain animals
    - sort_columns (List of Strings): Defines sorting order for the DataFrame
    - predictions (Numpy Array of Numpy Arrays): Logits of all entries in "animals"
    - species (CSV File): File mapping index to species
    - station_col (String): The name of the station column
    - empty (Optional) (Pandas DataFrame): Sub-selection of all images that do not contain animals
    - maxdiff (float) (Optional): Maximum time difference between any two images in a sequence

    Raises:
    - Exception: If 'animals' is not a pandas DataFrame
    - Exception: If 'sort_columns' is not a list or is empty
    - Exception: If 'station_col' is not a string or is empty
    - Exception: If 'empty' is defined and is not a pandas DataFrame
    - Exception: If maxdiff is defined and is not a positive number

    Output:
    - final_df (Pandas DataFrame): Sequence classified data from both animals and empty
    """

    # Sanity check to verify that animals is a Pandas DataFrame
    if not isinstance(animals, pd.DataFrame):
        raise Exception("'animals' must be a DataFrame")

    if not isinstance(station_col, str) or station_col == '':
        raise Exception("'station_col' must be a non-empty string")

    # Sanity check to verify that empty is a Pandas DataFrame, if defined
    if empty is not None and not isinstance(animals, pd.DataFrame):
        raise Exception("'empty' must be a DataFrame")

    # Sanity check to verify that maxdiff is a positive number
    if not isinstance(maxdiff, (int, float)) or maxdiff < 0:
        raise Exception("'maxdiff' must be a number >= 0")

    if "conf" not in animals.columns:
        animals["conf"] = 1

    if empty_class > "":
        empty_col = class_list[class_list == "empty"].index[0]
    else:
        empty_col = None

    if empty is not None or not empty.empty:
        empty["ID"] = range(0, empty.shape[0])
        predempty = empty.pivot(index="ID", columns="prediction", values="confidence")
        # Replace NaN with 0
        predempty = predempty.fillna(0)
        predempty = pd.concat([pd.DataFrame(np.zeros((empty.shape[0], len(class_list)))), predempty], axis=1)

        if empty_class > "":
            # replace md empty with empty col
            predempty[empty_col] = predempty["empty"]
            predempty = predempty.drop("empty", axis=1)
            class_list = pd.concat([class_list,
                                    pd.Series([x for x in empty["prediction"].unique() if x != "empty"])], ignore_index=True)

        else:
            class_list = pd.concat([class_list, pd.Series(empty["prediction"].unique())], ignore_index=True)
            empty_col = predempty.columns.get_loc("empty")

        # placeholders
        animals["prediction"] = list(class_list[np.argmax(predictions_raw, axis=1)])
        animals["confidence"] = animals["conf"].mul(np.max(predictions_raw, axis=1))

        empty["conf"] = 1
        animals_merged = pd.concat([animals, empty.iloc[:, :-1]]).reset_index(drop=True)  # dont add ID column
        predictions = np.hstack((predictions_raw,
                                 np.zeros((predictions_raw.shape[0], len(predempty.columns) - predictions_raw.shape[1]))))
        # concat
        predictions = np.vstack((predictions, np.array(predempty)))

    if sort_columns is None:
        sort_columns = [station_col, "DateTime"]

    animals_merged['FileModifyDate'] = pd.to_datetime(animals_merged['FileModifyDate'], format="%Y-%m-%d %H:%M:%S")

    sort = animals_merged.sort_values(by=sort_columns).index
    animals_sort = animals_merged.loc[sort].reset_index(drop=True)
    predsort = predictions[sort]

    conf_placeholder = np.zeros(len(animals_sort))
    predict_placeholder = np.empty(len(animals_sort), dtype='U30')
    sequence_placeholder = np.zeros(len(animals_sort))

    i = 0
    s = 0
    while i < len(animals_sort):
        rows = [i]
        last_index = i+1

        while last_index < len(animals_sort) and not pd.isna(animals_sort.loc[i, "DateTime"]) \
            and not pd.isna(animals_sort.loc[last_index, "DateTime"]) \
            and animals_sort.loc[last_index, station_col] == animals_sort.loc[i, station_col] \
            and (animals_sort.loc[last_index, "FileModifyDate"] - animals_sort.loc[i, "FileModifyDate"]).total_seconds() <= maxdiff:
            rows.append(last_index)
            last_index += 1

        rows = np.array(rows)

        # multiple detections in sequence
        if len(rows) > 1:
            predclass = np.argmax(predsort[rows], axis=1)

            # no empties
            if empty_col is None or empty_col not in predclass:
                predsort_confidence = predsort[rows] * np.reshape(animals_sort.loc[rows, 'conf'].values, (-1, 1))
                predbest = np.mean(predsort_confidence, axis=0)
                conf_placeholder[rows] = np.max(predsort_confidence[:, np.argmax(predbest)])
                predict_placeholder[rows] = class_list[np.argmax(predbest)]

            else:
                mask = pd.DataFrame((predclass == empty_col))
                filtered_animals = animals_sort.loc[rows, file_col].reset_index(drop=True)
                sum_values = mask.groupby(filtered_animals).sum()
                count_values = mask.groupby(filtered_animals).count()

                sel_all_empty = count_values == sum_values
                sel_mixed = np.where(filtered_animals.isin(sel_all_empty[~sel_all_empty[0]].index))[0]
                sel_no_empties = np.where(filtered_animals.isin(sel_all_empty[~sel_all_empty[0]].index) & (predclass!=empty_col))[0]
                    
                if len(sel_mixed) > 0 and len(sel_no_empties) > 0:
                    predsort_confidence = predsort[rows[sel_no_empties]] * np.reshape(animals_sort.loc[rows[sel_no_empties], 'conf'].values, (-1, 1))
                    predbest = np.mean(predsort_confidence, axis=0)
                    conf_placeholder[rows[sel_mixed]] = np.max(predsort_confidence[:, np.argmax(predbest)])
                    predict_placeholder[rows[sel_mixed]] = class_list[np.argmax(predbest)]

                for file in sel_all_empty[sel_all_empty[0]].index:
                    empty_row = np.where(animals_sort[file_col] == file)
                    predsort_confidence = predsort[empty_row] * np.reshape(animals_sort.loc[empty_row, 'conf'].values, (-1, 1))
                    predbest = np.mean(predsort_confidence, axis=0)
                    conf_placeholder[empty_row] = np.max(predsort_confidence[:, np.argmax(predbest)])
                    predict_placeholder[empty_row] = class_list[np.argmax(predbest)]
        # single row in sequence
        else:
            predbest = predsort[rows]
            conf_placeholder[rows] = np.max(predbest * animals_sort.loc[rows, 'conf'].values)
            predict_placeholder[rows] = class_list[np.argmax(predbest)]

        i = last_index
        s+=1

    animals_sort['confidence'] = conf_placeholder
    animals_sort['prediction'] = predict_placeholder
    animals_sort['sequence'] = sequence_placeholder

    return animals_sort


def classify_with_config(config):
    """
    Run Classification from Config File

    Args:
        - config (str): path to config file

    Returns:
        predictions dataframe
    """
    # get config file
    print(f'Using config "{config}"')
    cfg = yaml.safe_load(open(config, 'r'))

    manifest = pd.read_csv(cfg['manifest'])

    # get available device
    device = get_device()

    classifier, class_list = load_model(cfg['classifier_file'], cfg['class_list'],
                                        device=device, architecture=cfg.get('class_list', "CTL"))

    if cfg.get('split_animals', True):
        manifest = split.get_animals(manifest)

    predictions = predict_species(manifest, classifier, class_list, device=device, out_file=cfg['out_file'],
                                  raw=cfg.get('raw', False), file_col=cfg['file_col'], crop=cfg['crop'],
                                  resize_width=cfg['resize_width'], resize_height=cfg['resize_height'],
                                  normalize=cfg.get('normalize', True), batch_size=cfg.get('batch_size', 1),
                                  workers=cfg.get('workers', 1))
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='exp_resnet18.yaml')
    args = parser.parse_args()
    classify_with_config(args.config)
