'''
Tools for Saving, Loading, and Using Species Classifiers

@ Kyra Swanson 2023
'''
from typing import Optional
import pandas as pd
import numpy as np
from pathlib import Path
from time import time
from tqdm import tqdm

import torch
import onnxruntime

from animl import generator, file_management
from animl.model_architecture import EfficientNet, ConvNeXtBase
from animl.utils.general import get_device, softmax, tensor_to_onnx, NUM_THREADS


def save_classifier(model,
                    out_dir: str,
                    epoch: int,
                    stats: dict,
                    optimizer=None,
                    scheduler=None):
    '''
    Saves model state weights.

    Args:
        model: pytorch model
        out_dir (str): directory to save model to
        epoch (int): current training epoch
        stats (dict): performance metrics of current epoch
        optimizer: pytorch optimizer (optional)
        scheduler: pytorch scheduler (optional)

    Returns:
        None
    '''
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # get model parameters and add to stats
    checkpoint = {'model': model.state_dict(),
                  'stats': stats}
    # save optimizer and scheduler state dicts if they are provided
    if optimizer is not None or scheduler is not None:
        checkpoint['epoch'] = epoch
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()

    torch.save(checkpoint, open(f'{out_dir}/{epoch}.pt', 'wb'))


def load_classifier(model_path: str,
                    num_classes: int,
                    device: Optional[str] = None,
                    architecture: str = "CTL"):
    '''
    Creates a model instance and loads the latest model state weights.

    Args:
        model_path (str): file or directory path to model weights
        num_classes (int): path to associated class list
        device (str): specify to run on cpu or gpu
        architecture (str): expected model architecture

    Returns:
        model: model object of given architecture with loaded weights
        start_epoch (int, optional): current epoch, 0 if not resuming training
    '''
    model_path = Path(model_path)

    # check to make sure GPU is available if chosen
    if device is None:
        device = get_device()
    print('Device set to', device)

    # Create a new model intance for training
    if model_path.is_dir():
        model_path = str(model_path)
        start_epoch = 0
        if (architecture == "CTL") or (architecture == "efficientnet_v2_m"):
            model = EfficientNet(num_classes, device=device)
        elif architecture == "convnext_base":
            model = ConvNeXtBase(num_classes)
        else:  # can only resume models from a directory at this time
            raise AssertionError('Please provide the correct model')
        return model, start_epoch

    # load a specific model file
    elif model_path.is_file():
        print(f'Loading model at {model_path}')
        start_time = time()
        # PyTorch dict
        if model_path.suffix == '.pt':
            if (architecture == "CTL") or (architecture == "efficientnet_v2_m"):
                model = EfficientNet(num_classes, device=device, tune=False)
                # TODO: torch 2.6 defaults to weights_only = True
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model'])
                model.to(device)
                model.eval()
                model.framework = "EfficientNet"
            elif architecture == "convnext_base":
                model = ConvNeXtBase(num_classes, tune=False)
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


def load_class_list(classlist_file):
    """
    Return classlist file as pd.DataFrame.

    Args:
        classlist_file (str): file path to class list

    Returns:
        pd.DataFrame of class list
    """
    return pd.read_csv(classlist_file)


# TODO: change default resize after training models
def classify(model,
             detections,
             device: Optional[str] = None,
             out_file: Optional[str] = None,
             file_col: str = 'Frame',
             crop: bool = True,
             normalize: bool = True,
             resize_width: int = 299,
             resize_height: int = 299,
             batch_size: int = 1,
             num_workers: int = NUM_THREADS):
    """
    Predict species using classifier model.

    Args:
        model: preloaded classifier model
        detections (mult): dataframe of (animal) detections, list of filepaths or filepath str
        device (str): specify to run model on cpu or gpu, default to cpu
        out_file (str): path to save prediction results to
        file_col (str): column name containing file paths
        crop (bool): use bbox to crop images before feeding into model
        normalize (bool): normalize the tensor before inference
        resize_width (int): image width input size
        resize_height (int): image height input size
        batch_size (int): data generator batch size
        num_workers (int): number of cores

    Returns:
        detections (pd.DataFrame): MD detections with classifier prediction and confidence
    """
    if file_management.check_file(out_file):
        return file_management.load_data(out_file)

    if device is None:
        device = get_device()

    # initialize lists
    raw_output = []

    # Manifest
    if isinstance(detections, pd.DataFrame):
        dataset = generator.manifest_dataloader(detections, file_col=file_col, crop=crop,
                                                resize_width=resize_width, resize_height=resize_height,
                                                normalize=normalize, batch_size=batch_size, num_workers=num_workers)
    # Single File
    elif isinstance(detections, str):
        detections = pd.DataFrame({file_col: detections}, index=[0])
        dataset = generator.manifest_dataloader(detections, file_col=file_col, crop=False,
                                                resize_width=resize_width, resize_height=resize_height,
                                                normalize=normalize, batch_size=1, num_workers=1)
    # List of Files
    elif isinstance(detections, list):
        detections = pd.DataFrame({file_col: detections}, index=range(len(detections)))
        dataset = generator.manifest_dataloader(detections, file_col=file_col, crop=False,
                                                resize_width=resize_width, resize_height=resize_height,
                                                normalize=normalize, batch_size=batch_size, num_workers=1)
    else:
        raise AssertionError("Input must be a data frame of crops, single file path or vector of file paths.")

    # Predict
    with torch.no_grad():
        for _, batch in tqdm(enumerate(dataset), total=len(dataset)):
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

    raw_output = np.vstack(raw_output)

    if out_file:
        file_management.save_data(pd.DataFrame(raw_output), out_file)

    return raw_output


def individual_classification(animals: pd.DataFrame,
                              predictions_raw: np.array,
                              class_list: pd.DataFrame):
    """
    Get maximum likelihood prediction from softmaxed logits.

    Args:
        animals (pd.DataFrame): animal detections from manifest
        predictions_raw (np.array): softmaxed logits from classify()
        class_list (pd.DataFrame): class list associated with model

    Returns:
        animals dataframe with "prediction" label an "confidence" columns
    """
    class_list = pd.Series(class_list)
    animals["prediction"] = list(class_list[np.argmax(predictions_raw, axis=1)])
    animals["confidence"] = animals["conf"].mul(np.max(predictions_raw, axis=1))
    return animals


def sequence_classification(animals: pd.DataFrame,
                            empty: Optional[pd.DataFrame],
                            predictions_raw: np.array,
                            class_list: pd.DataFrame,
                            station_col: str,
                            empty_class: str = "",
                            sort_columns: list[str] = None,
                            file_col: str = "FilePath",
                            maxdiff: int = 60):
    """
    Applies class labels to images based on sequential information.

    This function applies image classifications at a sequence level by leveraging information from
    multiple images. A sequence is defined as all images at the same camera and station where the
    time between consecutive images is <=maxdiff. This can improve classification accuracy, but
    assumes that only one species is present in each sequence. If you regularly expect multiple
    species to occur in an image or sequence don't use this function.

    Args:
        animals (pd.DataFrame): Sub-selection of all images that contain animals
        empty (Optional) (pd.DataFrame): Sub-selection of all images that do not contain animals
        predictions (Numpy Array of Numpy Arrays): Logits of all entries in "animals"
        class_list (pd.DataFrame): class list associated with classifier model
        station_col (str): The name of the station column
        empty_class (str) (Optional): the name of class_list 'empty' label
        sort_columns (List of Strings): Defines sorting order for the DataFrame
        file_col (str): The name of the filepath column
        maxdiff (int): Maximum time difference in seconds between any two images in a sequence

    Returns:
        final_df (pd.DataFrame): Sequence classified data from both animals and empty

    Raises:
        Exception: If 'sort_columns' is not a list or is empty
        Exception: If 'station_col' is not a string or is empty
        Exception: If 'empty' is defined and is not a pandas DataFrame
        Exception: If maxdiff is defined and is not a positive number
    """
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
        empty_col = class_list[class_list == empty_class].index[0]
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

        while (last_index < len(animals_sort) and not pd.isna(animals_sort.loc[i, "DateTime"]) and
               not pd.isna(animals_sort.loc[last_index, "DateTime"]) and
               animals_sort.loc[last_index, station_col] == animals_sort.loc[i, station_col] and
               (animals_sort.loc[last_index, "FileModifyDate"] - animals_sort.loc[i, "FileModifyDate"]).total_seconds() <= maxdiff):
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
                sel_no_empties = np.where(filtered_animals.isin(sel_all_empty[~sel_all_empty[0]].index) & (predclass != empty_col))[0]

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
        s += 1

    animals_sort['confidence'] = conf_placeholder
    animals_sort['prediction'] = predict_placeholder
    animals_sort['sequence'] = sequence_placeholder

    return animals_sort


# TODO
def multi_species_detection(animals: pd.DataFrame,
                            threshold: float,
                            file_col: str = "FilePath") -> pd.DataFrame:
    """
    This function applies image classifications at a image level. All images which have multiple
    species present with confidence above threshold, will be returned as a DataFrame

    @ Ayush Singh 2024

    Args:
        animals (Pandas DataFrame): Sub-selection of all images that contain animals
        threshold (float): Minimum confidence for the image to be considered
        file_col (str): The name of the filepath column

    Raises:
        - Exception: If threshold is not a float or threshold < 0 or threshold > 1

    Returns:
        result_df (Pandas DataFrame): Rows from images having more than one species
    """
    # Sanity check to verify that threshold is a float and is in range [0,1]
    if (not isinstance(threshold, float)) or (threshold < 0) or (threshold > 1):
        raise Exception("Threshold must be a value between 0 and 1, both inclusive")

    # Sorting by file name to accumulate all rows belonging to the same image
    animals[file_col] = animals[file_col].astype(str)
    animals = animals.sort_values(by=file_col)

    # Initializing data frame to store the result
    result_df = pd.DataFrame()

    # Making a new column for count
    result_df['count'] = []

    # List to store all the rows which have the same file name
    curr_picture = []

    # Iterating through all rows and gathering the ones which belong to the same image
    for index, row in animals.iterrows():
        # Initializing the list when it's empty
        if len(curr_picture) == 0:
            curr_picture.append(row)

        # Check if row belongs to the same image
        elif row[file_col] == curr_picture[0][file_col]:
            curr_picture.append(row)

        # All rows for the current image have been collected
        else:
            # Key is the specie, value is list [row, count]
            dic = {}

            # For all images above threshold, save the one with highest confidence for each class
            for element in curr_picture:
                if element['confidence'] > threshold:
                    if element['prediction'] in dic:
                        dic[element['prediction']][1] += 1
                        if dic[element['prediction']][0]['confidence'] < element['confidence']:
                            dic[element['prediction']][0] = element
                    else:
                        dic[element['prediction']] = [element, 1]

            # Make current image a part of output only if it has more than 1 species
            if len(dic) > 1:
                for key in dic:
                    dic[key][0]['count'] = dic[key][1]
                    result_df = pd.concat([result_df, pd.DataFrame([dic[key][0]])])

            # Reset list for next image
            curr_picture = [row]

    # Handeling the last batch
    dic = {}
    for element in curr_picture:
        if element['confidence'] > threshold and element['prediction'] != 'empty':
            if element['prediction'] in dic:
                dic[element['prediction']][1] += 1
                if dic[element['prediction']][0]['confidence'] < element['confidence']:
                    dic[element['prediction']][0] = element
            else:
                dic[element['prediction']] = [element, 1]

    # Make current image a part of output only if it has more than 1 species
    if len(dic) > 1:
        for key in dic:
            dic[key][0]['count'] = dic[key][1]
            result_df = pd.concat([result_df, pd.DataFrame([dic[key][0]])])

    return result_df
