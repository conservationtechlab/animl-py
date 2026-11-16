"""
Code to run Miew_ID and other re-identification models

(https://github.com/WildMeOrg/wbia-plugin-miew-id)

"""
from pathlib import Path
from typing import Optional

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import onnxruntime as ort
from scipy.spatial.distance import cdist
import torch.nn.functional as F

from animl.model_architecture import MiewIdNet, MIEWID_SIZE
from animl.utils.general import get_torch_device, get_onnx_device
from animl.generator import manifest_dataloader


def load_miew(file_path: str,
              device: Optional[str] = None):
    """
    Load MiewID from file path

    Args:
        file_path (str): file path to model file
        device (str): device to load model to

    Returns:
        loaded miewid model object
    """
    if Path(file_path).suffix == '.onnx':
        providers = get_onnx_device(user_set=device)
        miew = ort.InferenceSession(file_path, providers=providers)
        miew.architecture = 'onnx'  # treat like generic onnx model
        return miew

    else:
        device = get_torch_device(user_set=device)
        print(f'Sending model to {device}')
        weights = torch.load(file_path, weights_only=True)
        miew = MiewIdNet(device=device)
        miew.to(device)
        miew.device = device
        miew.architecture = 'miewid'
        miew.load_state_dict(weights, strict=False)
        miew.eval()
    return miew


def extract_miew_embeddings(miew_model,
                            manifest: pd.DataFrame,
                            file_col: str = "filepath",
                            batch_size: int = 1,
                            num_workers: int = 1,
                            device: Optional[str] = None):
    """
    Wrapper for MiewID embedding extraction

    Args:
        miew_model: MiewID model object
        manifest (pd.DataFrame): dataframe with columns 'filepath', 'emb_id'
        file_col (str): column name for file paths in manifest
        batch_size (int): batch size for dataloader
        num_workers (int): number of workers for dataloader
        device (str): device to run model on

    Returns:
        output (np.ndarray): array of extracted embeddings
    """
    if not {file_col}.issubset(manifest.columns):
        raise ValueError(f"DataFrame must contain '{file_col}' column.")

    output = []

    if miew_model.architecture == 'onnx':
        dataloader = manifest_dataloader(manifest,
                                         file_col=file_col,
                                         crop=True,
                                         resize_height=MIEWID_SIZE,
                                         resize_width=MIEWID_SIZE,
                                         architecture="miewid",
                                         normalize=True,
                                         batch_size=1,
                                         num_workers=num_workers)
        for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            collated, failed = batch
            if collated is None:  # entire batch was bad
                continue
            img = collated[0].numpy()
            inp = miew_model.get_inputs()[0]
            emb = miew_model.run(None, {inp.name: img})[0]
            output.extend(emb)
        output = np.vstack(output)
    else:
        device = get_torch_device(user_set=device)
        dataloader = manifest_dataloader(manifest,
                                         file_col=file_col,
                                         crop=True,
                                         resize_height=MIEWID_SIZE,
                                         resize_width=MIEWID_SIZE,
                                         architecture="miewid",
                                         normalize=True,
                                         batch_size=batch_size,
                                         num_workers=num_workers)
        with torch.no_grad():
            for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                collated, failed = batch
                if collated is None:  # entire batch was bad
                    continue
                img = collated[0]
                emb = miew_model.extract_feat(img.to(device))
                output.extend(emb.detach().cpu().numpy())
        output = np.vstack(output)
    return output


# ==============================================================================
# Distance computation functions for re-identification tasks.
# ==============================================================================

def remove_diagonal(A):
    """
    Removes the diagonal elements from a square matrix.

    Args:
        A (torch.Tensor): Input square matrix.

    Returns:
        torch.Tensor: Matrix with diagonal elements removed.
    """
    print("A.shape", A.shape)
    if A.size(0) != A.size(1):
        raise ValueError("Input must be a square matrix")

    mask = ~torch.eye(A.size(0), dtype=torch.bool)
    return A[mask].reshape(A.size(0), -1)


def euclidean_squared_distance(input1, input2):
    """
    Computes euclidean squared distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = input1.size(0), input2.size(0)
    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat = mat1 + mat2
    distmat.addmm_(input1, input2.t(), beta=1, alpha=-2)
    return distmat


def cosine_distance(input1, input2):
    """
    Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat


def compute_distance_matrix(input1, input2, metric='euclidean'):
    """
    A wrapper function for computing distance matrix.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".

    Returns:
        torch.Tensor: distance matrix.

    """
    if not isinstance(input1, torch.Tensor):
        input1 = torch.from_numpy(input1)
    if not isinstance(input2, torch.Tensor):
        input2 = torch.from_numpy(input2)
    # check input
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(input1.dim())
    assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(input2.dim())
    assert input1.size(1) == input2.size(1)

    if metric == 'euclidean':
        distmat = euclidean_squared_distance(input1, input2)
    elif metric == 'cosine':
        distmat = cosine_distance(input1, input2)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )

    return distmat.numpy()


def compute_batched_distance_matrix(input1, input2, metric='cosine', batch_size=10):
    """
    Computes the distance matrix in a batched manner to save memory.

    Args:
        input1 (np.ndarray): 2-D array of query features.
        input2 (np.ndarray): 2-D array of database features.
        metric (str): The distance metric to use. Options include 'euclidean', 'cosine', etc.
        batch_size (int): The number of rows from input1 to process at a time.

    Returns:
        np.ndarray: The computed distance matrix.
    """
    # Ensure input is in numpy format for compatibility with cdist
    if isinstance(input1, torch.Tensor):
        input1 = input1.numpy()
    if isinstance(input2, torch.Tensor):
        input2 = input2.numpy()

    num_batches = int(np.ceil(input1.shape[0] / batch_size))
    dist_matrix = []

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, input1.shape[0])
        batch_distances = cdist(input1[start:end], input2, metric=metric)
        dist_matrix.append(batch_distances)

    return np.vstack(dist_matrix)
