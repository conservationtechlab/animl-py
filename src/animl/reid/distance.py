"""
Distance metrics and pairwise distance computation functions.

This module provides functions to compute distance matrices between embidding vectors
using different metrics such as Euclidean and Cosine distances. It also includes a
batched computation function to handle large datasets efficiently.

Original script from WildMe

"""
import numpy as np
from scipy.spatial.distance import cdist

import torch
import torch.nn.functional as F


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
