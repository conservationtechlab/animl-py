"""
Code to run Miew_ID and other re-identification models

(https://github.com/WildMeOrg/wbia-plugin-miew-id)

"""
from typing import Optional
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

from animl.reid.miewid import MiewIdNet, MIEWID_SIZE
from animl.utils.general import get_device
from animl.generator import manifest_dataloader
from torchvision.transforms import Compose, Normalize


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
    if device is None:
        device = get_device()
    print(f'Sending model to {device}')
    weights = torch.load(file_path, weights_only=True)
    miew = MiewIdNet(device=device)
    miew.to(device)
    miew.device = device
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
    if device is None:
        device = get_device()

    if not {file_col}.issubset(manifest.columns):
        raise ValueError(f"DataFrame must contain '{file_col}' column.")

    output = []
    if isinstance(manifest, pd.DataFrame):
        transform = Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

        dataloader = manifest_dataloader(manifest, batch_size=batch_size, num_workers=num_workers,
                                         file_col=file_col, crop=True, normalize=True,
                                         resize_width=MIEWID_SIZE,
                                         resize_height=MIEWID_SIZE,
                                         transform=transform)
        with torch.no_grad():
            for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                img = batch[0]
                emb = miew_model.extract_feat(img.to(device))
                output.extend(emb.detach().cpu().numpy())
        output = np.vstack(output)
    return output
