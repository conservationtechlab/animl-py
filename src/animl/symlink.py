"""
    Symlink Module

    Provides functions for creating, removing, and updating sorted symlinks.

    @ Kyra Swanson 2023
"""

import os
from pathlib import Path


def symlink_species(manifest, linkdir, file_col="FilePath", copy=False):
    """
    Creates symbolic links of images into species folders

    Args
        - manifest (DataFrame): dataframe containing images and associated predictions
        - linkdir (str): root directory for species folders
        - file_col (str): column containing source paths
        - copy (bool): if true, hard copy

    Returns
        copy of manifest with link path column
    """
    linkdir = Path(linkdir)
    # Create species folders
    for species in manifest['prediction'].unique():
        path = linkdir / Path(species)
        path.mkdir(exist_ok=True)

    # create new column
    manifest['Link'] = linkdir

    for i, row in manifest.iterrows():
        name = row['UniqueName'] if 'UniqueName' in manifest.columns else os.path.basename(row[file_col])
        link = linkdir / Path(row['prediction']) / Path(name)
        manifest.loc[i, 'Link'] = str(link)
        if copy:
            print("Hard copy enabled. This will overwrite existing files.")

            link.hardlink_to(row[file_col])
        else:
            try:
                link.symlink_to(row[file_col])
            except Exception as e:
                print('Exception: {}'.format(e))
                continue

    return manifest


def symlink_MD(manifest, linkdir, file_col="file", copy=False):
    """
    Creates symbolic links of images into species folders

    Args
        - manifest (DataFrame): dataframe containing images and associated predictions
        - linkdir (str): root directory for species folders
        - file_col (str): column containing source paths
        - copy (bool): if true, hard copy

    Returns
        copy of manifest with link path column
    """
    linkdir = Path(linkdir)
    # Create class subfolders
    classes = ["empty", "animal", "human", "vehicle"]
    for i in range(classes):
        path = linkdir / Path(classes)
        path.mkdir(exist_ok=True)

    # create new column
    manifest['Link'] = linkdir
    for i, row in manifest.iterrows():
        name = row['UniqueName'] if 'UniqueName' in manifest.columns else os.path.basename(row[file_col])
        link = linkdir / Path(row['category']) / Path(name)
        manifest.loc[i, 'Link'] = str(link)
        if copy:
            print("Hard copy enabled. This will overwrite existing files.")
            link.hardlink_to(row[file_col])
        else:
            try:
                link.symlink_to(row[file_col])
            except Exception as e:
                print('Exception: {}'.format(e))
                continue

    return manifest


def remove_symlink(manifest):
    """
    Deletes symbolic links of images

    Args
        - manifest: dataframe containing images and associated predictions
        - linkdir: root directory for species folders
        - classes: full list of class names
    """


def update_labels(manifest, linkdir):
    """
    Update manifest after human review of symlink directories

    Args
        - manifest: dataframe containing images and associated predictions
        - linkdir: root directory for species folders
        - classes: full list of class names

    Return
        - manifest: dataframe with updated
    """
