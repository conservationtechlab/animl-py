"""
    Symlink Module

    Provides functions for creating, removing, and updating sorted symlinks.

    @ Kyra Swanson 2023
"""

import os
from shutil import copy2


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
    # Create species folders
    for species in manifest['prediction'].unique():
        os.makedirs(linkdir + species, exist_ok=True)

    # create new column
    manifest['Link'] = linkdir

    for i, row in manifest.iterrows():
        name = row['UniqueName'] if 'UniqueName' in manifest.columns else os.path.basename(row[file_col])
        link = linkdir + row['prediction'] + "/" + name
        manifest.loc[i, 'Link'] = link
        if copy:
            print("Hard copy enabled. This will overwrite existing files.")
            copy2(row[file_col], link)
        else:
            try:
                os.symlink(row[file_col], link)
            except Exception as e:
                print('File already exists. Exception: {}'.format(e))
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
    # Create class subfolders
    classes = ["empty", "animal", "human", "vehicle"]
    for i in range(classes):
        os.makedirs(linkdir + i, exist_ok=True)

    # create new column
    manifest['Link'] = linkdir
    for i, row in manifest.iterrows():
        name = row['UniqueName'] if 'UniqueName' in manifest.columns else os.path.basename(row[file_col])
        link = linkdir + str(row['category']) + "/" + name
        manifest.loc[i, 'Link'] = link
        if copy:
            print("Hard copy enabled. This will overwrite existing files.")
            copy2(row[file_col], link)
        else:
            try:
                os.symlink(row[file_col], link)
            except Exception as e:
                print('File already exists. Exception: {}'.format(e))
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
