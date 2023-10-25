import pandas as pd
import os


def symlink_species(manifest, linkdir, classes):
    """
    Creates symbolic links of images into species folders

    Args
        - manifest: dataframe containing images and associated predictions
        - linkdir: root directory for species folders
        - classes: full list of class names
    """
    # Create species folders
    table = pd.read_table(classes, sep=" ", index_col=0)
    for i in range(0, len(table.index)):
        directory = str(table['x'].values[i])
        if not os.path.isdir(linkdir + directory):
            os.makedirs(linkdir + directory)

    for i in range(0, len(manifest.index)):
        try:
            os.symlink(manifest.at[i, 'file'],
                       linkdir + manifest.at[i, 'class'] + "/" + os.path.basename(manifest.at[i, 'file']))
        except Exception as e:
            print('File already exists. Exception: {}'.format(e))
            continue


def symlink_MD(manifest, linkdir):
    """
    Creates symbolic links of images into species folders

    Args
        - manifest: dataframe containing images and associated detections
        - linkdir: root directory for species folders
        - classes: full list of class names
    """
    # Create class subfolders
    classes = ["empty", "animal", "human", "vehicle"]
    for i in range(len(classes)):
        if not os.path.isdir(linkdir + classes[i]):
            os.makedirs(linkdir + classes[i])


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
