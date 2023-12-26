import os


def symlink_species(manifest, linkdir, file_col="FilePath"):
    """
    Creates symbolic links of images into species folders

    Args
        - manifest: dataframe containing images and associated predictions
        - linkdir: root directory for species folders
        - classes: full list of class names
    """
    # Create species folders
    for species in manifest['prediction'].unique():
        os.makedirs(linkdir + species, exist_ok=True)

    manifest['Symlink'] = linkdir
    for i, row in manifest.iterrows():
        link = linkdir + row['prediction'] + "/" + os.path.basename(row[file_col])
        manifest.loc[i, 'Symlink'] = link
        try:
            os.symlink(row[file_col], link)
        except Exception as e:
            print('File already exists. Exception: {}'.format(e))
            continue

    return manifest


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
