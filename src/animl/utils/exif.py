import exiftool
from pathlib import Path


def convert_dict_to_adobe(tag_dict: dict) -> dict:
    """
    Convert a dictionary of EXIF tags to Adobe format.

    Args:
        - tag_dict (dict): dictionary of EXIF tags where keys are tag names and values are the corresponding values.

    Returns:
        - dict: dictionary of EXIF tags in Adobe format.

    Raises:
        - Exception: if the input is not a valid dictionary.
    """
    if not isinstance(tag_dict, dict):
        raise Exception("'tag_dict' must be a dictionary")

    adobe_dict = {}
    for key, value in tag_dict.items():
        adobe_key = f"XMP:Adobe{key}"
        adobe_dict[adobe_key] = value

    return adobe_dict


def edit_exif_tags(filepath: str, tags: dict) -> None:
    """
    Edit the EXIF tags of an image file.

    Args:
        - filepath (str): path to the image file.
        - tags (dict): dictionary of EXIF tags to edit, where keys are tag names and values are the new values.

    Raises:
        - Exception: if the file cannot be opened or the EXIF tags cannot be edited.
    """
    assert isinstance(filepath, str), "'filepath' must be a string"
    assert isinstance(tags, dict), "'tags' must be a dictionary"

    if not Path(filepath).exists():
        raise Exception(f"File '{filepath}' does not exist.")

    try:
        with exiftool.ExifToolHelper() as et:

            # Read current metadata
            metadata = et.get_metadata(filepath)
            
            # Check if Bridge tags exist
            if 'XMP-xmpBJ:Marks' in metadata:
                print("Current Bridge tags:", metadata['XMP-xmpBJ:Marks'])


            # Write new tags
            et.execute(
                b'-XMP-xmpBJ:Marks=new|tags',
                b'image.jpg'
            )


    except Exception as e:
        raise Exception(f"Failed to edit EXIF tags for {filepath}: {e}")