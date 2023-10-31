# modified from CameraTraps/MegaDetector
from tqdm import tqdm
import json
import os
import pandas as pd
from shutil import copyfile
from PIL import Image
import torch
from animl.file_management import build_file_manifest

FAILURE_INFER = 'Failure inference'
FAILURE_IMAGE_OPEN = 'Failure image access'

def chunks_by_number_of_chunks(ls, n):
    for i in range(0, n):
        yield ls[i::n]


def load_model(model_path):
    if torch.cuda.is_available():
        print("GPU available")
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
    checkpoint = torch.load(model_path, map_location=device)

    for m in checkpoint['model'].modules():
        t = type(m)
        if t is torch.nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None

    model = checkpoint['model'].float().fuse().eval()  # FP32 model
    return model, device


def process_image(im_file, detector, confidence_threshold, image=None, 
                  quiet=False, image_size=None, skip_image_resizing=False):
    """
    From AgentMorris/MegaDetector
    Runs MegaDetector on a single image file.

    Args
    - im_file: str, path to image file
    - detector: loaded model
    - confidence_threshold: float, only detections above this threshold are returned
    - image: previously-loaded image, if available
    - skip_image_resizing: whether to skip internal image resizing and rely on external resizing

    Returns:
    - result: dict representing detections on one image
        see the 'images' key in 
        https://github.com/agentmorris/MegaDetector/tree/master/api/batch_processing#batch-processing-api-output-format
    """
    
    if not quiet:
        print('Processing image {}'.format(im_file))
    
    if image is None:
        try:
            image = Image.open(im_file).convert(mode='RGB')
            image.load()
        except Exception as e:
            if not quiet:
                print('Image {} cannot be loaded. Exception: {}'.format(im_file, e))
            result = {
                'file': im_file,
                'failure': FAILURE_IMAGE_OPEN
            }
            return result

    try:
        result = detector.generate_detections_one_image(image, im_file, 
                                                        confidence_threshold=confidence_threshold,
                                                        image_size=image_size, 
                                                        skip_image_resizing=skip_image_resizing)
    except Exception as e:
        if not quiet:
            print('Image {} cannot be processed. Exception: {}'.format(im_file, e))
        result = {
            'file': im_file,
            'failure': FAILURE_INFER
        }
        return result


    return result


def detect_MD_batch(detector, image_file_names, checkpoint_path=None, checkpoint_frequency=-1,
                    confidence_threshold=0.005, results=None, quiet=False, image_size=None):
    """
    Args
        - detector: preloaded md model
        - image_file_names: list of strings (image filenames), a single image filename,
                            a folder to recursively search for images in, or a .json file containing
                            a list of images.
        - checkpoint_path: str, path to JSON checkpoint file
        - confidence_threshold: float, only detections above this threshold are returned
        - checkpoint_frequency: int, write results to JSON checkpoint file every N images
        - results: list of dict, existing results loaded from checkpoint

    Returns
        - results: list of dict, each dict represents detections on one image
    """
    if confidence_threshold is None:
        confidence_threshold = 0.005 #Defult from MegaDetector

    if checkpoint_frequency is None:
        checkpoint_frequency = -1

    # Handle the case where image_file_names is not yet actually a list
    if isinstance(image_file_names, str):
        # Find the images to score; images can be a directory, may need to recurse
        if os.path.isdir(image_file_names):
            image_dir = image_file_names
            image_file_names = build_file_manifest(image_dir, True)
            print('{} image files found in folder {}'.format(len(image_file_names), image_dir))

        # A json list of image paths
        elif os.path.isfile(image_file_names) and image_file_names.endswith('.json'):
            list_file = image_file_names
            with open(list_file) as f:
                image_file_names = json.load(f)
            print('Loaded {} image filenames from list file {}'.format(len(image_file_names), list_file))

        # DatsFrme, expected input
        elif isinstance(image_file_names, pd.Series):
            pass

        # A single image file
        elif os.path.isfile(image_file_names):
            image_file_names = [image_file_names]
            # print('Processing image {}'.format(image_file_names[0]))

        else:
            raise ValueError('image_file_names is a string, but is not a directory, a json ' +
                             'list (.json), or an image file (png/jpg/jpeg/gif)')

    if results is None:
        results = []

    already_processed = set([i['Frame'] for i in results])

    count = 0
    for im_file in tqdm(image_file_names):

        # Will not add additional entries not in the starter checkpoint
        if im_file in already_processed:
            if not quiet:
                print('Bypassing image {}'.format(im_file))
            continue

        count += 1

        result = process_image(im_file, detector,
                               confidence_threshold, quiet=quiet,
                               image_size=image_size)
        results.append(result)

        # Write a checkpoint if necessary
        if checkpoint_frequency != -1 and count % checkpoint_frequency == 0:

            print('Writing a new checkpoint after having processed {} images since last restart'.format(count))

            assert checkpoint_path is not None
            # Back up any previous checkpoints, to protect against crashes while we're writing
            # the checkpoint file.
            checkpoint_tmp_path = None
            if os.path.isfile(checkpoint_path):
                checkpoint_tmp_path = checkpoint_path + '_tmp'
                copyfile(checkpoint_path, checkpoint_tmp_path)

            # Write the new checkpoint
            with open(checkpoint_path, 'w') as f:
                json.dump({'images': results}, f, indent=1)

            # Remove the backup checkpoint if it exists
            if checkpoint_tmp_path is not None:
                os.remove(checkpoint_tmp_path)

    return results
