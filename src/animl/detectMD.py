# modified from CameraTraps/MegaDetector
from tqdm import tqdm
import time
import json
import humanfriendly
import os
import pandas as pd

from detection import run_detector
from detection import run_detector_batch

def chunks_by_number_of_chunks(ls, n):
    for i in range(0, n):
        yield ls[i::n]

        
def detect_MD_batch(model_file, image_file_names, checkpoint_path=None,
                                confidence_threshold=run_detector.DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD,
                                checkpoint_frequency=-1, results=None, n_cores=1, quiet=False, image_size=None):
    """
    Args
    - model_file: str, path to .pb model file
    - image_file_names: list of strings (image filenames), a single image filename, 
                        a folder to recursively search for images in, or a .json file containing
                        a list of images.
    - checkpoint_path: str, path to JSON checkpoint file
    - confidence_threshold: float, only detections above this threshold are returned
    - checkpoint_frequency: int, write results to JSON checkpoint file every N images
    - results: list of dict, existing results loaded from checkpoint
    - n_cores: int, # of CPU cores to use

    Returns
    - results: list of dict, each dict represents detections on one image
    """
    
    if n_cores is None:
        n_cores = 1
    
    if confidence_threshold is None:
        confidence_threshold=run_detector.DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD
        
    if checkpoint_frequency is None:
        checkpoint_frequency = -1
        
    # Handle the case where image_file_names is not yet actually a list
    if isinstance(image_file_names,str):
        
        # Find the images to score; images can be a directory, may need to recurse
        if os.path.isdir(image_file_names):
            image_dir = image_file_names
            image_file_names = run_detector.ImagePathUtils.find_images(image_dir, True)
            print('{} image files found in folder {}'.format(len(image_file_names),image_dir))
            
        # A json list of image paths
        elif os.path.isfile(image_file_names) and image_file_names.endswith('.json'):
            list_file = image_file_names
            with open(list_file) as f:
                image_file_names = json.load(f)
            print('Loaded {} image filenames from list file {}'.format(len(image_file_names),list_file))
            
        elif isinstance(image_file_names,pd.Series):
            pass
            
        # A single image file
        elif os.path.isfile(image_file_names) and run_detector.ImagePathUtils.is_image_file(image_file_names):
            image_file_names = [image_file_names]
            #print('Processing image {}'.format(image_file_names[0]))
            
        else:        
            raise ValueError('image_file_names is a string, but is not a directory, a json ' + \
                             'list (.json), or an image file (png/jpg/jpeg/gif)')
    
    if results is None:
        results = []

    already_processed = set([i['Frame'] for i in results])

    print('GPU available: {}'.format(run_detector.is_gpu_available(model_file)))
    
    if n_cores > 1 and run_detector.is_gpu_available(model_file):
        print('Warning: multiple cores requested, but a GPU is available; parallelization across ' + \
              'GPUs is not currently supported, defaulting to one GPU')
        n_cores = 1

        
    elif n_cores <= 1:

        # Load the detector
        start_time = time.time()
        detector = run_detector.load_detector(model_file)
        elapsed = time.time() - start_time
        print('Loaded model in {}'.format(humanfriendly.format_timespan(elapsed)))

        # Does not count those already processed
        count = 0

        for im_file in tqdm(image_file_names):

            # Will not add additional entries not in the starter checkpoint
            if im_file in already_processed:
                if not quiet:
                    print('Bypassing image {}'.format(im_file))
                continue

            count += 1

            result = run_detector_batch.process_image(im_file, detector, 
                                   confidence_threshold, quiet=quiet, 
                                   image_size=image_size)
            results.append(result)

            # Write a checkpoint if necessary
            if checkpoint_frequency != -1 and count % checkpoint_frequency == 0:
                
                print('Writing a new checkpoint after having processed {} images since '
                      'last restart'.format(count))
                
                assert checkpoint_path is not None                
                
                # Back up any previous checkpoints, to protect against crashes while we're writing
                # the checkpoint file.
                checkpoint_tmp_path = None
                if os.path.isfile(checkpoint_path):
                    checkpoint_tmp_path = checkpoint_path + '_tmp'
                    shutil.copyfile(checkpoint_path,checkpoint_tmp_path)
                    
                # Write the new checkpoint
                with open(checkpoint_path, 'w') as f:
                    json.dump({'images': results}, f, indent=1)
                    
                # Remove the backup checkpoint if it exists
                if checkpoint_tmp_path is not None:
                    os.remove(checkpoint_tmp_path)
                    
            # ...if it's time to make a checkpoint
            
    else:
        
        # When using multiprocessing, let the workers load the model
        detector = model_file

        print('Creating pool with {} cores'.format(n_cores))

        if len(already_processed) > 0:
            print('Warning: when using multiprocessing, all images are reprocessed')

        pool = workerpool(n_cores)

        image_batches = list(chunks_by_number_of_chunks(image_file_names, n_cores))
        results = pool.map(partial(run_detector_batch.process_images, detector=detector,
                                   confidence_threshold=confidence_threshold,image_size=image_size), 
                           image_batches)

        results = list(itertools.chain.from_iterable(results))

    return results
