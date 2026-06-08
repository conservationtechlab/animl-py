"""
Forked from 
https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/repeat_detection_elimination/repeat_detections_core.py

"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import copy
from pathlib import Path
import cv2
import json
from dataclasses import dataclass

from numpy import save
from tqdm import tqdm
from operator import attrgetter
import fastquadtree.pyqtree as pyqtree

#from animl.model_architecture import MD_LABELS
from animl.utils.general import get_iou
from animl.utils.visualization import plot_box


@dataclass
class RepeatDetectionOptions:
    """
    Options that control the behavior of repeat detection elimination
    """
    # Location to save images for manual review; if empty, don't save any images for review
    outputBase: str = ''
    #: Don't consider detections with confidence lower than this as suspicious
    confidenceMin: float = 0.1
    #: Don't consider detections with confidence higher than this as suspicious
    confidenceMax: float = 1.0

    #: What's the IOU threshold for considering two boxes the same?
    iouThreshold: float = 0.9
    #: How many occurrences of a single location before we declare it suspicious?
    occurrenceThreshold: int = 50

    #: Ignore "suspicious" detections smaller than some size
    minSuspiciousDetectionSize: float = 0.0
    #: Ignore "suspicious" detections larger than some size; these are often animals
    #: taking up the whole image.  This is expressed as a fraction of the image size.
    maxSuspiciousDetectionSize: float = 0.5

    #: A list of category IDs (ints) that we don't want consider as candidate repeat detections.
    #: Typically used to say, e.g., "don't bother analyzing people or vehicles for repeat
    #: detections", which you could do by saying excludeClasses = [2,3].
    excludeClasses: list = []

    #: Include only specific folders, mutually exclusive with [excludeFolders]
    includeFolders: list = []
    #: Exclude specific folders, mutually exclusive with [includeFolders]
    excludeFolders: list = []

    categoryMap: dict = {0: 'empty', 1: 'animal', 2: 'person', 3: 'vehicle'}


def set_detection_options(options, **kwargs):
    """
    Set options for repeat detection elimination.  See RepeatDetectionOptions for details on the options.
    """
    for key, value in kwargs.items():
        if hasattr(options, key):
            setattr(options, key, value)
        else:
            raise ValueError(f'Invalid option {key} for repeat detection elimination')


@dataclass
class IndexedDetection:
    """
    A single detection event on a single image
    """
    #: index of this detection within all detections for this filepath
    i_detection: int = -1
    #: original index from full manifest
    original_id: int = -1
    #: path to the image corresponding to this detection
    filepath: str = ''
    #: [x_min, y_min, width_of_box, height_of_box]
    bbox: list = []
    #: confidence value of this detection
    confidence: float = -1
    #: category ID (not name) of this detection
    category: int = 0


class DetectionLocation:
    """
    A unique-ish detection location, meaningful in the context of one
    directory. All detections within an IoU threshold of self.bbox
    will be stored in IndexedDetection objects.
    """
    def __init__(self, instance, bbox, relative_dir, category, id=None):
        assert isinstance(bbox,list)
        assert isinstance(instance,IndexedDetection)
        assert isinstance(relative_dir,str)
        assert isinstance(category,int)
        #: list of IndexedDetections that match this detection
        self.instances = [instance]
        #: category ID (not name) for this detection
        self.category = category
        #: bbox as x,y,w,h
        self.bbox = bbox
        #: relative folder (i.e., camera name) in which this detectin was found
        self.relativeDir = relative_dir
        #: relative path to the canonical image representing this detection
        self.sampleImageRelativeFileName = ''
        #: list of detections on that canonical image that match this detection
        self.sampleImageDetections = None
        #: ID for this detection; this ID is only guaranteed to be unique within a directory
        self.id = id


def _detection_rect_to_rtree_rect(detection_rect):
    """
    We store detections as x/y/w/h, rtree and pyqtree use l/b/r/t.  Convert from
    our representation to rtree's.
    """
    left = detection_rect[0]
    bottom = detection_rect[1]
    right = detection_rect[0] + detection_rect[2]
    top = detection_rect[1] + detection_rect[3]
    return (left,bottom,right,top)


def _find_matches_in_directory(dir_name_and_rows, options):
    """
    dir_name_and_rows is a tuple of (name,rows).

    "name" is a location name, typically a folder name, though this may be an arbitrary
    location identifier.

    "rows" is a Pandas dataframe with one row per image in this location, with columns:

        * 'file': relative file name
        * 'detections': a list of MD detection objects, i.e. dicts with keys ['category','conf','bbox']
        * 'max_detection_conf': maximum confidence of any detection, in any category

    "rows" can also point to a .csv file, in which case the detection table will be read from that
    .csv file, and results will be written to a .csv file rather than being returned.

    Find all unique detections in this directory.

    Returns a list of DetectionLocation objects.
    """
    # Create a tree to store candidate detections
    candidate_detections_index = pyqtree.Index(bbox=(-0.1,-0.1,1.1,1.1))

    assert len(dir_name_and_rows) == 2, 'find_matches_in_directory: invalid input'
    assert isinstance(dir_name_and_rows[0],str), 'find_matches_in_directory: invalid location name'
    dir_name = dir_name_and_rows[0]
    rows = dir_name_and_rows[1]

    # skip specific folders
    if options.includeFolders is not None:
        assert options.excludeFolders is None, 'Cannot specify include and exclude folder lists'
        if dir_name not in options.includeFolders:
            print('Ignoring folder {}, not in inclusion list'.format(dir_name))
            return []
    if options.excludeFolders is not None:
        assert options.includeFolders is None, 'Cannot specify include and exclude folder lists'
        if dir_name in options.excludeFolders:
            print('Ignoring folder {}, on exclusion list'.format(dir_name))
            return []


    for i_directory_row, row in rows.iterrows():
        original_id = row['original_index']
        filepath = row['filepath']
        if not Path(filepath).is_file():
            continue

        # If detection confidence is outside threshold
        confidence = row['conf']
        assert confidence >= -1.0 and confidence <= 1.0
        if confidence < options.confidenceMin or confidence > options.confidenceMax:
            continue

        category = int(row['category'])
        
        # Optionally exclude some classes from consideration as suspicious
        if (options.excludeClasses is not None) and (len(options.excludeClasses) > 0):
            
            if category in options.excludeClasses:
                continue

        bbox = [row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']]
   
        # Is this detection too big or too small for consideration?
        w, h = bbox[2], bbox[3]
        if (w == 0 or h == 0):
            continue

        area = h * w
        if area < 0:
            print('Warning: negative-area bounding box for file {}'.format(filepath))
            area = abs(area); h = abs(h); w = abs(w)

        assert area >= 0.0 and area <= 1.0, \
            'Illegal bounding box area {} in image {}'.format(area, filepath)

        if area < options.minSuspiciousDetectionSize:
            continue
        if area > options.maxSuspiciousDetectionSize:
            continue

        instance = IndexedDetection(i_detection=i_directory_row,
                                    original_id=original_id,
                                    filepath=row['filepath'], bbox=bbox,
                                    confidence=confidence, category=category)
        b_found_similar_detection = False

        # find candidate detections that overlap with this detection
        rtree_rect = _detection_rect_to_rtree_rect(bbox)
        overlapping_candidate_detections = candidate_detections_index.intersect(rtree_rect)
        overlapping_candidate_detections.sort(key=lambda x: x.id, reverse=False)

        # CALCULATE IOU
        # For each detection in our candidate list
        for _, candidate in enumerate(overlapping_candidate_detections):
            # Don't match across categories
            if (candidate.category != category):
                continue
            try:
                iou = get_iou(bbox, candidate.bbox)
            except Exception as e:
                print(\
                'Warning: IOU computation error on boxes ({},{},{},{}),({},{},{},{}): {}'.\
                    format(bbox[0],bbox[1],bbox[2],bbox[3],
                           candidate.bbox[0],candidate.bbox[1],
                           candidate.bbox[2],candidate.bbox[3], str(e)))
                continue
            # Match
            if iou >= options.iouThreshold:
                b_found_similar_detection = True
                candidate.instances.append(instance)

        # If we found no matches, add this to the candidate list as single node
        if not b_found_similar_detection:
            candidate = DetectionLocation(instance=instance,
                                          bbox=bbox,
                                          relative_dir=dir_name,
                                          category=category,
                                          id=i_directory_row)
            # pyqtree
            candidate_detections_index.insert(item=candidate,bbox=rtree_rect)

    # Get all candidate detections
    candidate_detections = candidate_detections_index.intersect([-100,-100,100,100])

    # For debugging only, it's convenient to have these sorted
    # as if they had never gone into a tree structure.  Typically
    # this is in practice a sort by filepath.
    candidate_detections.sort(key=lambda x: x.id, reverse=False)
    return candidate_detections


def find_repeat_detections(manifest, 
                           options=RepeatDetectionOptions(),
                           manual_review=False,
                           output_dir=None,
                           parallel=False):
    """
    Find detections in a MD results file that occur repeatedly and are likely to be
    rocks/sticks.

    Args:
        manifest (pd.DataFrame): the MD results .json file to analyze
        options (RepeatDetectionOptions, optional): all the interesting options controlling
            this process; see RepeatDetectionOptions for details.
        manual_review (bool, optional): whether to immediately display images for manual review
        output_dir (str, optional): if specified, save images for manual review to this directory;
        parallel (bool, optional): whether to run the process in parallel across directories

    Returns:
        filtered_manifest (pd.DataFrame): a copy of the input manifest with false positives removed
        filterFile (str): path to a .json file containing the suspicious detections
    """
    # save index as column to be able to remove false positives from manifest_filtered later
    manifest = manifest.copy()
    manifest.reset_index(drop=True, inplace=True)
    manifest['original_index'] = manifest.index

    # manifest must have station columm
    assert 'station' in manifest.columns, 'Manifest must have station column'
    manifest_by_stations = manifest.groupby('station')

    dirs_to_search = list(manifest_by_stations.groups.keys())
    num_dirs = len(dirs_to_search)

    all_candidate_detections = [None] * len(dirs_to_search)
    suspicious_detections = [None] * num_dirs
    dir_index_to_name = {}

    # compare number of detections in each directory to the threshold, and mark suspicious detections
    def mark_suspicious(candidate_detections_this_dir):
        suspicious_detections_this_dir = []
        for _, candidate_location in enumerate(candidate_detections_this_dir):
            n_occurrences = len(candidate_location.instances)
            # check against the threshold to determine whether this is a suspicious detection
            if n_occurrences >= options.occurrenceThreshold:
                 suspicious_detections_this_dir.append(candidate_location)

        return suspicious_detections_this_dir
    
    # parallel processing of directories
    if parallel:
        def _process_directory(args):
            i_dir, dir_name, manifest_by_stations, options = args
            rows_this_directory = manifest_by_stations.get_group(dir_name)
            candidate_detections_this_dir = _find_matches_in_directory((dir_name, rows_this_directory), options)
            suspicious = sorted(mark_suspicious(candidate_detections_this_dir),
                                key=lambda x: (x.bbox[0]) + (x.bbox[2] / 2.0))
            return i_dir, dir_name, candidate_detections_this_dir, suspicious

        args_list = [(i_dir, dir_name, manifest_by_stations, options)
                     for i_dir, dir_name in enumerate(dirs_to_search)]

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(_process_directory, args): args[0] for args in args_list}
            for future in tqdm(as_completed(futures), total=len(dirs_to_search)):
                i_dir, dir_name, candidate_detections_this_dir, suspicious = future.result()
                dir_index_to_name[i_dir] = dir_name
                all_candidate_detections[i_dir] = candidate_detections_this_dir
                suspicious_detections[i_dir] = suspicious
                print(f'Found {len(suspicious_detections[i_dir])} suspicious detections in station {dir_name}')
    
    # sequential
    else:
        # get all candidate detections for each directory
        for i_dir, dir_name in tqdm(enumerate(dirs_to_search)):
            dir_index_to_name[i_dir] = dir_name
            rows_this_directory = manifest_by_stations.get_group(dir_name)
            candidate_detections_this_dir = _find_matches_in_directory((dir_name, rows_this_directory), options)
            all_candidate_detections[i_dir] = candidate_detections_this_dir
            # mark suspicious detections
            suspicious_detections[i_dir] = sorted(mark_suspicious(candidate_detections_this_dir),
                                                key=lambda x: ((x.bbox[0]) + (x.bbox[2]/2.0) ))
            print(f'Found {len(suspicious_detections[i_dir])} suspicious detections in station {dirs_to_search[i_dir]}')


    # if output directory is specified, save images for manual review of suspicious detections
    if output_dir is not None and output_dir != '':
        filtering_dir = Path(output_dir) / 'filtering'
        print(f'Creating filtering folder: {filtering_dir}/')
        os.makedirs(filtering_dir, exist_ok=True)

        detection_index_file_name = filtering_dir / 'detectionIndex.json'
        # Prepare the data we're going to write to the detection index file
        detection_info = {
            'suspicious_detections': suspicious_detections,
            'dir_index_to_name': dir_index_to_name,
            'options': options
        }
        # file_manifest.save_json(detection_info, detection_index_file_name)
    else:
        filtering_dir = None

    # sets to keep track of true positives and false positives, as marked by the user during manual review
    true_positives = set()
    false_positives = set()

    # if manual review is enabled, display images for review and allow user to mark true positives and false positives
    if manual_review:
        for i_dir, suspicious_detections_this_dir in enumerate(tqdm(suspicious_detections)):
            for i_detection, detection in enumerate(suspicious_detections_this_dir):
                # Sort instances in descending order by confidence
                detection.instances.sort(key=attrgetter('confidence'),reverse=True)

                # Choose the highest-confidence index
                instance = detection.instances[0]
                filepath = instance.filepath
                # get all detections for that image
                all_detections = manifest[manifest['filepath'] == filepath]
                # mark all detections as non-suspicious by default
                all_detections['category'] = 0 
                # mark the suspicious detection as category 0
                all_detections.loc[instance.original_id, 'category'] = 1

                # plot the image with the suspicious detection highlighted, and all other detections in gray
                im = plot_box(all_detections, colors={"0": (192, 192, 192), "1": (255, 0, 0)}, return_img=True)

                 # output path to save image with bbox
                if filtering_dir is not None:
                    output_filename = 'dir{:0>4d}_det{:0>4d}_n{:0>4d}.jpg'.format(
                        i_dir, i_detection, len(detection.instances))
                    output_path = filtering_dir / output_filename
                    cv2.imwrite(output_path, im)

                # display the image for review
                cv2.imshow('Suspicious Detection: ' + filepath, im)
                print(f'Displaying {output_filename} with {len(detection.instances)} instances. \
                      Press T to mark as a true positive, F to mark as a false positive, or Esc to exit.')
                while True:
                    key = cv2.waitKey(0) & 0xFF

                    if key == ord("t"):
                        # Mark this detection as a true positive
                        true_positives = true_positives.union(set(match.original_id for match in detection_location.instances))
                        break
                    elif key == ord("f"):
                        # Mark all instances of this detection as false positives
                        false_positives = false_positives.union(set(match.original_id for match in detection_location.instances))
                        break
                    elif key == 27:  # Esc
                        print("Pressed Esc, image ignored.")
                        break

                cv2.destroyAllWindows()

    # if manual review is not enabled, get all original indices of suspicious detections and mark them as false positives
    else:
        print('Manual review not enabled, skipping image display and manual marking of true positives and false positives.')
        # get all original indices of suspicious detections
        for directory in suspicious_detections:
            for detection_location in directory:
                false_positives = false_positives.union(set(match.original_id for match in detection_location.instances))


    # mark false positives in manifest
    manifest_marked = copy.deepcopy(manifest)

    fp_category_id = max(options.categoryMap.keys()) + 1

    manifest_marked.loc[manifest_marked['original_index'].isin(false_positives), 'category'] = fp_category_id
    manifest_marked.loc[manifest_marked['original_index'].isin(false_positives), 'category_label'] = 'false_positive'

    return manifest_marked, true_positives, false_positives


def remove_false_positives(manifest_marked):
    """
    Remove false positives from the manifest, based on the 'false_positive' column added by find_repeat_detections.

    Args:
        manifest_marked (pd.DataFrame): the manifest with a 'false_positive' column indicating which detections are false positives

    Returns:
        filtered_manifest (pd.DataFrame): a copy of the input manifest with false positives removed
    """
    filtered_manifest = manifest_marked[manifest_marked['category_label'] != 'false_positive'].copy()
    filtered_manifest.drop(columns=['category_label'], inplace=True)
    return filtered_manifest

