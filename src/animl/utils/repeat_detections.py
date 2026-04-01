"""
Forked from 
https://github.com/agentmorris/MegaDetector/blob/main/megadetector/postprocessing/repeat_detection_elimination/repeat_detections_core.py

"""
import os
import copy
from pathlib import Path
import cv2
import json

from tqdm import tqdm
from operator import attrgetter
import fastquadtree.pyqtree as pyqtree

from animl.utils.general import get_iou
from animl.utils.visualization import plot_box


class RepeatDetectionOptions:
    """
    Options that control the behavior of repeat detection elimination
    """
    def __init__(self):
        self.outputBase = ''

        #: Don't consider detections with confidence lower than this as suspicious
        self.confidenceMin = 0.1
        #: Don't consider detections with confidence higher than this as suspicious
        self.confidenceMax = 1.0

        #: What's the IOU threshold for considering two boxes the same?
        self.iouThreshold = 0.9
        #: How many occurrences of a single location before we declare it suspicious?
        self.occurrenceThreshold = 10

        #: Ignore "suspicious" detections smaller than some size
        self.minSuspiciousDetectionSize = 0.0
        #: Ignore "suspicious" detections larger than some size; these are often animals
        #: taking up the whole image.  This is expressed as a fraction of the image size.
        self.maxSuspiciousDetectionSize = 0.5

        #: A list of category IDs (ints) that we don't want consider as candidate repeat detections.
        #: Typically used to say, e.g., "don't bother analyzing people or vehicles for repeat
        #: detections", which you could do by saying excludeClasses = [2,3].
        self.excludeClasses = []

        #: Include only specific folders, mutually exclusive with [excludeFolders]
        self.includeFolders = None
        #: Exclude specific folders, mutually exclusive with [includeFolders]
        self.excludeFolders = None

        #: Should we write the folder of images used to manually review repeat detections?
        self.saveImages = True

        #: Optionally show *other* detections (i.e., detections other than the
        #: one the user is evaluating), typically in a light gray.
        self.bRenderOtherDetections = False
        self.otherDetectionsColors = [(105,105,105,100)]


class RepeatDetectionResults:
    """
    The results of an entire repeat detection analysis
    """
    def __init__(self):
        # original manifest, with an additional column for the original index in the manifest
        self.manifest = None

        #: An array of length nDirs, where each element is a list of DetectionLocation
        #: objects for that directory that have been flagged as suspicious
        self.suspicious_detections = None

        #: A mapping from directory index to directory name, where the directory index 
        #: is the index of the directory in the suspicious_detections array
        self.dir_index_to_name = None

        #: The data table after modification
        self.manifest_filtered = None

        #: The location of the .json file written with information about the RDE
        #: review images (typically detectionIndex.json)
        self.filterFile = None


class IndexedDetection:
    """
    A single detection event on a single image
    """
    def __init__(self, i_detection=-1, original_id=-1, filepath='', bbox=None, confidence=-1, category=0):

        if bbox is None:
            bbox = []
        assert isinstance(i_detection,int)
        assert isinstance(filepath,str)
        assert isinstance(bbox,list)
        assert isinstance(category,int)

        #: index of this detection within all detections for this filepath
        self.i_detection = i_detection
        #: original index from full manifest
        self.original_id = original_id
        #: path to the image corresponding to this detection
        self.filepath = filepath
        #: [x_min, y_min, width_of_box, height_of_box]
        self.bbox = bbox
        #: confidence value of this detection
        self.confidence = confidence
        #: category ID (not name) of this detection
        self.category = category


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


def _render_sample_image_for_detection(detection, filtering_dir):
    """
    Render a sample image for one unique detection, possibly containing lightly-colored
    high-confidence detections from elsewhere in the sample image.

    "detections" is a DetectionLocation object.

    Depends on having already sorted instances within this detection by confidence, and
    having already generated an output file name for this sample image.
    """
    # get original filepath and output path
    output_relative_path = detection.sampleImageOutputPath
    assert len(output_relative_path) > 0
    output_full_path = os.path.join(filtering_dir, output_relative_path)

    im = plot_box(detection.sampleImageDetections, return_img=True)
    cv2.imwrite(output_full_path, im)



def find_repeat_detections(manifest, 
                           options=RepeatDetectionOptions()):
    """
    Find detections in a MD results file that occur repeatedly and are likely to be
    rocks/sticks.

    Args:
        input_filename (str): the MD results .json file to analyze
        out_file (str, optional): the filename to which we should write results
            with repeat detections removed, typically set to None during the first
            part of the RDE process.
        options (RepeatDetectionOptions, optional): all the interesting options controlling
            this process; see RepeatDetectionOptions for details.

    Returns:
        RepeatDetectionResults: results of the RDE process; see RepeatDetectionResults
        for details.
    """
    manifest.reset_index(drop=True)
    manifest['original_index'] = manifest.index

    # manifest must have station columm
    assert 'station' in manifest.columns, 'Manifest must have station column'
    manifest_by_stations = manifest.groupby('station')

    # setup return struct
    to_return = RepeatDetectionResults()
    to_return.manifest = manifest

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

    # get all candidate detections for each directory
    for i_dir, dir_name in tqdm(enumerate(dirs_to_search)):
        dir_index_to_name[i_dir] = dir_name
        rows_this_directory = manifest_by_stations.get_group(dir_name)
        print(f'Processing dir {i_dir} of {len(dirs_to_search)}: {dir_name}')
        candidate_detections_this_dir = _find_matches_in_directory((dir_name, rows_this_directory), options)
        all_candidate_detections[i_dir] = candidate_detections_this_dir
        # mark suspicious detections
        suspicious_detections[i_dir] = sorted(mark_suspicious(candidate_detections_this_dir),
                                              key=lambda x: ((x.bbox[0]) + (x.bbox[2]/2.0) ))
        print(f'Found {len(suspicious_detections[i_dir])} suspicious detections in station {dirs_to_search[i_dir]}')

    to_return.suspicious_detections = suspicious_detections
    to_return.dir_index_to_name = dir_index_to_name

    ##%% Save images for manual review
    if options.saveImages:
        filtering_dir = os.path.join(options.outputBase, 'filtering')
        print(f'Creating filtering folder: {filtering_dir}/')
        os.makedirs(filtering_dir, exist_ok=True)

        all_suspicious_detections = []
        
        for i_dir, suspicious_detections_this_dir in enumerate(tqdm(suspicious_detections)):
            for i_detection, detection in enumerate(suspicious_detections_this_dir):
                # Sort instances in descending order by confidence
                detection.instances.sort(key=attrgetter('confidence'),reverse=True)

                # Choose the highest-confidence index
                instance = detection.instances[0]
                relative_path = instance.filename

                # output path to save image with bbox
                detection.sampleImageOutputPath = 'dir{:0>4d}_det{:0>4d}_n{:0>4d}.jpg'.format(
                    i_dir, i_detection, len(detection.instances))

                # get all detections for that image
                print(manifest[manifest['filepath'] == relative_path])
                detection.sampleImageDetections = manifest[manifest['filepath'] == relative_path]

                all_suspicious_detections.append(detection)

        # Serial loop over detections
        for detection in all_suspicious_detections:
            _render_sample_image_for_detection(detection, filtering_dir, options)
            # Clear the sample image detections to save memory, since we won't need them anymore
            detection.sampleImageDetections = None                                

    # Write out the detection index
    detection_index_file_name = os.path.join(filtering_dir, 'detectionIndex.json')
    # Prepare the data we're going to write to the detection index file
    detection_info = {}
    detection_info['suspicious_detections'] = suspicious_detections
    detection_info['dir_index_to_name'] = dir_index_to_name
    detection_info['options'] = options
    with open(detection_index_file_name, 'w') as f:
        json.dump(detection_info, f)
    to_return.filterFile = detection_index_file_name

    # remove false positives from manifest_filtered
    manifest_filtered = copy.deepcopy(manifest)
    false_positives = set()
    for directory in suspicious_detections:
        for detection_location in directory:
            false_positives = false_positives.union(set(match.original_id for match in detection_location.instances))

    to_return.manifest_filtered = manifest_filtered[~manifest_filtered['original_index'].isin(false_positives)].reset_index(drop=True)

    return to_return
