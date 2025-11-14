"""
Test custom detector

"""
import unittest
import time
from pathlib import Path
import pandas as pd

from animl import file_management, detection, model_architecture, visualization


@unittest.skip
def buow_test():
    start_time = time.time()
    # get files
    image_dir = Path.cwd() / 'examples' / 'BUOW'
    detector = Path.cwd() / 'models/mani_buow_2025.pt'
    files = file_management.build_file_manifest(image_dir, exif=False)

    activetimes = file_management.active_times(files, depth=1)
    print(activetimes)

    detector = detection.load_detector(detector, "YOLO")

    md_results = detection.detect(detector,
                                  files,
                                  model_architecture.MEGADETECTORv5_SIZE,
                                  model_architecture.MEGADETECTORv5_SIZE,
                                  letterbox=False,
                                  file_col="filepath",
                                  batch_size=4,
                                  num_workers=4,
                                  confidence_threshold=0.1)

    detections = detection.parse_detections(md_results, manifest=files)

    gt_path = Path.cwd() / 'tests' / 'GroundTruth' / 'buow' / 'buow_detections.csv'
    gt_manifest = pd.read_csv(gt_path)

    visualization.plot_all_bounding_boxes(detections, 'buow_boxes/', file_col='filepath', min_conf=0.1,
                                          label_col='category', show_confidence=True,
                                          detector_labels={'1': 'adult', '2': 'juvenile'})

    try:
        detections.equals(gt_manifest)
        print("BUOW Detection Test Passed!")
    except ValueError:
        print("filepath columns do not match. Test Failure :(")
        print(detections.compare(gt_manifest))
        exit(1)

    print(f"Test completed in {time.time() - start_time:.2f} seconds")


buow_test()
