"""
Test custom detector

"""
import unittest
import time
from pathlib import Path

from animl import file_management, detection, visualization


# @unittest.skip
def buow_test():
    start_time = time.time()
    # get files
    print(Path.cwd())
    image_dir = Path.cwd() / 'examples' / 'BUOW'
    detector = Path.cwd() / 'models/mani_buow_2025.pt'
    files = file_management.build_file_manifest(image_dir, exif=False)

    detector = detection.load_detector(detector, "YOLO")

    md_results = detection.detect(detector,
                                  files,
                                  detection.MEGADETECTORv5_SIZE,
                                  detection.MEGADETECTORv5_SIZE,
                                  letterbox=False,
                                  file_col="Frame",
                                  batch_size=4,
                                  num_workers=4,
                                  confidence_threshold=0.1)
    detections = detection.parse_detections(md_results, manifest=files)

    print(detections)

    visualization.plot_all_bounding_boxes(detections, 'buow_boxes/', file_col='Frame', min_conf=0.1, prediction=False)
    print(f"Test completed in {time.time() - start_time:.2f} seconds")


buow_test()
