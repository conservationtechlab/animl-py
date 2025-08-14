import time
from pathlib import Path

from animl import file_management, detection, visualization

start_time = time.time()

print(Path.cwd())
image_dir = Path.cwd() / 'examples' / 'BUOW'
detector = Path.cwd() / 'models/mani_buow_2025.pt'

# get files
files = file_management.build_file_manifest(image_dir, exif=False)

detector = detection.load_detector(detector, "YOLO")

md_results = detection.detect(detector,
                              files,
                              file_col="Frame",
                              batch_size=4,
                              num_workers=4,
                              confidence_threshold=0.5)

detections = detection.parse_detections(md_results, manifest=files)

print(detections)

# Plot boxes
visualization.plot_all_bounding_boxes(detections, 'test/', file_col='Frame', prediction=False)
