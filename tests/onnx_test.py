"""
Test pipeline using ONNX models.

"""
import unittest
import time
import shutil
from pathlib import Path
import yaml

import animl


@unittest.skip
def onnx_test():
    start_time = time.time()

    workingdir = Path.cwd() / 'examples' / 'Southwest' / 'Animl-Directory'
    shutil.rmtree(workingdir, ignore_errors=True)

    config = Path.cwd() / 'examples' / 'animl.yml'

    results = animl.from_config(config)

    # export timelapse
    animl.export_timelapse(results, image_dir=workingdir, only_animl=True)

    # export coco
    class_list=animl.load_class_list(Path.cwd() / 'models' / 'sdzwa_southwest_v3_classes.csv')
    animl.export_coco(results, class_list=class_list, out_file=workingdir / 'coco_export.json')

    # export.remove_link
    # export.update_labels_from_folders
    print(f"Test completed in {time.time() - start_time:.2f} seconds")


@unittest.skip
def onnx_gpu_test():
    # test onnx model on gpu
    start_time = time.time()

    workingdir = animl.WorkingDirectory(Path.cwd() / 'examples' / 'Southwest')

    config = Path.cwd() / 'examples' / 'animl.yml'
    cfg = yaml.safe_load(open(config, 'r'))

    allframes = animl.load_data(workingdir.imageframes)

    model_cpu = animl.load_detector('models/md_v1000.0.0-sorrel.pt', model_type="yolo", device='cpu')

    results = animl.detect(model_cpu, allframes,
                           resize_width=960, resize_height=960,
                           batch_size=4, device='cpu')
    detections = animl.parse_detections(results, manifest=allframes)

    model_gpu = animl.load_detector(cfg['detector_file'], model_type="onnx", device='cuda:0')
    results_gpu = animl.detect(model_gpu, allframes,
                               resize_width=960, resize_height=960,
                               batch_size=4, device='cuda:0')
    detections_gpu = animl.parse_detections(results_gpu, manifest=allframes)

    print(detections)
    print(detections_gpu)
    print("GPU matches CPU:", detections.equals(detections_gpu))

    print(f"ONNX GPU Test completed in {time.time() - start_time:.2f} seconds")


onnx_test()
onnx_gpu_test()
