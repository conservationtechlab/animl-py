import unittest
import time
from pathlib import Path

import animl


@unittest.skip
def md_test():
    start_time = time.time()

    image_dir = Path.cwd() / 'examples' / 'Southwest'

    manifest = animl.build_file_manifest(image_dir, exif=False)

    animl.list_models()

    md_version = 'MDV6-yolov10-e'

    #animl.download_model(animl.MEGADETECTOR[md_version], out_dir='models')
    detector = animl.load_detector('models/' + animl.MD_FILENAMES[md_version], 'yolo')
    #detector = animl.load_detector('models/md_v1000.0.0-sorrel.onnx', 'onnx')

    string = animl.detect(detector,
                          manifest.iloc[2]['filepath'],
                          resize_height=animl.MEGADETECTORv5_SIZE,
                          resize_width=animl.MEGADETECTORv5_SIZE)
    string_parsed = animl.parse_detections(string)

    # Series
    series = animl.detect(detector,
                          manifest.iloc[2],
                          resize_height=animl.MEGADETECTORv5_SIZE,
                          resize_width=animl.MEGADETECTORv5_SIZE)
    series_parsed = animl.parse_detections(series)
    print("Series match:", string_parsed.equals(series_parsed))

    # DataFrame slice
    slice = animl.detect(detector,
                         manifest.iloc[2:3],
                         resize_height=animl.MEGADETECTORv5_SIZE,
                         resize_width=animl.MEGADETECTORv5_SIZE)
    slice_parsed = animl.parse_detections(slice)
    print("Slice match:", string_parsed.equals(slice_parsed))

    # List
    slist = animl.detect(detector,
                         manifest['filepath'].tolist()[2:3],
                         resize_height=animl.MEGADETECTORv5_SIZE,
                         resize_width=animl.MEGADETECTORv5_SIZE)
    slist_parsed = animl.parse_detections(slist)
    print("List match:", string_parsed.equals(slist_parsed))

    gt_path = Path.cwd() / 'tests' / 'GroundTruth' / 'md' / 'md_gt.json'
    md_gt = animl.load_json(gt_path)

    slist_parsed['prediction'] = 'test'
    slist_parsed['confidence'] = 1

    animl.export_megadetector(slist_parsed, "md_export_test.json", prompt=False)

    print(slist_parsed)

    if (string == md_gt['images']):
        print("MD Test Successful!")
    else:
        print("MD Test Failed :(")

    print(f"Test completed in {time.time() - start_time:.2f} seconds")


md_test()
