import unittest
import time
from pathlib import Path


import animl

# @unittest.skip
def md_test():
    start_time = time.time()

    image_dir = Path.cwd() / 'examples' / 'Southwest'

    manifest = animl.build_file_manifest(image_dir, exif=False)

    mdv5 = Path.cwd() / 'models/md_v5a.0.0.pt'

    detector = animl.load_detector(mdv5, 'mdv5')

    string = animl.detect(detector,
                          manifest.iloc[2]['filepath'],
                          resize_height=animl.MEGADETECTORv5_SIZE,
                          resize_width=animl.MEGADETECTORv5_SIZE)
    string_parsed = animl.parse_detections(string)

    

    series = animl.detect(detector,
                                manifest.iloc[2],
                                resize_height=animl.MEGADETECTORv5_SIZE,
                                resize_width=animl.MEGADETECTORv5_SIZE)
    series_parsed = animl.parse_detections(series)
    
    print("Series match:", string_parsed.equals(series_parsed))


    slice = animl.detect(detector,
                         manifest.iloc[2:3],
                         resize_height=animl.MEGADETECTORv5_SIZE,
                         resize_width=animl.MEGADETECTORv5_SIZE)
    slice_parsed = animl.parse_detections(slice)

    print("Slice match:", string_parsed.equals(slice_parsed))

    slist = animl.detect(detector,
                         manifest['filepath'].tolist()[2:3],
                          resize_height=animl.MEGADETECTORv5_SIZE,
                          resize_width=animl.MEGADETECTORv5_SIZE)
    slist_parsed = animl.parse_detections(slist)
    
    print("List match:", string_parsed.equals(slist_parsed))

    gt_path = Path.cwd() / 'tests' / 'GroundTruth' / 'md' / 'md_gt.json'
    md_gt = animl.load_json(gt_path)

    # animl.export_megadetector()

    if (string == md_gt['images']):
        print("MD Test Successful!")
    
    print(f"Test completed in {time.time() - start_time:.2f} seconds")




md_test()