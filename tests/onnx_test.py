import unittest
import time
import shutil
from pathlib import Path
import pandas as pd


import animl

@unittest.skip
def onnx_test():
    # test onnx model
    # test sequence classification
    start_time = time.time()

    workingdir = Path.cwd() / 'examples' / 'Southwest' / 'Animl-Directory'
    shutil.rmtree(workingdir, ignore_errors=True)

    config = Path.cwd() / 'examples' / 'animl.yml'

    results = animl.from_config(config)
    print(results)

    #export_coco, 
    #export_timelapse, 


    #export.remove_link
    #export.update_labels_from_folders
    print(f"Test completed in {time.time() - start_time:.2f} seconds")

onnx_test()