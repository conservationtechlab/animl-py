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

    config = Path.cwd() / 'examples' / 'animl.yml'

    animl.from_config(config)

    #export_coco, 
    #export_timelapse, 


    #export.remove_link
    #export.update_labels_from_folders
