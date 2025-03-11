import os
import json
import yaml
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from ultralytics import YOLO

class CustomYOLO:
    """ Custom YOLO class for image detection"""
    def __init__(self, config_path="custom_yolo.yaml", device="cpu"):
        """
        Initializes YOLO with settings from a configuration file.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        model_path = self.config["detector_file"]
        self.device = self.config["device"]
        
        try:
            self.model = YOLO(model_path)  # Load the YOLO model
            self.model.to(self.device)    # Move model to device (CPU/GPU)
        except Exception as e:
            raise ValueError(f"Failed to load the YOLO model from {model_path}: {e}")

    #TODO try to feed it from the manifest
    def detect_batch(self, image_input=None,image_size = None):
        """
        Runs YOLO on a batch of images.
        - Inputs: image_input can be a folder path (str) or a DataFrame.
        - Outputs: a list of dictionaries with image file paths and detections
        """



        if image_input is None:
            image_input = self.config["paths"]["image_folder"]


        if isinstance(image_input, str) and os.path.isdir(image_input):
            image_file_names = [
                os.path.join(image_input, f) for f in os.listdir(image_input)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            print(f'Found {len(image_file_names)} images in {image_input}')
        

        elif isinstance(image_input, pd.DataFrame):
            file_col = self.config["detection"]["file_col"]  # Column that stores file paths
            image_file_names = image_input[file_col].tolist()
            print(f'Loaded {len(image_file_names)} images from DataFrame.')
        elif isinstance(image_input, list):
            image_file_names = image_input
            print(f'Processing {len(image_file_names)} image files from a list.')

        else:
            raise ValueError("Invalid input: image_input must be a folder path, DataFrame, or a list.")

        results = []
        for image_file in tqdm(image_file_names):
            try:
                prediction = self.model.predict(image_file)
                detections = [
                    {
                        "class": int(box.cls.item()),
                        "conf": float(box.conf.item()),
                        "bbox1": float(box.xyxy[0][0].item()), 
                        "bbox2": float(box.xyxy[0][1].item()),  
                        "bbox3": float(box.xyxy[0][2].item()),  ##can be changed to xywh
                        "bbox4": float(box.xyxy[0][3].item())   
                    }
                    for box in prediction[0].boxes
                    if box.conf.item() >= 0.01
                ]

                results.append({"file": image_file, "detections": detections})

            except Exception as e:
                print(f"Detection failed for image {image_file}: {e}")
                print(f"Error occurred at line: {e.__traceback__.tb_lineno}")
    
        return results