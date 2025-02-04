from ultralytics import YOLO
import os
import json
from tqdm import tqdm
import pandas as pd
from pathlib import Path

class OwlYOLO:
    def __init__(self, model_path= Path(__file__).parent.parent.parent / "models/best.pt", device="cpu"):
        """
        Initializes the YOLO model with the given model path.
        Args:
            model_path (str): Path to the YOLO model file.
            device (str): The device to run the model on ("cpu" or "cuda").
        """
        self.device = device
        try:
            self.model = YOLO(model_path)  # Load the model
            self.model.to(self.device)    # Set the device
        except Exception as e:
            raise ValueError(f"Failed to load the YOLO model from {model_path}: {e}")

    def detect_batch(self, image_file_names, checkpoint_path=None, checkpoint_frequency=-1,
                     confidence_threshold=0.1, quiet=True, image_size=None, file_col='Frame'):
        """
        Runs a YOLO model on a batch of image files.
        """
        # Handle confidence threshold and checkpoint defaults
        if confidence_threshold is None:
            confidence_threshold = 0.005

        if checkpoint_frequency is None:
            checkpoint_frequency = -1

        # Handle different input types for image_file_names
        if isinstance(image_file_names, str):
            if os.path.isdir(image_file_names):
                image_file_names = [
                    os.path.join(image_file_names, f) for f in os.listdir(image_file_names)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))
                ]
                print(f'{len(image_file_names)} image files found in folder {image_file_names}')
            elif os.path.isfile(image_file_names) and image_file_names.endswith('.json'):
                with open(image_file_names) as f:
                    image_file_names = json.load(f)
                print(f'Loaded {len(image_file_names)} image filenames from JSON file {image_file_names}')
            elif os.path.isfile(image_file_names):
                image_file_names = [image_file_names]
            else:
                raise ValueError('image_file_names is a string, but not a directory, JSON file, or image file.')
        elif isinstance(image_file_names, pd.DataFrame):
            image_file_names = image_file_names[file_col].tolist()
        elif isinstance(image_file_names, pd.Series):
            image_file_names = image_file_names.tolist()
        elif not isinstance(image_file_names, list):
            raise ValueError('image_file_names is not a recognized object.')

        # Load checkpoint if it exists
        if checkpoint_path and os.path.isfile(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data.get('images', [])
        else:
            results = []

        # Remove already-processed images
        already_processed = set(result['file'] for result in results)
        image_file_names = [f for f in image_file_names if f not in already_processed]

        count = 0
        for image_file in tqdm(image_file_names, disable=quiet):
            try:
                # Run YOLO detection on the image
                prediction = self.model.predict(image_file, verbose=not quiet)
                if not prediction or not prediction[0].boxes:
                    results.append({'file': image_file, 'detections': []})
                    continue
                detections = [
                    {
                        'class': int(box.cls.item()),
                        'conf': float(box.conf.item()),
                        'bbox1': float(box.xywh[0][0].item()),
                        'bbox2': float(box.xywh[0][1].item()),
                        'bbox3': float(box.xywh[0][2].item()),
                        'bbox4': float(box.xywh[0][3].item())
                        
                    }
                    for box in prediction[0].boxes
                    if box.conf.item() >= confidence_threshold
                ]

                # Append result
                results.append({'file': image_file, 'detections': detections})

                count += 1

                # Write checkpoint if necessary
                if checkpoint_frequency > 0 and count % checkpoint_frequency == 0:
                    print(f'Writing a checkpoint after processing {count} images.')
                    temp_checkpoint_path = checkpoint_path + '_tmp' if checkpoint_path else None
                    if checkpoint_path and os.path.isfile(checkpoint_path):
                        os.rename(checkpoint_path, temp_checkpoint_path)
                    with open(checkpoint_path, 'w') as f:
                        json.dump({'images': results}, f, indent=1)
                    if temp_checkpoint_path:
                        os.remove(temp_checkpoint_path)

            except Exception as e:
                print(f'Detection failed for image {image_file}: {e}') 
                print('error occured at line:' + str(e.__traceback__.tb_lineno))

        return results

    def predict(self, image_file, confidence_threshold=0.1, quiet=True):
        """
        Runs a single-image prediction using the YOLO model.
        """
        try:
            prediction = self.model.predict(image_file, verbose=not quiet)
            detections = [
                {
                    'class': int(box.cls),
                    'conf': float(box.conf),
                    'bbox': [float(coord) for coord in box.xywh]
                }
                for box in prediction[0].boxes
                if box.conf >= confidence_threshold
            ]
            return {'file': image_file, 'detections': detections}
        except Exception as e:
            raise RuntimeError(f"Prediction failed for image {image_file}: {e}")
