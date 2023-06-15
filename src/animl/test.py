#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:09:03 2023

@author: kyra
"""

import argparse
import sys
import json
import os


imagedir = "/home/kyra/animl-py/examples/Southwest"


    # load the checkpoint if available
    # relative file names are only output at the end; all file paths in the checkpoint are still full paths
    if args.resume_from_checkpoint:
        assert os.path.exists(args.resume_from_checkpoint), 'File at resume_from_checkpoint specified does not exist'
        with open(args.resume_from_checkpoint) as f:
            saved = json.load(f)
        assert 'images' in saved, \
            'The file saved as checkpoint does not have the correct fields; cannot be restored'
        results = saved['images']
        print('Restored {} entries from the checkpoint'.format(len(results)))
    else:
        results = []
    images = imagesFromVideos(args.image_dir, out_dir=args.output_dir,
                              fps=args.fps, frames=args.frames,
                              parallel=args.parallel)

    # test that we can write to the output_file's dir if checkpointing requested
    if args.checkpoint_frequency != -1:
        checkpoint_path = os.path.join(args.output_dir,
                                       'checkpoint_{}.json'.format(datetime.utcnow().strftime("%Y%m%d%H%M%S")))
        with open(checkpoint_path, 'w') as f:
            json.dump({'images': []}, f)
        print('The checkpoint file will be written to {}'.format(checkpoint_path))
    else:
        checkpoint_path = None

    # run MegaDetector and format results
    detections = detectObjectBatch(images, MegaDetector_file, checkpoint_path, 
                                   args.threshold, args.checkpoint_frequency,
                                   results)

    df = parseMD(detections)
    # filter out all non animal detections
    animalDataframe = getAnimals(df)
    otherDataframe = getEmpty(df)

    # Create generator for classification model
    generator = GenerateCropsFromFile(animalDataframe)

    # Create and predict model
    model = keras.models.load_model(Classification_file)
    predictions = model.predict(generator)

    # Format Classification results
    maxDataframe = applyPredictions(animalDataframe, otherDataframe, predictions, classes)
    # Symlink
    symlinkSpecies(maxDataframe, args.output_dir, classes)


if __name__ == '__main__':
    main()
