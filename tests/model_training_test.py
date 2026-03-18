"""
Test model training pipeline

"""
import unittest
import yaml
import shutil
from pathlib import Path

import animl


@unittest.skip
def model_training_test():
    print(Path.cwd())
    config = Path.cwd() / 'examples/Cats_vs_Dogs' / 'cat_dog_train.yml'

    cfg = yaml.safe_load(open(config, 'r'))

    output = Path(cfg['experiment_folder'])
    if output.exists():
        shutil.rmtree(output)
    Path(output).mkdir(exist_ok=True)

    animl.train_classifier(config)
    animl.test_classifier(config)


model_training_test()
