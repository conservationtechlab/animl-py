"""
Test model training pipeline

"""
from pathlib import Path

from animl import train, test

print(Path.cwd())
config = Path.cwd() / 'examples/Cats_vs_Dogs' / 'cat_dog_train.yml'

train.train_main(config)
test.test_main(config)