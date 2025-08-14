import yaml
from pathlib import Path

from animl import train

print(Path.cwd())
config = Path.cwd() / 'examples/Cat_vs_Dog' / 'cat_dog_train.yml'
# load cfg file
cfg = yaml.safe_load(open(config, 'r'))


train.main(cfg)