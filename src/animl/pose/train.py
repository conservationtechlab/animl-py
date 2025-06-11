from animl import generator
import pandas as pd
from animl.split import train_val_test
#from animl import classification
import numpy as np
import comet_ml

if __name__ == "__main__":
    test_dataset = pd.read_csv('/mnt/machinelearning/Viewpoint/csvs/merged-dataset.csv').reset_index(drop=True)
    train, val = train_val_test(manifest=test_dataset,out_dir='csvs',label_col='dataset_name',percentage=(0.8,0.2,0.0), other_groups=['file_name','viewpoint'])
    #exp = comet_ml.start(project_name="demo-project")

#dl_train = generator.train_dataloader(manifest=test_dataset,classes='viewpoint',batch_size=32,file_col='file_name',
#                          label_col='viewpoint',crop=True,augment=True,cache_dir='pose/cache',crop_coord='absolute')
#15 epochs