import pandas as pd
import os


def symlinkSpecies(data, linkdir, classes):
    table = pd.read_table(classes, sep=" ", index_col=0)
    for i in range(0, len(table.index)):
        directory = str(table['x'].values[i])
        if not os.path.isdir(linkdir + directory):
            os.makedirs(linkdir + directory)

    for i in range(0, len(data.index)):
        try:
            os.symlink(data.at[i, 'file'],
                       linkdir + data.at[i, 'class'] + "/" + os.path.basename(data.at[i, 'file']))
        except Exception as e:
            print('File already exists. Exception: {}'.format(e))
            continue
            
            
#def symlinkMD():

#der symUnlink():
