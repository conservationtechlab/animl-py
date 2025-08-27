"""
Static links to download the MegaDetector and Classifier models.
"""
import os
import wget


MEGADETECTOR = {
    'MDV5a': 'https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt',
    'MDV5b': 'https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5b.0.0.pt',
}

CLASSIFIER = {
    'SDZWA_Amazon_v2': 'None',
    'SDZWA_Andes_v1': 'https://sandiegozoo.box.com/shared/static/a25q2uojqj8undj1x9mz26w1xayotcbl.pt',
    'SDZWA_Savanna_v3': 'https://sandiegozoo.box.com/shared/static/m1h1q689bma52rosuk00k3o6zgt2nrc1.pt',
    'SDZWA_Southwest_v3': 'https://sandiegozoo.box.com/shared/static/ucbk8kc2h3qu15g4xbg0nvbghvo1cl97.pt',
}

CLASS_LIST = {
    'SDZWA_Amazon_v2': 'None',
    'SDZWA_Andes_v1': 'https://sandiegozoo.box.com/shared/static/c50li41abi3cvoaxzuo1pzs1x9zw37s8.csv',
    'SDZWA_Savanna': 'https://sandiegozoo.box.com/shared/static/r5fcvksluzgk1qfi1ayjik3ew5v9279s.csv',
    'SDZWA_Southwest_v3': 'https://sandiegozoo.box.com/shared/static/lplo7ifz1xwdie1jw4400r377kweahsu.csv',
}

def download_model(model_url: str,
                   out_dir: str = 'models'):
    """Download specified model to the given directory.

    Args:
        model_url (str): url of the model to download, obtained via the constants above
        home (str): Directory to save the model.

    Returns:
        None
    """
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    print('Saving to', out_dir)
    wget.download(model_url, out=out_dir)
