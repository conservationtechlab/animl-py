"""
Static links to download the MegaDetector and Classifier models.
"""
from pathlib import Path
import wget


MEGADETECTOR = {
    'MDV5a': 'https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt',
    'MDV5b': 'https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5b.0.0.pt',
    'MDV1000-redwood': 'https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-redwood.pt',
    'MDV1000-larch': 'https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-larch.pt',
    'MDV1000-sorrel': 'https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-sorrel.pt',
    'MDV1000-spruce': 'https://github.com/agentmorris/MegaDetector/releases/download/v1000.0/md_v1000.0.0-spruce.pt',
    'MDV6-yolov10-c': 'https://zenodo.org/records/15398270/files/MDV6-yolov10-c.pt?download=1',
    'MDV6-yolov10-e': 'https://zenodo.org/records/15398270/files/MDV6-yolov10-e-1280.pt?download=1',
    'MDV6-yolov9-c': 'https://zenodo.org/records/15398270/files/MDV6-yolov9-c.pt?download=1',
    'MDV6-yolov9-e': 'https://zenodo.org/records/15398270/files/MDV6-yolov9-e-1280.pt?download=1',
}

MD_FILENAMES = {
    'MDV5a': 'md_v5a.0.0.pt',
    'MDV5b': 'md_v5b.0.0.pt',
    'MDV1000-redwood': 'md_v1000.0.0-redwood.pt',
    'MDV1000-larch': 'md_v1000.0.0-larch.pt',
    'MDV1000-sorrel': 'md_v1000.0.0-sorrel.pt',
    'MDV1000-spruce': 'md_v1000.0.0-spruce.pt',
    'MDV6-yolov10-c': 'MDV6-yolov10-c.pt',
    'MDV6-yolov10-e': 'MDV6-yolov10-e-1280.pt',
    'MDV6-yolov9-c': 'MDV6-yolov9-c.pt',
    'MDV6-yolov9-e': 'MDV6-yolov9-e-1280.pt',
}

CLASSIFIER = {
    'sdzwa_amazon_v2': 'None',
    'sdzwa_andes_v1': 'https://sandiegozoo.box.com/shared/static/a25q2uojqj8undj1x9mz26w1xayotcbl.pt',
    'sdzwa_savanna_v3': 'https://sandiegozoo.box.com/shared/static/m1h1q689bma52rosuk00k3o6zgt2nrc1.pt',
    'sdzwa_southwest_v3': 'https://sandiegozoo.box.com/shared/static/ucbk8kc2h3qu15g4xbg0nvbghvo1cl97.pt',
}

CLASS_LIST = {
    'sdzwa_amazon_v2': 'None',
    'sdzwa_andes_v1': 'https://sandiegozoo.box.com/shared/static/c50li41abi3cvoaxzuo1pzs1x9zw37s8.csv',
    'sdzwa_savanna_v3': 'https://sandiegozoo.box.com/shared/static/r5fcvksluzgk1qfi1ayjik3ew5v9279s.csv',
    'sdzwa_southwest_v3': 'https://sandiegozoo.box.com/shared/static/lplo7ifz1xwdie1jw4400r377kweahsu.csv',
}


def download_model(model_url: str,
                   out_dir: str = 'models'):
    """Download specified model to the given directory.

    Args:
        model_url (str): url of the model to download, obtained via the constants above
        out_dir (str): Directory to save the model.

    Returns:
        None
    """
    Path(out_dir).mkdir(exist_ok=True)
    print('Saving to', out_dir)
    wget.download(model_url, out=out_dir)


def list_models():
    """List available models for download.

    Args:
        None    
    Returns:
        None
    """
    print('MegaDetector models:')
    for k in MEGADETECTOR.keys():
        print(f'  {k}')
    print('\nClassifier models:')
    for k in CLASSIFIER.keys():
        print(f'  {k}')
    print('\nClassifier class lists:')
    for k in CLASS_LIST.keys():
        print(f'  {k}')