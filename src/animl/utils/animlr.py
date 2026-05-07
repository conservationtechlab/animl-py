"""
Utils for animl-r
"""

from animl import __version__


def get_version():
    """Returns the version of animl-py."""
    return __version__


def check_exiftool():
    """Checks if exiftool is installed and accessible."""
    import exiftool
    try:
        with exiftool.ExifToolHelper() as et:
<<<<<<< detect_include_categories
            return et.version()
    except Exception:
=======
            return et.version
    except Exception as e:
>>>>>>> dev
        return False


def check_torch_cuda():
    """Checks if CUDA is available for PyTorch."""
    import torch
    return torch.cuda.is_available()


def check_onnx_cuda():
    """Checks if CUDA is available for ONNX Runtime."""
    import onnxruntime as ort
    providers = ort.get_available_providers()
    return 'CUDAExecutionProvider' in providers
