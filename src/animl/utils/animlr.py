"""
Utils for animl-r
"""

from animl import __version__

def get_version():
    """Returns the version of animl-r."""
    return __version__

def check_installation():
    """Checks if animl-py is installed correctly."""
    # TODO: Implement more comprehensive checks, such as verifying that all 
    # dependencies are installed and that the package can be imported without errors.
    try:
        version = get_version()
        print(f"animl-py version {version} is installed correctly.")
    except Exception as e:
        print("Error: animl-py is not installed correctly.")
        print(str(e))