import os

PROJECT_PATH = os.path.dirname(__file__)

# absolute path to mnist.pkl file
DATA_PATH = None

try:
    from user_settings import *
except ImportError:
    pass
