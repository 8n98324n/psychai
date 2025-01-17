        
import numpy as np
import os
import pandas as pd

"""

"""

def check_path_or_create(path):
    # Check if the path exists
    if not os.path.exists(path):
        # If it does not exist, create the directory
        os.makedirs(path)
    return path
