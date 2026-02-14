"""
Visualiser for datasets using python
"""

# Imports
import numpy as np
import pandas as pd

from streamlit_common_utils.csv.csv_utils import *

# Main Vars
INPUT_DATA_TYPES = {
    "int": [int, np.int64], 
    "float": [float, np.float64], 
    "text": [str, object]
}

# Main Functions