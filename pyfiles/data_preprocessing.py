import pandas as pd
import numpy as np
import datetime

## Creating a function that can preprocess our data.

def get_data(pth):

    data = pd.read_csv(pth)