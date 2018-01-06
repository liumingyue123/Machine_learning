import csv
import os
import pandas as pd

data = pd.read_csv(filepath_or_buffer=os.path.join(os.path.dirname(__file__), 'test.csv'))
