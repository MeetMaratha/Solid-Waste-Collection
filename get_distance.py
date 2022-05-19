import numpy as np
import pandas as pd

df = pd.read_csv('Data/Bin Locations.csv', index_col='id')
dist_mat = np.zeros((df.shape[0], df.shape[0]))
print(dist_mat.shape)