import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dynamic_function import dyn_opt

# Constants
B_TO_B = 100
B_TO_T = 10

# Set Random Seed
np.random.seed(42)

# Import Data
data = pd.read_csv('Data/Bin Locations.csv', index_col= 'id').sort_index()
distance = pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1)

# Add Fill_ratio, distance and fill per meter
fill_ratio = [0.0] + [np.random.rand() for i in range(data.shape[0] - 1)]
distance_from_0 = distance.iloc[:, 0]
data['fill_ratio'] = fill_ratio
data['distance_from_0'] = distance_from_0
fill_p_m = [0.0] + list(B_TO_B * data.loc[1:, 'fill_ratio'] / data.loc[1:, 'distance_from_0'])
data['fill_p_m'] = fill_p_m

# Optimization
visit1, visit2, visit3 = (
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    )
data1 = data[data.Ward == 0]
data2 = data[data.Ward == 1]
data3 = data[data.Ward == 2]

obj_value = dyn_opt(data1, data2, data3, distance, folder_path = 'Data/Dynamic Data/Unweighted/', w1 = 0.5, w2 = 0.5, visit1 = visit1, visit2 = visit2, visit3 = visit3)

# Collect Data

v1 = pd.read_csv('Data/Dynamic Data/Unweighted/Visited Truck 1/visited_truck1_0.5_0.5.csv')
v2 = pd.read_csv('Data/Dynamic Data/Unweighted/Visited Truck 2/visited_truck2_0.5_0.5.csv')
v3 = pd.read_csv('Data/Dynamic Data/Unweighted/Visited Truck 3/visited_truck3_0.5_0.5.csv')
v1.Node = v1.Node.astype('int')
v2.Node = v2.Node.astype('int')
v3.Node = v3.Node.astype('int')
path1 = []
path2 = []
path3 = []
for i in range(len(v1) - 1):
    path1.append((v1.iloc[i, 0], v1.iloc[i + 1, 0]))
for i in range(len(v2) - 1):
    path2.append((v2.iloc[i, 0], v2.iloc[i + 1, 0]))
for i in range(len(v3) - 1):
    path3.append((v3.iloc[i, 0], v3.iloc[i + 1, 0]))
gar1 = v1.iloc[-1,1]*10
gar2 = v2.iloc[-1,1]*10
gar3 = v3.iloc[-1,1]*10
dist1 = sum([distance.iloc[i,j] for i,j in path1])
dist2 = sum([distance.iloc[i,j] for i,j in path2])
dist3 = sum([distance.iloc[i,j] for i,j in path3])

# Uncomment if you want to print Stats

# print("\n")
# print(f'Fill Ratio of truck 1 : {round(gar1, 4)}')
# print(f'Fill Ratio of truck 2 : {round(gar2, 4)}')
# print(f'Fill Ratio of truck 3 : {round(gar3, 4)}')

# print("\n")
# print(f'Garbage collected by truck 1 : {round(gar1/10 * B_TO_B, 4)}')
# print(f'Garbage collected by truck 2 : {round(gar2/10 * B_TO_B, 4)}')
# print(f'Garbage collected by truck 3 : {round(gar3/10 * B_TO_B, 4)}')

# print("\n")
# print(f'Distance travelled by truck 1 : {round(dist1, 4)}')
# print(f'Distance travelled by truck 2 : {round(dist2, 4)}')
# print(f'Distance travelled by truck 3 : {round(dist3, 4)}')

# print("\n")
# print(f'Garbage per meter for truck 1 : {round(gar1/dist1, 4)}')
# print(f'Garbage per meter for truck 2 : {round(gar2/dist2, 4)}')
# print(f'Garbage per meter for truck 3 : {round(gar3/dist3, 4)}')

# Save Statistics

stats = pd.DataFrame(
    {
        'Fill (in %)' : [
            round(gar1, 4), 
            round(gar2, 4), 
            round(gar3, 4)],
        'Garbage Fill (in Litres)' : [
            round(gar1/10 * B_TO_B, 4),
            round(gar2/10 * B_TO_B, 4),
            round(gar3/10 * B_TO_B, 4)],
        'Distance Travelled (in m)' : [
            round(dist1, 4),
            round(dist2, 4),
            round(dist3, 4)],
        'Garbage per Meter (in KG/m)' : [
            round(gar1/dist1, 4),
            round(gar2/dist2, 4),
            round(gar3/dist3, 4)],
        'Percentage of Bins covered (in %)' : [
            round( 100 * (v1.shape[0] - 2)/ data[data.Ward == 0].shape[0], 4),
            round( 100 * (v2.shape[0] - 2)/ data[data.Ward == 1].shape[0], 4),
            round( 100 * (v3.shape[0] - 2)/ data[data.Ward == 2].shape[0], 4)]
    }, index=['Truck 1', 'Truck 2', 'Truck 3'])
stats.to_csv('Data/Dynamic Data/Unweighted/Statistics.csv')