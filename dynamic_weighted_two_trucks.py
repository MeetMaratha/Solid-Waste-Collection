import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multi_truck_function import dyn_multi_opt


# Constants
B_TO_B = 100
B_TO_T = 10

# Set Random Seed
np.random.seed(42)

# Import Data
data = pd.read_csv('Data/Bin Locations.csv', index_col= 'id').sort_index()
distance = pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1)
for i in range(distance.shape[0]):
    distance.iloc[:, i] = distance.iloc[:, i]/np.max(distance.iloc[:, i])

# Optimization

data1 = data[data.Ward == 0]
visit1, visit2 = (
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    )
obj_value1 = dyn_multi_opt(data1, [visit1, visit2], distances = distance, ward_name = 'Truck 1', t_name = 'truck1', folder_Path = 'Data/Dynamic Data/Multiple Trucks/2 Trucks/', w1 = 0.9, w2 = 0.1, n_done = [0, 0], n_trucks = 2, obj_value=[])
print('\n\n Truck 1 Done \n\n')
data2 = data[data.Ward == 1]
visit1, visit2 = (
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    )
obj_value2 = dyn_multi_opt(data2, [visit1, visit2], distances = distance, ward_name = 'Truck 2', t_name = 'truck2', folder_Path = 'Data/Dynamic Data/Multiple Trucks/2 Trucks/', w1 = 0.9, w2 = 0.1, n_done = [0, 0], n_trucks = 2, obj_value=[])
print('\n\n Truck 2 Done \n\n')
data3 = data[data.Ward == 2]
visit1, visit2 = (
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    )
obj_value3 = dyn_multi_opt(data3, [visit1, visit2], distances = distance, ward_name = 'Truck 3', t_name = 'truck3', folder_Path = 'Data/Dynamic Data/Multiple Trucks/2 Trucks/', w1 = 0.9, w2 = 0.1, n_done = [0, 0], n_trucks = 2, obj_value=[])
print('\n\n Truck 3 Done \n\n')


# Collect Data
distance = pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1)
path11 = []
path12 = []
path21 = []
path22 = []
path31 = []
path32 = []
v11 = pd.read_csv('Data/Dynamic Data/Multiple Trucks/2 Trucks/Visited Truck 1/visited_truck1_1_0.9_0.1.csv')
v12 = pd.read_csv('Data/Dynamic Data/Multiple Trucks/2 Trucks/Visited Truck 1/visited_truck1_2_0.9_0.1.csv')
v21 = pd.read_csv('Data/Dynamic Data/Multiple Trucks/2 Trucks/Visited Truck 2/visited_truck2_1_0.9_0.1.csv')
v22 = pd.read_csv('Data/Dynamic Data/Multiple Trucks/2 Trucks/Visited Truck 2/visited_truck2_2_0.9_0.1.csv')
v31 = pd.read_csv('Data/Dynamic Data/Multiple Trucks/2 Trucks/Visited Truck 3/visited_truck3_1_0.9_0.1.csv')
v32 = pd.read_csv('Data/Dynamic Data/Multiple Trucks/2 Trucks/Visited Truck 3/visited_truck3_2_0.9_0.1.csv')
v11.Node = v11.Node.astype('int')
v12.Node = v12.Node.astype('int')
v21.Node = v21.Node.astype('int')
v22.Node = v22.Node.astype('int')
v31.Node = v31.Node.astype('int')
v32.Node = v32.Node.astype('int')
for i in range(len(v11) - 1):
    path11.append((v11.iloc[i, 0], v11.iloc[i + 1, 0]))
for i in range(len(v12) - 1):
    path12.append((v12.iloc[i, 0], v12.iloc[i + 1, 0]))
for i in range(len(v21) - 1):
    path21.append((v21.iloc[i, 0], v21.iloc[i + 1, 0]))
for i in range(len(v22) - 1):
    path22.append((v22.iloc[i, 0], v22.iloc[i + 1, 0]))
for i in range(len(v31) - 1):
    path31.append((v31.iloc[i, 0], v31.iloc[i + 1, 0]))
for i in range(len(v32) - 1):
    path32.append((v32.iloc[i, 0], v32.iloc[i + 1, 0]))
gar11 = v11.iloc[-1,1]*10
gar12 = v12.iloc[-1,1]*10
gar21 = v21.iloc[-1,1]*10
gar22 = v22.iloc[-1,1]*10
gar31 = v31.iloc[-1,1]*10
gar32 = v32.iloc[-1,1]*10
dist11 = sum([distance.iloc[i,j] for i,j in path11])
dist12 = sum([distance.iloc[i,j] for i,j in path12])
dist21 = sum([distance.iloc[i,j] for i,j in path21])
dist22 = sum([distance.iloc[i,j] for i,j in path22])
dist31 = sum([distance.iloc[i,j] for i,j in path31])
dist32 = sum([distance.iloc[i,j] for i,j in path32])

# Uncomment if you want to print Stats

# print("\n")
# print(f'Garbage fill for truck 1 - 1 : {round(gar11, 4)}%')
# print(f'Garbage fill for truck 1 - 2 : {round(gar12, 4)}%')
# print(f'Garbage fill for truck 2 - 1 : {round(gar21, 4)}%')
# print(f'Garbage fill for truck 2 - 2 : {round(gar22, 4)}%')
# print(f'Garbage fill for truck 3 - 1 : {round(gar31, 4)}%')
# print(f'Garbage fill for truck 3 - 2 : {round(gar32, 4)}%')

# print("\n")
# print(f'Garbage collected for truck 1 - 1 : {round(gar11/10 * B_to_B, 4)}')
# print(f'Garbage collected for truck 1 - 2 : {round(gar12/10 * B_to_B, 4)}')
# print(f'Garbage collected for truck 2 - 1 : {round(gar21/10 * B_to_B, 4)}')
# print(f'Garbage collected for truck 2 - 2 : {round(gar22/10 * B_to_B, 4)}')
# print(f'Garbage collected for truck 3 - 1 : {round(gar31/10 * B_to_B, 4)}')
# print(f'Garbage collected for truck 3 - 2 : {round(gar32/10 * B_to_B, 4)}')

# print("\n")
# print(f'Distance travelled for truck 1 - 1 : {round(dist11, 4)}')
# print(f'Distance travelled for truck 1 - 2 : {round(dist12, 4)}')
# print(f'Distance travelled for truck 2 - 1 : {round(dist21, 4)}')
# print(f'Distance travelled for truck 2 - 2 : {round(dist22, 4)}')
# print(f'Distance travelled for truck 3 - 1 : {round(dist31, 4)}')
# print(f'Distance travelled for truck 3 - 2 : {round(dist32, 4)}')

# print("\n")
# print(f'Garbage per meter for truck 1 - 1 : {round(gar11/dist11, 4)}')
# print(f'Garbage per meter for truck 1 - 2 : {round(gar12/dist12, 4)}')
# print(f'Garbage per meter for truck 2 - 1 : {round(gar21/dist21, 4)}')
# print(f'Garbage per meter for truck 2 - 2 : {round(gar22/dist22, 4)}')
# print(f'Garbage per meter for truck 3 - 1 : {round(gar31/dist31, 4)}')
# print(f'Garbage per meter for truck 3 - 2 : {round(gar32/dist32, 4)}')

# Save Statistics

stats = pd.DataFrame(
    {
        'Fill Ward 1 (in %)' : [
            round(gar11, 4), 
            round(gar12, 4)],
        'Garbage Fill Ward 1 (in Litres)' : [
            round(gar11/10 * B_TO_B, 4),
            round(gar12/10 * B_TO_B, 4)],
        'Distance Travelled Ward 1 (in m)' : [
            round(dist11, 4),
            round(dist12, 4)],
        'Garbage per Meter Ward 2 (in KG/m)' : [
            round(gar11/dist11, 4),
            round(gar12/dist12, 4)],
        'Percentage of Bins covered Ward 1 (in %)' : [
            round( 100 * (v11.shape[0] - 2)/ data[data.Ward == 0].shape[0], 4),
            round( 100 * (v12.shape[0] - 2)/ data[data.Ward == 0].shape[0], 4)],
        'Fill Ward 2 (in %)' : [
            round(gar21, 4), 
            round(gar22, 4)],
        'Garbage Fill Ward 2 (in Litres)' : [
            round(gar21/10 * B_TO_B, 4),
            round(gar22/10 * B_TO_B, 4)],
        'Distance Travelled Ward 2 (in m)' : [
            round(dist21, 4),
            round(dist22, 4)],
        'Garbage per Meter Ward 2 (in KG/m)' : [
            round(gar21/dist21, 4),
            round(gar22/dist22, 4)],
        'Percentage of Bins covered Ward 2 (in %)' : [
            round( 100 * (v21.shape[0] - 2)/ data[data.Ward == 1].shape[0], 4),
            round( 100 * (v22.shape[0] - 2)/ data[data.Ward == 1].shape[0], 4)],
        'Fill Ward 3 (in %)' : [
            round(gar31, 4), 
            round(gar32, 4)],
        'Garbage Fill Ward 3 (in Litres)' : [
            round(gar31/10 * B_TO_B, 4),
            round(gar32/10 * B_TO_B, 4)],
        'Distance Travelled Ward 3 (in m)' : [
            round(dist31, 4),
            round(dist32, 4)],
        'Garbage per Meter Ward 3 (in KG/m)' : [
            round(gar31/dist31, 4),
            round(gar32/dist32, 4)],
        'Percentage of Bins covered Ward 3 (in %)' : [
            round( 100 * (v31.shape[0] - 2)/ data[data.Ward == 2].shape[0], 4),
            round( 100 * (v32.shape[0] - 2)/ data[data.Ward == 2].shape[0], 4)],
    }, index=['Truck 1', 'Truck 2'])
stats.to_csv('Data/Dynamic Data/Multiple Trucks/2 Trucks/Statistics.csv')