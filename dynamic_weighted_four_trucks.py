import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from four_plus_truck_function import dyn_multi_opt


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

# Truck 1

data1 = data[data.Ward == 0]
visit1, visit2, visit3, visit4 = (
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    )
visitedNodes = set()
obj_value1 = dyn_multi_opt(data1, [visit1, visit2, visit3, visit4], visitedNodes = visitedNodes, distances = distance, ward_name = 'Truck 1', t_name = 'truck1', folder_Path = 'Data/Dynamic Data/Multiple Trucks/4 Trucks/', w1 = 0.9, w2 = 0.1, n_done = [0, 0, 0, 0], n_trucks = 4, obj_value=[])
print('\n\n Truck 1 Done \n\n')

# Truck 2

data2 = data[data.Ward == 1]
visit1, visit2, visit3, visit4 = (
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    )
visitedNodes = set()
obj_value2 = dyn_multi_opt(data2, [visit1, visit2, visit3, visit4], visitedNodes = visitedNodes, distances = distance, ward_name = 'Truck 2', t_name = 'truck2', folder_Path = 'Data/Dynamic Data/Multiple Trucks/4 Trucks/', w1 = 0.9, w2 = 0.1, n_done = [0, 0, 0, 0], n_trucks = 4, obj_value=[])
print('\n\n Truck 2 Done \n\n')

# Truck 3

data3 = data[data.Ward == 2]
visit1, visit2, visit3, visit4 = (
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
    )
visitedNodes = set()
obj_value3 = dyn_multi_opt(data3, [visit1, visit2, visit3, visit4], visitedNodes = visitedNodes, distances = distance, ward_name = 'Truck 3', t_name = 'truck3', folder_Path = 'Data/Dynamic Data/Multiple Trucks/4 Trucks/', w1 = 0.9, w2 = 0.1, n_done = [0, 0, 0, 0], n_trucks = 4, obj_value=[])
print('\n\n Truck 3 Done \n\n')


# Collect Data
distance = pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1)
path11 = []
path12 = []
path13 = []
path14 = []
path21 = []
path22 = []
path23 = []
path24 = []
path31 = []
path32 = []
path33 = []
path34 = []
v11 = pd.read_csv('Data/Dynamic Data/Multiple Trucks/4 Trucks/Visited Truck 1/visited_truck1_1_0.9_0.1.csv')
v12 = pd.read_csv('Data/Dynamic Data/Multiple Trucks/4 Trucks/Visited Truck 1/visited_truck1_2_0.9_0.1.csv')
v13 = pd.read_csv('Data/Dynamic Data/Multiple Trucks/4 Trucks/Visited Truck 1/visited_truck1_3_0.9_0.1.csv')
v14 = pd.read_csv('Data/Dynamic Data/Multiple Trucks/4 Trucks/Visited Truck 1/visited_truck1_4_0.9_0.1.csv')
v21 = pd.read_csv('Data/Dynamic Data/Multiple Trucks/4 Trucks/Visited Truck 2/visited_truck2_1_0.9_0.1.csv')
v22 = pd.read_csv('Data/Dynamic Data/Multiple Trucks/4 Trucks/Visited Truck 2/visited_truck2_2_0.9_0.1.csv')
v23 = pd.read_csv('Data/Dynamic Data/Multiple Trucks/4 Trucks/Visited Truck 2/visited_truck2_3_0.9_0.1.csv')
v24 = pd.read_csv('Data/Dynamic Data/Multiple Trucks/4 Trucks/Visited Truck 2/visited_truck2_4_0.9_0.1.csv')
v31 = pd.read_csv('Data/Dynamic Data/Multiple Trucks/4 Trucks/Visited Truck 3/visited_truck3_1_0.9_0.1.csv')
v32 = pd.read_csv('Data/Dynamic Data/Multiple Trucks/4 Trucks/Visited Truck 3/visited_truck3_2_0.9_0.1.csv')
v33 = pd.read_csv('Data/Dynamic Data/Multiple Trucks/4 Trucks/Visited Truck 3/visited_truck3_3_0.9_0.1.csv')
v34 = pd.read_csv('Data/Dynamic Data/Multiple Trucks/4 Trucks/Visited Truck 3/visited_truck3_4_0.9_0.1.csv')
v11.Node = v11.Node.astype('int')
v12.Node = v12.Node.astype('int')
v13.Node = v13.Node.astype('int')
v14.Node = v14.Node.astype('int')
v21.Node = v21.Node.astype('int')
v22.Node = v22.Node.astype('int')
v23.Node = v23.Node.astype('int')
v24.Node = v24.Node.astype('int')
v31.Node = v31.Node.astype('int')
v32.Node = v32.Node.astype('int')
v33.Node = v33.Node.astype('int')
v34.Node = v34.Node.astype('int')
for i in range(len(v11) - 1):
    path11.append((v11.iloc[i, 0], v11.iloc[i + 1, 0]))
for i in range(len(v12) - 1):
    path12.append((v12.iloc[i, 0], v12.iloc[i + 1, 0]))
for i in range(len(v13) - 1):
    path13.append((v13.iloc[i, 0], v13.iloc[i + 1, 0]))
for i in range(len(v14) - 1):
    path14.append((v14.iloc[i, 0], v14.iloc[i + 1, 0]))
for i in range(len(v21) - 1):
    path21.append((v21.iloc[i, 0], v21.iloc[i + 1, 0]))
for i in range(len(v22) - 1):
    path22.append((v22.iloc[i, 0], v22.iloc[i + 1, 0]))
for i in range(len(v23) - 1):
    path23.append((v23.iloc[i, 0], v23.iloc[i + 1, 0]))
for i in range(len(v24) - 1):
    path24.append((v24.iloc[i, 0], v24.iloc[i + 1, 0]))
for i in range(len(v31) - 1):
    path31.append((v31.iloc[i, 0], v31.iloc[i + 1, 0]))
for i in range(len(v32) - 1):
    path32.append((v32.iloc[i, 0], v32.iloc[i + 1, 0]))
for i in range(len(v33) - 1):
    path33.append((v33.iloc[i, 0], v33.iloc[i + 1, 0]))
for i in range(len(v34) - 1):
    path34.append((v34.iloc[i, 0], v34.iloc[i + 1, 0]))
gar11 = v11.iloc[-1,1]*10
gar12 = v12.iloc[-1,1]*10
gar13 = v13.iloc[-1,1]*10
gar14 = v14.iloc[-1,1]*10
gar21 = v21.iloc[-1,1]*10
gar22 = v22.iloc[-1,1]*10
gar23 = v23.iloc[-1,1]*10
gar24 = v24.iloc[-1,1]*10
gar31 = v31.iloc[-1,1]*10
gar32 = v32.iloc[-1,1]*10
gar33 = v33.iloc[-1,1]*10
gar34 = v34.iloc[-1,1]*10
dist11 = sum([distance.iloc[i,j] for i,j in path11])
dist12 = sum([distance.iloc[i,j] for i,j in path12])
dist13 = sum([distance.iloc[i,j] for i,j in path13])
dist14 = sum([distance.iloc[i,j] for i,j in path14])
dist21 = sum([distance.iloc[i,j] for i,j in path21])
dist22 = sum([distance.iloc[i,j] for i,j in path22])
dist23 = sum([distance.iloc[i,j] for i,j in path23])
dist24 = sum([distance.iloc[i,j] for i,j in path24])
dist31 = sum([distance.iloc[i,j] for i,j in path31])
dist32 = sum([distance.iloc[i,j] for i,j in path32])
dist33 = sum([distance.iloc[i,j] for i,j in path33])
dist34 = sum([distance.iloc[i,j] for i,j in path34])

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
            round(gar12, 4),
            round(gar13, 4),
            round(gar14, 4)],
        'Garbage Fill Ward 1 (in Litres)' : [
            round(gar11/10 * B_TO_B, 4),
            round(gar12/10 * B_TO_B, 4),
            round(gar13/10 * B_TO_B, 4),
            round(gar14/10 * B_TO_B, 4)],
        'Distance Travelled Ward 1 (in m)' : [
            round(dist11, 4),
            round(dist12, 4),
            round(dist13, 4),
            round(dist14, 4)],
        'Garbage per Meter Ward 2 (in KG/m)' : [
            round(gar11/dist11, 4),
            round(gar12/dist12, 4),
            round(gar13/dist13, 4),
            round(gar14/dist14, 4)],
        'Percentage of Bins covered Ward 1 (in %)' : [
            round( 100 * (v11.shape[0] - 2)/ data[data.Ward == 0].shape[0], 4),
            round( 100 * (v12.shape[0] - 2)/ data[data.Ward == 0].shape[0], 4),
            round( 100 * (v13.shape[0] - 2)/ data[data.Ward == 0].shape[0], 4),
            round( 100 * (v14.shape[0] - 2)/ data[data.Ward == 0].shape[0], 4)],
        'Fill Ward 2 (in %)' : [
            round(gar21, 4), 
            round(gar22, 4),
            round(gar23, 4),
            round(gar24, 4)],
        'Garbage Fill Ward 2 (in Litres)' : [
            round(gar21/10 * B_TO_B, 4),
            round(gar22/10 * B_TO_B, 4),
            round(gar23/10 * B_TO_B, 4),
            round(gar24/10 * B_TO_B, 4)],
        'Distance Travelled Ward 2 (in m)' : [
            round(dist21, 4),
            round(dist22, 4),
            round(dist23, 4),
            round(dist24, 4)],
        'Garbage per Meter Ward 2 (in KG/m)' : [
            round(gar21/dist21, 4),
            round(gar22/dist22, 4),
            round(gar23/dist23, 4),
            round(gar24/dist24, 4)],
        'Percentage of Bins covered Ward 2 (in %)' : [
            round( 100 * (v21.shape[0] - 2)/ data[data.Ward == 1].shape[0], 4),
            round( 100 * (v22.shape[0] - 2)/ data[data.Ward == 1].shape[0], 4),
            round( 100 * (v23.shape[0] - 2)/ data[data.Ward == 1].shape[0], 4),
            round( 100 * (v24.shape[0] - 2)/ data[data.Ward == 1].shape[0], 4)],
        'Fill Ward 3 (in %)' : [
            round(gar31, 4), 
            round(gar32, 4),
            round(gar33, 4),
            round(gar34, 4)],
        'Garbage Fill Ward 3 (in Litres)' : [
            round(gar31/10 * B_TO_B, 4),
            round(gar32/10 * B_TO_B, 4),
            round(gar33/10 * B_TO_B, 4),
            round(gar34/10 * B_TO_B, 4)],
        'Distance Travelled Ward 3 (in m)' : [
            round(dist31, 4),
            round(dist32, 4),
            round(dist33, 4),
            round(dist34, 4)],
        'Garbage per Meter Ward 3 (in KG/m)' : [
            round(gar31/dist31, 4),
            round(gar32/dist32, 4),
            round(gar33/dist33, 4),
            round(gar34/dist34, 4)],
        'Percentage of Bins covered Ward 3 (in %)' : [
            round( 100 * (v31.shape[0] - 2)/ data[data.Ward == 2].shape[0], 4),
            round( 100 * (v32.shape[0] - 2)/ data[data.Ward == 2].shape[0], 4),
            round( 100 * (v33.shape[0] - 2)/ data[data.Ward == 2].shape[0], 4),
            round( 100 * (v34.shape[0] - 2)/ data[data.Ward == 2].shape[0], 4)],
    }, index=['Truck 1', 'Truck 2', 'Truck 3', 'Truck 4'])
stats.to_csv('Data/Dynamic Data/Multiple Trucks/4 Trucks/Statistics.csv')