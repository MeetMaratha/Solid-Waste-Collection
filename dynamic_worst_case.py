import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dynamic_function import dyn_opt
from show_routes import CreateMap

# Constants
B_TO_B = 100
B_TO_T = 10
N_WARDS = 3
N_TRUCKS = 1
W1 = 0.9
W2 = 0.1

# Set Random Seed
np.random.seed(42)

# Import Data
data = pd.read_csv('Data/Bin Locations.csv', index_col= 'id').sort_index()
distance = pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1)
for i in range(distance.shape[0]):
    distance.iloc[:, i] = distance.iloc[:, i]/np.max(distance.iloc[:, i])


# Add Fill_ratio, distance and fill per meter
fill_ratio = [0.0] + [1.0] * (data.shape[0] - 1)
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

obj_value = dyn_opt(data1, data2, data3, distance, folder_path = 'Data/Dynamic Data/Worst Case/', w1 = W1, w2 = W2, visit1 = visit1, visit2 = visit2, visit3 = visit3)

# Collect Data
distance = pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1)
v1 = pd.read_csv(f'Data/Dynamic Data/Worst Case/Visited Truck 1/visited_truck1_{W1}_{W2}.csv')
v2 = pd.read_csv(f'Data/Dynamic Data/Worst Case/Visited Truck 2/visited_truck2_{W1}_{W2}.csv')
v3 = pd.read_csv(f'Data/Dynamic Data/Worst Case/Visited Truck 3/visited_truck3_{W1}_{W2}.csv')
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

print('--------------- SAVING STATISTICS ----------------------\n')
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
stats.to_csv('Data/Dynamic Data/Worst Case/Statistics.csv')

print('--------------- GENERATING MAP ----------------------')
# Plotting routes

map = CreateMap()
map.createRoutes('Data/Dynamic Data/Worst Case/', N_WARDS, N_TRUCKS, W1, W2, Multiple_truck = False)
map.createLatLong('Data/Bin Locations.csv', N_WARDS)
map.createRoutesDict(N_WARDS)
map.addRoutesToMap(N_WARDS, N_TRUCKS)
map.addDepot()
map.addNodes('Data/Bin Locations.csv')
map.saveMap('Data/Dynamic Data/Worst Case/')
map.displayMap('Data/Dynamic Data/Worst Case/')