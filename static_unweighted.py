import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from static_function import opt

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
routes, _ = opt(data, distance, w1 = 0.5, w2 = 0.5)

# Saving all data
data.to_csv('Data/Static Data/Unweighted/Truck Data.csv')
#Save Data Truck 1
nodes1 = [0]
fill1 = [0]
next_element = -1
while next_element != 0:
    next_element = next( y for x, y in routes[0] if x == nodes1[-1] )
    nodes1.append(next_element)
    if next_element == 0:
        fill1.append(np.sum(fill1))
    else:
        fill1.append(data.loc[next_element, 'fill_ratio'])
visit_static_1 = pd.DataFrame({'Nodes' : nodes1, 'Fill Ratio' : fill1})
file_name = 'Data/Static Data/Unweighted/Visited Truck Data/Visited Truck 1.csv'
visit_static_1.to_csv(file_name, index=False)
# print(len(nodes), len(fill))

#Save Data Truck 2
nodes2 = [0]
fill2 = [0]
next_element = -1
while next_element != 0:
    next_element = next( y for x, y in routes[1] if x == nodes2[-1] )
    nodes2.append(next_element)
    if next_element == 0:
        fill2.append(np.sum(fill2))
    else:
        fill2.append(data.loc[next_element, 'fill_ratio'])
visit_static_2 = pd.DataFrame({'Nodes' : nodes2, 'Fill Ratio' : fill2})
file_name = 'Data/Static Data/Unweighted/Visited Truck Data/Visited Truck 2.csv'
visit_static_2.to_csv(file_name, index=False)
# print(len(nodes), len(fill))

#Save Data Truck 3
nodes3 = [0]
fill3 = [0]
next_element = -1
while next_element != 0:
    next_element = next( y for x, y in routes[2] if x == nodes3[-1] )
    nodes3.append(next_element)
    if next_element == 0:
        fill3.append(np.sum(fill3))
    else:
        fill3.append(data.loc[next_element, 'fill_ratio'])
visit_static_3 = pd.DataFrame({'Nodes' : nodes3, 'Fill Ratio' : fill3})
file_name = 'Data/Static Data/Unweighted/Visited Truck Data/Visited Truck 3.csv'
visit_static_3.to_csv(file_name, index=False)

stat_dist = [0,0,0]
j = 0
for i in routes:
    for k in i:
        stat_dist[j] = stat_dist[j] + distance.iloc[k[0], k[1]]
    j = j + 1

# Uncomment if you want to print Stats

# print(f"Fill of truck 1 : {round(visit_static_1.iloc[-1, 1]*10, 4)}%")
# print(f"Fill of truck 2 : {round(visit_static_2.iloc[-1, 1]*10, 4)}%")
# print(f"Fill of truck 3 : {round(visit_static_3.iloc[-1, 1]*10, 4)}%")

# print("\n")
# print(f"Garbage fill of truck 1 : {round(visit_static_1.iloc[-1, 1]*B_TO_B, 4)}")
# print(f"Garbage fill of truck 2 : {round(visit_static_2.iloc[-1, 1]*B_TO_B, 4)}")
# print(f"Garbage fill of truck 3 : {round(visit_static_3.iloc[-1, 1]*B_TO_B, 4)}")


# print("\n")
# print(f"Distance travelled by truck 1 : {round(stat_dist[0], 4)}")
# print(f"Distance travelled by truck 2 : {round(stat_dist[1], 4)}")
# print(f"Distance travelled by truck 3 : {round(stat_dist[2], 4)}")    
    
# print("\n")
# print(f"Garbage per meter by truck 1 : {round(visit_static_1.iloc[-1, 1]*B_TO_T / stat_dist[0], 4)}")
# print(f"Garbage per meter by truck 2 : {round(visit_static_2.iloc[-1, 1]*B_TO_T / stat_dist[1], 4)}")
# print(f"Garbage per meter by truck 3 : {round(visit_static_3.iloc[-1, 1]*B_TO_T / stat_dist[2], 4)}")

# print("\n")
# print(f"Percentage of bins covered by truck 1 : {round( 100 * (visit_static_1.shape[0] - 2)/ data[data.Ward == 0].shape[0], 4)} %")
# print(f"Percentage of bins covered by truck 2 : {round( 100 * (visit_static_2.shape[0] - 2)/ data[data.Ward == 1].shape[0], 4)} %")
# print(f"Percentage of bins covered by truck 3 : {round( 100 * (visit_static_3.shape[0] - 2)/ data[data.Ward == 2].shape[0], 4)} %")

stats = pd.DataFrame(
    {
        'Fill (in %)' : [
            round(visit_static_1.iloc[-1, 1]*10, 4), 
            round(visit_static_2.iloc[-1, 1]*10, 4), 
            round(visit_static_3.iloc[-1, 1]*10, 4)],
        'Garbage Fill (in KG)' : [
            round(visit_static_1.iloc[-1, 1]*B_TO_B, 4), 
            round(visit_static_2.iloc[-1, 1]*B_TO_B, 4),
            round(visit_static_3.iloc[-1, 1]*B_TO_B, 4)],
        'Distance Travelled (in m)' : [
            round(stat_dist[0], 4),
            round(stat_dist[1], 4),
            round(stat_dist[2], 4)],
        'Garbage per Meter (in KG/m)' : [
            round(visit_static_1.iloc[-1, 1]*B_TO_T / stat_dist[0], 4),
            round(visit_static_2.iloc[-1, 1]*B_TO_T / stat_dist[0], 4),
            round(visit_static_3.iloc[-1, 1]*B_TO_T / stat_dist[0], 4)],
        'Percentage of Bins covered (in %)' : [
            round( 100 * (visit_static_1.shape[0] - 2)/ data[data.Ward == 0].shape[0], 4),
            round( 100 * (visit_static_2.shape[0] - 2)/ data[data.Ward == 0].shape[0], 4),
            round( 100 * (visit_static_3.shape[0] - 2)/ data[data.Ward == 0].shape[0], 4)]
    }, index=['Truck 1', 'Truck 2', 'Truck 3'])
stats.to_csv('Data/Static Data/Unweighted/Statistics.csv')