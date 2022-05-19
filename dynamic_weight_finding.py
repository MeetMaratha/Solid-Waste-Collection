import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from dynamic_function import dyn_opt

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

# Add Fill_ratio, distance and fill per meter
fill_ratio = [0.0] + [np.random.rand() for i in range(data.shape[0] - 1)]
distance_from_0 = distance.iloc[:, 0]
data['fill_ratio'] = fill_ratio
data['distance_from_0'] = distance_from_0
fill_p_m = [0.0] + list(B_TO_B * data.loc[1:, 'fill_ratio'] / data.loc[1:, 'distance_from_0'])
data['fill_p_m'] = fill_p_m

# Optimization
w1s = [round(i/10, 1) for i in range(0, 11)]
obj_values = []
for i in range(11):
    w1, w2 = round(i/10, 1), round(1 - i/10, 1)
    print(f"\n----------------- Processing w1 : {w1} | w2 : {w2} -----------------")
    visit1, visit2, visit3 = (
        pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
        pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
        pd.DataFrame({'Node': pd.Series(0, dtype='int'), 'fill_ratio': pd.Series(0, dtype='float')}), 
        )
    data1 = data[data.Ward == 0]
    data2 = data[data.Ward == 1]
    data3 = data[data.Ward == 2]

    obj_value = dyn_opt(data1, data2, data3, distance, folder_path = 'Data/Dynamic Data/Weight Finding/', w1 = w1, w2 = w2, visit1 = visit1, visit2 = visit2, visit3 = visit3)
    obj_values.append(np.sum(obj_value))

Figure = figure(figsize=(15, 15))
plt.scatter(w1s, obj_values)
plt.show()

arg = np.argmin(obj_values)
print(f"Best w1 value is : {w1s[arg]}.")
distance = pd.read_csv('Data/distance.csv').drop('Unnamed: 0', axis = 1)
# Storing statistics of best case
w1 = w1s[arg]
w2 = round(1 - w1, 1)
v1 = pd.read_csv(f'Data/Dynamic Data/Weight Finding/Visited Truck 1/visited_truck1_{w1}_{w2}.csv')
v2 = pd.read_csv(f'Data/Dynamic Data/Weight Finding/Visited Truck 2/visited_truck2_{w1}_{w2}.csv')
v3 = pd.read_csv(f'Data/Dynamic Data/Weight Finding/Visited Truck 3/visited_truck3_{w1}_{w2}.csv')
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
stats.to_csv('Data/Dynamic Data/Weight Finding/Statistics.csv')