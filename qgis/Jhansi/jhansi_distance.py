import numpy as np
import pandas as pd

df = pd.read_csv('C:\\Users\\Meet\\Documents\\Internship\\bin_locations.csv')
df = df.drop([df.columns[0], df.columns[-1]], axis = 1)
mat = np.zeros((df.shape[0],df.shape[0]))
print(df.shape, mat.shape)
##
#count = 0
for i in range(df.shape[0]):
    x1 = df.iloc[i].to_list()
    for j in range(df.shape[0]):
        print(i,j)
        if i == j:
            mat[i,j] = 0.0
        else:
            x2 = df.iloc[j].to_list()
            st_point = str(x1[0]) + ', ' + str(x1[1]) + ' [EPSG:3857]'
            ed_point = str(x2[0]) + ', ' + str(x2[1]) + ' [EPSG:3857]'
#            params = {'INPUT':'jhansi_road_map [EPSG:4326]','STRATEGY':0,'DIRECTION_FIELD':'','VALUE_FORWARD':'','VALUE_BACKWARD':'','VALUE_BOTH':'','DEFAULT_DIRECTION':2,'SPEED_FIELD':'','DEFAULT_SPEED':50,'TOLERANCE':0,'START_POINT':st_point,'END_POINT':ed_point,'OUTPUT':'memory:'}
            params = {
            'DEFAULT_DIRECTION' : 2,
            'DEFAULT_SPEED' : 50,
            'DIRECTION_FIELD' : 'oneway',
            'END_POINT' : ed_point,
            'INPUT' : 'C:/Users/Meet/Documents/Internship/qgis/Jhansi/jhansi_road_map.kml|layername=jhansi_road_map|geometrytype=LineString',
            'OUTPUT' : 'TEMPORARY_OUTPUT',
            'SPEED_FIELD' : '',
            'START_POINT' : st_point,
            'STRATEGY' : 0,
            'TOLERANCE' : 0,
            'VALUE_BACKWARD' : '',
            'VALUE_BOTH' : '',
            'VALUE_FORWARD' : ''
            }
            algresult = processing.run("native:shortestpathpointtopoint", params)
            buff = algresult['OUTPUT']
            mat[i,j] = buff.getFeature(1).attributes()[2]
#x1 = df.iloc[1].to_list()
#x2 = df.iloc[5].to_list()
#st_point = str(x1[0]) + ', ' + str(x1[1]) + ' [EPSG:3857]'
#ed_point = str(x2[0]) + ', ' + str(x2[1]) + ' [EPSG:3857]'
##            params = {'INPUT':'jhansi_road_map [EPSG:4326]','STRATEGY':0,'DIRECTION_FIELD':'','VALUE_FORWARD':'','VALUE_BACKWARD':'','VALUE_BOTH':'','DEFAULT_DIRECTION':2,'SPEED_FIELD':'','DEFAULT_SPEED':50,'TOLERANCE':0,'START_POINT':st_point,'END_POINT':ed_point,'OUTPUT':'memory:'}
#params = { 'DEFAULT_DIRECTION' : 2, 'DEFAULT_SPEED' : 50, 'DIRECTION_FIELD' : 'oneway', 'END_POINT' : ed_point, 'INPUT' : 'C:/Users/Meet/Documents/Internship/qgis/Jhansi/jhansi_road_map.kml|layername=jhansi_road_map|geometrytype=LineString', 'OUTPUT' : 'TEMPORARY_OUTPUT', 'SPEED_FIELD' : '', 'START_POINT' : st_point, 'STRATEGY' : 0, 'TOLERANCE' : 0, 'VALUE_BACKWARD' : '', 'VALUE_BOTH' : '', 'VALUE_FORWARD' : '' }
#algresult = processing.run("native:shortestpathpointtopoint", params)
#print(algresult)
#f = open('result.txt', 'w')
#f.write('A line')
#f.close()
ddf = pd.DataFrame(mat)
ddf.to_csv('C:\\Users\\Meet\\Documents\\Internship\\distance.csv', index = False)
print(ddf.head())