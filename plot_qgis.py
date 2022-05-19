import numpy as np
import pandas as pd

path1 = [(0, 57),(25, 0),(20, 25),(10, 96),(73, 20),(33, 36),(62, 33),(6, 26),(57, 22),(22, 28),(36, 42),(43, 92),(42, 73),(92, 62),(96, 6),(26, 43),(28, 10)]
df = pd.read_csv('C:\\Users\\Meet\\Documents\\Internship\\Final\\bin_locations.csv')
for i in path1:
    st_point = str(round(df.loc[i[0], 'x'], 6)) + ', ' + str(round(df.loc[i[0], 'y'], 6)) + ' [EPSG:4326]'
    ed_point = str(round(df.loc[i[1], 'x'], 6)) + ', ' + str(round(df.loc[i[1], 'y'], 6)) + ' [EPSG:4326]'
    params = {
    'DEFAULT_DIRECTION' : 2,
    'DEFAULT_SPEED' : 50,
    'DIRECTION_FIELD' : '',
    'END_POINT' : ed_point,
    'INPUT' : 'C:\\Users\\Meet\\Documents\\Internship\\qgis\\Chandigarh\\chandigarh_roads.kml|layername=chandigarh_roads',
    'OUTPUT' : 'TEMPORARY_OUTPUT',
    'SPEED_FIELD' : '',
    'START_POINT' : st_point,
    'STRATEGY' : 0,
    'TOLERANCE' : 0,
    'VALUE_BACKWARD' : '',
    'VALUE_BOTH' : '',
    'VALUE_FORWARD' : ''
    }
    processing.run("native:shortestpathpointtopoint", params)
    