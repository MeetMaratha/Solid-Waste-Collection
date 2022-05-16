import sys
from qgis import processing
st_point = str(73.65) + ', ' + str(30.25) + ' [EPSG:3857]'
ed_point = str(73.67) + ', ' + str(30.75) + ' [EPSG:3857]'
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
processing.run("native:shortestpathpointtopoint", params)
