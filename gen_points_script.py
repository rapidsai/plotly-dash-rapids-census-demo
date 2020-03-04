import cudf
import pandas as pd
import numpy as np
import random
import time
import datetime
import os
import geopandas as gpd
from shapely.geometry import Point
import sys

def random_points_in_polygon(number, polygon):
    points_x = np.array([])
    points_y = np.array([])
    min_x, min_y, max_x, max_y = polygon.bounds
    i= 0
    while i < number:
        point_x = random.uniform(min_x, max_x)
        point_y = random.uniform(min_y, max_y)
        if polygon.contains(Point(point_x, point_y)):
            points_x = np.append(points_x, point_x)
            points_y = np.append(points_y, point_y)
            i += 1
    return points_x, points_y # returns list of points(lat), list of points(long)

#read block data
df = cudf.read_csv('/home/ajay/data/census/all_us_block_level_population_census2010/nhgis0014_csv/nhgis0014_ds172_2010_block.csv', usecols=['GISJOIN', 'H7V001' ,'STATEA'])

df.GISJOIN = df.GISJOIN.str.replace('G', '').astype('int')
df.STATEA = df.STATEA.astype('int')

states = {
1 :"AL",
2 :"AK",
4 :"AZ",
5 :"AR",
6 :"CA",
8 :"CO",
9 :"CT",
10:"DE",
11:"DC",
12:"FL",
13:"GA",
15:"HI",
16:"ID",
17:"IL",
18:"IN",
19:"IA",
20:"KS",
21:"KY",
22:"LA",
23:"ME",
24:"MD",
25:"MA",
26:"MI",
27:"MN",
28:"MS",
29:"MO",
30:"MT",
31:"NE",
32:"NV",
33:"NH",
34:"NJ",
35:"NM",
36:"NY",
37:"NC",
38:"ND",
39:"OH",
40:"OK",
41:"OR",
42:"PA",
44:"RI",
45:"SC",
46:"SD",
47:"TN",
48:"TX",
49:"UT",
50:"VT",
51:"VA",
53:"WA",
54:"WV",
55:"WI",
56:"WY",
72:"PR"
}


def generate_data(state, df_temp, gpdf):
    t1 = datetime.datetime.now()
    gis_index_df = df_temp.index.to_array()
    final_points_x = np.array([])
    final_points_y = np.array([])
    gisjoin = np.array([])

    for index, row in gpdf.iterrows():
        points_x = np.array([])
        points_y = np.array([])
        gisjoin_temp = np.array([])
        if row['GISJOIN'] in gis_index_df and df_temp.loc[row['GISJOIN']]>0:
            num_points = df_temp.loc[row['GISJOIN']]
            polygon = row['geometry']
            if polygon is not None:
                points_x, points_y = random_points_in_polygon(num_points, polygon)
                gisjoin_temp = np.array([row['GISJOIN']]*len(points_x))
                gisjoin = np.append(gisjoin, gisjoin_temp)
                final_points_x = np.append(final_points_x, points_x)
                final_points_y = np.append(final_points_y, points_y)
                print('Processing '+str(state)+' - Completed:', "{0:0.2f}".format((index/len(gpdf))*100), '%', end='')
                print('', end='\r')

    print('Processing for '+str(state)+' complete \n total time', datetime.datetime.now() - t1)
    
    df_fin = cudf.DataFrame({'GISJOIN': gisjoin,'x': final_points_x, 'y':final_points_y})
    df_fin.GISJOIN = df_fin.GISJOIN.astype('int').astype('str')
    df_fin.to_csv('./census_data/population_'+str(state)+'.csv', index=False)

def exec_data(state_key_list):
    for i in state_key_list:
        if i< 10:
            i_str = '0'+str(i)
        else:
            i_str = str(i)
        path = '/home/ajay/data/census/all_us_block_level_shape_file/nhgis0015_shapefile_tl2010_%s0_block_2010/%s_block_2010.shp'%(i_str,states[i])
        print("started reading shape file for state ", states[i])
        if os.path.isfile(path):    
            gpdf = gpd.read_file(path)[['GISJOIN', 'geometry']]
            gpdf.GISJOIN = gpdf.GISJOIN.str.replace('G', '').astype('int')
            print("completed reading shape file for state ", states[i])
            df_temp = df.query('STATEA == @i')[['GISJOIN', 'H7V001']]
            df_temp.index = df_temp.GISJOIN
            df_temp = df_temp['H7V001']
            print("starting to generate data for "+str(states[i])+"... ")
            generate_data(states[i], df_temp, gpdf)
            del(df_temp)
        else:
            print("shape file does not exist")
            continue

if __name__ == "__main__":
    exec_data(eval(sys.argv[1]))