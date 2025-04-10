from pyntcloud import PyntCloud
import time
import csv
import numpy as np
import glob
import pandas as pd
import os


#background = '/home/ubuntu/data/iphone_data/mix_trials/background.ply'
cow = '/home/ubuntu/data/iphone_data/iphone/2025-04-08_012436_from_orig_rotated.ply'
cow = "/Users/daniel/Documents/Work/Jobs/Scanabull/tmp/2025-04-10_043021-3-std.ply"
out = '/home/ubuntu/data/iphone_data/iphone'
out = "/Users/daniel/Documents/Work/Jobs/Scanabull/tmp"
dfnames=['x', 'y', 'z', 'red','green','blue']

#background_array = np.asarray(PyntCloud.from_file(background).points)
cow_array = np.asarray(PyntCloud.from_file(cow).points)

cow_array = cow_array[(cow_array[:,3]>10),:]
cow_array = cow_array[(cow_array[:,4]>10),:]
cow_array = cow_array[(cow_array[:,5]>10),:]

cow_array[:,0] = cow_array[:,0] *1000
cow_array[:,1] = cow_array[:,1] *-1000 
cow_array[:,2] = cow_array[:,2] *-1000 

##rotate
anglex = -0.15 * np.pi
rot_cos, rot_sin = np.cos(anglex), np.sin(anglex)
rot_t_x = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
cow_array[:,:3] = np.dot(cow_array[:,:3], np.transpose(rot_t_x)).astype(np.float32)

angley = 1 * np.pi
rot_cos, rot_sin = np.cos(angley), np.sin(angley)
rot_t_y = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
cow_array[:,:3] = np.dot(cow_array[:,:3], np.transpose(rot_t_y)).astype(np.float32)


anglez = -0.11 * np.pi
rot_cos, rot_sin = np.cos(anglez), np.sin(anglez)
rot_t_z = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
cow_array[:,:3] = np.dot(cow_array[:,:3], np.transpose(rot_t_z)).astype(np.float32)


cow_array[:,0] = cow_array[:,0] + 500
cow_array[:,1] = cow_array[:,1] + 800
cow_array[:,2] = cow_array[:,2] + 1000

combined = cow_array#np.concatenate((background_array, cow_array), axis=0)

##cow_array[:,3:6] = (255 * (cow_array[:,3:6]/255) ** gamma).astype('uint8')


df_x = pd.DataFrame(combined)

df_x.columns = dfnames

ptcloud_x = PyntCloud(df_x)
ptcloud_x.points['red'] = np.asarray(combined[:,3], dtype=np.uint8)
ptcloud_x.points['green'] = np.asarray(combined[:,4], dtype=np.uint8)
ptcloud_x.points['blue'] = np.asarray(combined[:,5], dtype=np.uint8)

output = os.path.join(out, os.path.basename(cow)[:-4] + '_moved_.ply')
ptcloud_x.to_file(output)