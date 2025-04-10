from pyntcloud import PyntCloud
import time
import csv
import numpy as np
import glob
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os



out = '/home/ubuntu/data/iphone_data/iphone/'
dfnames=['x', 'y', 'z', 'red','green','blue']#, 'class']

##frankton 8/4/25
anglex = -0.3140524
angley = -1.5665827
anglez = -0.020147873
lidar = '/home/ubuntu/data/iphone_data/iphone/2025-04-08_012436_from_orig.ply'

##015056
# anglex = 0.21867225
# angley = 1.0347086
# anglez = -0.027863627
# lidar = '/home/ubuntu/data/iphone_data/corrected/2025-02-25_015056_0.05s_new.ply'
# neg_factor = True

#121508
# anglex = -0.051295612
# angley = 0.08321033
# anglez = -0.083494574
# lidar = '/home/ubuntu/data/iphone_data/corrected/2025-02-24_121508_new.ply'
# neg_factor = False

##015139
# anglex = 0.42166454
# angley = 0.42325234
# anglez = 0.016988242
# lidar = '/home/ubuntu/data/iphone_data/corrected/2025-02-25_015139_0.2s_new.ply'
# neg_factor = True


# ##014711
# anglex = 0.27532446
# angley = 1.5067523
# anglez = 0.041711833
# lidar = '/home/ubuntu/data/iphone_data/corrected/2025-02-25_014711_0.05s_new.ply'

# # #010032
# anglex = 0.30473393
# angley = 0.7526087
# anglez = -0.040030636
# lidar = '/home/ubuntu/data/iphone_data/corrected/2025-02-25_010032_0.2_new.ply'

# # ##012351
# anglex = 0.5657641
# angley = 1.5147032
# anglez = 0.02591664
# lidar = '/home/ubuntu/data/iphone_data/corrected/2025-02-25_012351_0.05s_new.ply'

##014550
# anglex = 0.23958008
# angley = 0.13259323
# anglez = -0.0079216445
# lidar = '/home/ubuntu/data/iphone_data/corrected/2025-02-25_014550_0.05s_new.ply'

# anglex = 0.07311985
# angley = -0.4735857
# anglez = 0.005345105
# lidar = '/home/ubuntu/data/iphone_data/corrected/2025-03-11_100658_0.05s.ply'


cow_array = np.asarray(PyntCloud.from_file(lidar).points)

        
rot_cos, rot_sin = np.cos(anglex), np.sin(anglex)
rot_t_x = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])

rot_cos, rot_sin = np.cos(angley), np.sin(angley)
rot_t_y = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        
rot_cos, rot_sin = np.cos(anglez), np.sin(anglez)
rot_t_z = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])

print ("before")
print(cow_array[:,0].min(), cow_array[:,0].max())
print(cow_array[:,1].min(), cow_array[:,1].max())
print(cow_array[:,2].min(), cow_array[:,2].max())





cow_array[:,:3] = np.dot(cow_array[:,:3], np.transpose(rot_t_y)).astype(np.float32)

cow_array[:,:3] = np.dot(cow_array[:,:3], np.transpose(rot_t_x)).astype(np.float32)
#cow_array[:,:3] = np.dot(cow_array[:,:3], np.transpose(rot_t_z)).astype(np.float32)

##yzx
order = '_rotated'

cow_array = cow_array[(cow_array[:,2]>-5),:]
print ("after")
print(cow_array[:,0].min(), cow_array[:,0].max())
print(cow_array[:,1].min(), cow_array[:,1].max())
print(cow_array[:,2].min(), cow_array[:,2].max())

df_x = pd.DataFrame(cow_array)

df_x.columns = dfnames

ptcloud_x = PyntCloud(df_x)
ptcloud_x.points['red'] = np.asarray(cow_array[:,3], dtype=np.uint8)
ptcloud_x.points['green'] = np.asarray(cow_array[:,4], dtype=np.uint8)
ptcloud_x.points['blue'] = np.asarray(cow_array[:,5], dtype=np.uint8)

output = os.path.join(out, os.path.basename(lidar)[:-4] + order + '.ply')
ptcloud_x.to_file(output)

