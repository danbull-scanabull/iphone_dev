import torch.optim as optim
from torchvision import datasets, transforms

from pyntcloud import PyntCloud
import time
import csv
from torch.utils.data import Subset
#from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
import numpy as np
import os
import torch
from z_order import xyz2key as z_order_encode_
import coremltools as coreml
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#from model import *

dfnames=['x', 'y', 'z', 'red','green','blue']
num_pts=1024

#data = '/home/ubuntu/data/iphone_data/iphone/2025-04-08_012436_from_orig_rotated_moved_.ply'
#data = '/home/ubuntu/data/iphone_data/iphone/2025-04-08_012436_from_orig_rotated_moved_.ply'


dsj = True
if dsj:
    output_dir = "/Users/daniel/Documents/Work/Jobs/Scanabull/code/Python/dsj/test_output/"
    saved_model = "/Users/daniel/Documents/Work/Jobs/Scanabull/code/Python/dsj/"
    complete_model = os.path.join(saved_model, 'model_v30.mlpackage')
    model = coreml.models.MLModel(complete_model, compute_units=coreml.ComputeUnit.CPU_ONLY)
else:
    output_dir = '/home/ubuntu/data/iphone_data/iphone/output/'
    saved_model = '/home/ubuntu/models'
    complete_model = os.path.join(saved_model, 'complete_model_30.pt')
    model = torch.jit.load(complete_model)
    #print (model)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()


files_to_process = [{
                'filepath': "test_data/a_b_testing/2025-04-08_012436_from_orig_rotated_moved_.ply",
                'outputs': []
            },
            {
                'filepath': "test_data/a_b_testing/daniel-iphone-output-2025-04-08_012436_from_orig_rotated_moved_.ply",
                'outputs': []
            }]
    

def save_ply(array, filepath, stage):
    df_x = pd.DataFrame(array)
    df_x.columns = dfnames

    ptcloud_x = PyntCloud(df_x)
    ptcloud_x.points['red'] = np.asarray(array[:,3], dtype=np.uint8)
    ptcloud_x.points['green'] = np.asarray(array[:,4], dtype=np.uint8)
    ptcloud_x.points['blue'] = np.asarray(array[:,5], dtype=np.uint8)

    filename = os.path.basename(filepath)
    output_name = f"{stage}_{filename}"
    output_path = os.path.join(output_dir, output_name)
    ptcloud_x.to_file(output_path)


def z_order_encode(coord: torch.Tensor, depth: int = 16):
    x, y, z = coord[:, 0].long(), coord[:, 1].long(), coord[ :, 2].long()
    # we block the support to batch, maintain batched code in Point class
    code = z_order_encode_(x, y, z, b=None, depth=depth)
    return code

class ReOrder(object):
    def __init__(self, num_pts=num_pts):
        self.num_pts = num_pts

    def __call__(self, data):
        depth = int(data.max()).bit_length()
        code = z_order_encode(torch.from_numpy(data), depth=depth)
        ind = np.argsort(code)
        data = data[ind]
        return data
# model = LiDARTransformer(finetune=True)
# ft_weights = os.path.join(saved_model, 'ft_model_30.pt')
# model.load_state_dict(torch.load(ft_weights, weights_only=True))

def process(filepath_dict, seed):
    filepath = filepath_dict['filepath']

    #load data
    cloud = PyntCloud.from_file(filepath)
    centroid = cloud.centroid
    array = np.asarray(cloud.points)[:,:6].astype(np.float32)

    #pick 1024 points
    np.random.seed(seed)
    index = np.random.choice(array.shape[0], num_pts, replace=False)  
    array = array[index]

    ##centralise
    array[:,:3] = np.subtract(array[:,:3],centroid)
    #save_ply(array, output, f'{seed}_')

    ##reorder with z values
    reorder = ReOrder()
    array = reorder(array)
    #save_ply(array, output, 'reorder')

    ##scale data (as per model)
    array[:,:3] = array[:,:3]/90
    #array[:,2] = 0
    array[:,3:6] = 0#array[:,3:6]/127.5 -1

    #save_ply(array, output, 'scale')

    ##add batch dim

    if dsj:
        out_dict = model.predict({"input": [array]})
        spec = model.get_spec()
        model_output_name = [out.name for out in spec.description.output][0]
        weight_value = out_dict[model_output_name][0][0]
    else:
        array = np.expand_dims(array,0)
        array = torch.from_numpy(array).to(device="cuda", dtype=torch.float32)

        with torch.no_grad():
            outputs = model(array)
                
            for i in range(len(outputs)):
                weight_value = np.round(outputs.cpu().detach().numpy()[i][0],0)

    filepath_dict['outputs'].append(weight_value)

    #print(filepath)
    #print ("weight is " + str(weight_value))
    #save_ply(array, filepath, f'{weight_value}kg_{seed}')



for seed in range(0, 50):
    for filepath_dict in files_to_process:
        process(filepath_dict, seed)



for filepath_dict in files_to_process: 
    print(filepath_dict['filepath'])
    outputs = np.asarray(filepath_dict['outputs'])

    print(f"Mean: {np.mean(outputs)}")
    print(f"Range: {np.min(outputs)} - {np.max(outputs)}")
    print(f"Std: {np.std(outputs)}")
