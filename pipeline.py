import os
import sys
import time

import numpy as np
import pandas as pd
from ultralytics import YOLO
from pyntcloud import PyntCloud
import cv2
import json
from z_order import xyz2key as z_order_encode_
import torch
import coremltools as coreml
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

models_path = "/Users/daniel/Documents/Work/Jobs/Scanabull/models/"

seg_model = None
seg_model_name = "yolo11l-seg.pt"
seg_model_ext = os.path.splitext(seg_model_name)[1][1:]
seg_model_path = os.path.join(models_path, "seg_models", "yolo", seg_model_ext, seg_model_name)

#weigh_model_name = "model_v40.mlpackage"
weigh_model_name = "model_v30_export_v1.mlpackage"
#weigh_model_name = "model_v33_export_v1.mlpackage"
weigh_model_ext = os.path.splitext(weigh_model_name)[1][1:]
weigh_model_path = os.path.join(models_path, "weigh_models", weigh_model_name)

weigh_model_name_pt = "complete_model_30.pt"
weigh_model_ext_pt = os.path.splitext(weigh_model_name_pt)[1][1:]
weigh_model_path_pt = os.path.join(models_path, "weigh_models", weigh_model_name_pt)



show_results = True
show_all_results = False
save_results = True
min_confidence = 0.7


def get_segmentation_model():
    global seg_model
    if seg_model is None:
        seg_model = YOLO(model=seg_model_path, task="segment")
    return seg_model


def get_segmentation_result(file_paths):
    seg_model = get_segmentation_model()
    results = seg_model.predict(source=file_paths['image'])
    
    '''A class for storing and manipulating detection masks.

    This class extends BaseTensor and provides functionality for handling segmentation masks,
    including methods for converting between pixel and normalized coordinates.

    Attributes:
        data (torch.Tensor | numpy.ndarray): The raw tensor or array containing mask data.
        orig_shape (tuple): Original image shape in (height, width) format.
        xy (List[numpy.ndarray]): A list of segments in pixel coordinates.
        xyn (List[numpy.ndarray]): A list of normalized segments.

    Methods:
        cpu(): Returns a copy of the Masks object with the mask tensor on CPU memory.
        numpy(): Returns a copy of the Masks object with the mask tensor as a numpy array.
        cuda(): Returns a copy of the Masks object with the mask tensor on GPU memory.
        to(*args, **kwargs): Returns a copy of the Masks object with the mask tensor on specified device and dtype.

    Examples:
        >>> masks_data = torch.rand(1, 160, 160)
        >>> orig_shape = (720, 1280)
        >>> masks = Masks(masks_data, orig_shape)
        >>> pixel_coords = masks.xy
        >>> normalized_coords = masks.xyn'''
    return results

def get_segmentation_mask(detection_results, required_detection_type):
    segmentation_box = None
    segmentation_mask = None
    if detection_results:
        for result in detection_results:
            if show_all_results:
                result.show()

            boxes = result.boxes
            masks = result.masks
            #probs = result.probs   # Probs object for classification outputs
            #obb = result.obb       # Oriented boxes object for OBB outputs

            for i, box in enumerate(boxes):
                detection_type = seg_model.names.get(box.cls.item())
                confidence = box.conf
                xy = box.xyxy
                print(f"{detection_type} - Confidence: {confidence}")
                if detection_type == required_detection_type and confidence >= min_confidence:
                    print(f"Selected: {detection_type} - Confidence: {confidence}")
                    # Relative positions
                    ## segmentation_mask = masks.xyn[i]
                    # Pixel positions
                    segmentation_box = box
                    segmentation_mask = masks.xy[i]
                    break
    
    return segmentation_box, segmentation_mask


def get_weigh_result(input_array):
    mlmodel = coreml.models.MLModel(weigh_model_path, compute_units=coreml.ComputeUnit.CPU_ONLY)
    out_dict = mlmodel.predict({"input": [input_array]})
    
    spec = mlmodel.get_spec()
    output_name = [out.name for out in spec.description.output][0]

    out_value = out_dict[output_name][0][0]
    return out_value


def get_weigh_result_pt(input_array):
    model = torch.load(weigh_model_path_pt, device=torch.device('cpu'))
    
    # model.requires_grad_(False)
    #model.eval()

    '''array = torch.from_numpy(input_array).to(device="cpu", dtype=torch.float32)
    with torch.no_grad():
        outputs = model(array)
            
        for i in range(len(outputs)):
            output = np.round(outputs.cpu().detach().numpy()[i][0],0)
            print ("weight is " + str(output))'''

    #return out_value


def get_points_inside_mask(segmentation_mask, file_paths):
    mask_points = np.int32([segmentation_mask])

    # 2D Lidar (1920x1440) as numpy
    pointcloud_2D = PyntCloud.from_file(file_paths['lidar_2D'])
    pointcloud_2D_points = pointcloud_2D.points.to_numpy()

    # 3D Lidar as numpy
    pointcloud_3D = PyntCloud.from_file(file_paths['lidar_3D'])
    pointcloud_3D_points = pointcloud_3D.points.to_numpy()

    cow_array = []

    idx = 0
    for row in pointcloud_2D_points:
        x_point = row[0]
        y_point = row[1]
        #x_point = round(x_point)
        #y_point = round(y_point)
        #row_z = row[2]
        xy = (x_point, y_point)
        lidar_point_inside_mask = cv2.pointPolygonTest(mask_points, xy, False)

        # If lidar point inside seg mask
        saturation_adjustment = -200
        if (lidar_point_inside_mask > 0.0):
            saturation_adjustment = 100
            cow_array.append(pointcloud_3D_points[idx])

        # Adjust saturation by given amount
        # Clip the values to ensure they are within the valid range
        # Get colour for manipulation
        rgb = np.array([[[row[3], row[4], row[5]]]], dtype=np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = cv2.add(hsv[:, :, 1], saturation_adjustment)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

        # Adjust 2D Lidar coords scale
        row[0] = (row[0]/500) * -1
        row[1] = (row[1]/500) * -1
        row[2] = 0

        # Adjust colours
        '''new_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        row[3] = new_rgb[0][0][0]
        row[4] = new_rgb[0][0][1]
        row[5] = new_rgb[0][0][2]

        pointcloud_3D_points[idx][3] = new_rgb[0][0][0]
        pointcloud_3D_points[idx][4] = new_rgb[0][0][1]
        pointcloud_3D_points[idx][5] = new_rgb[0][0][2]'''

        idx += 1
    
    cow_points = np.array(cow_array)

    return_obj = {
        '2D': pointcloud_2D_points,
        '3D': pointcloud_3D_points,
        '1_mask_original': cow_points
    }

    return return_obj

def get_rotated(points, filepaths):
    # Get camera euler angles from json data
    cameraEulerAngles = {}
    with open(filepaths['json']) as f:
        json_file_as_json = json.load(f)
        cameraEulerAngles = json_file_as_json['cameraEulerAngles']

    # Flip all axes
    anglex = cameraEulerAngles[0] * -1
    angley = cameraEulerAngles[1] * -1
    anglez = cameraEulerAngles[2] * -1

    rot_cos, rot_sin = np.cos(anglex), np.sin(anglex)
    rot_t_x = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])

    rot_cos, rot_sin = np.cos(angley), np.sin(angley)
    rot_t_y = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])

    rot_cos, rot_sin = np.cos(anglez), np.sin(anglez)
    rot_t_z = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])

    for axis in ['y', 'x', 'z']:
        if axis == "x":
            _transposed = np.transpose(rot_t_x)
        elif axis == "y":
            _transposed = np.transpose(rot_t_y)
        elif axis == "z":
            _transposed = np.transpose(rot_t_z)
        points[:,:3] = np.dot(points[:,:3], _transposed).astype(np.float32)
    
    return points


def get_standard_deviation_trim(points, std_multipler):
    mean = np.mean(points[:,2])
    std_deviation = np.std(points[:,2]) * std_multipler
    min_std_deviation = mean - std_deviation
    max_std_deviation = mean + std_deviation
    #points = points[~(points[:,2]>= max_std_deviation),:]
    points = points[~(points[:,2]<= min_std_deviation),:]
    return points


def get_random_selection(points, num_points):
    np.random.seed(0)
    index = np.random.choice(points.shape[0], num_points, replace=False)  
    points = points[index]
    return points


def get_centralised_points(points):
    pd_points = pd.DataFrame(points)
    pd_points.columns = ['x', 'y', 'z', 'red', 'green', 'blue', 'confidence']
    ptcloud = PyntCloud(points)
    centroid = ptcloud.centroid # pyntcloud
    #data_dict["centre"] = (pts.min(axis=0) + pts.max(axis=0))/2 #numpy
    points[:,:3] = np.subtract(points[:,:3], centroid)
    return points


def get_z_ordered_points(points):
    points = torch.from_numpy(points)
    depth = int(points.max()).bit_length()
    x, y, z = points[:, 0].long(), points[:, 1].long(), points[ :, 2].long()
    code = z_order_encode_(x, y, z, b=None, depth=depth)
    ind = np.argsort(code)
    points = points[ind]
    return points


def get_draw_image(segmentation_box, segmentation_mask, file_paths, weight_result):
    mask_points = np.in32([segmentation_mask])
    segmentation_box_xy = np.int32(segmentation_box.xyxy[0])

    img = cv2.imread(file_paths['image'])
    overlay = img.copy()

    # Apply mask
    # img_640 = cv2.resize(img, (640, 640))
    # cv2.polylines(img, points, True, (255, 0, 0), 1)
    cv2.fillPoly(overlay, mask_points, [255,0,0])
    mask_alpha = 0.4  # Transparency factor.
    img_new = cv2.addWeighted(overlay, mask_alpha, img, 1 - mask_alpha, 0)

    lt = (segmentation_box_xy[0], segmentation_box_xy[1])
    rb = (segmentation_box_xy[2], segmentation_box_xy[3])

    # Add detection box
    rect_color = (255, 255, 255)
    rect_thickness = 4
    img_new = cv2.rectangle(img_new, lt, rb, rect_color, rect_thickness)

    # Add weight value
    font = cv2.FONT_HERSHEY_SIMPLEX
    org_width = int((rb[0] - lt[0]) * 0.5)
    font_org = (lt[0] + org_width, lt[1] + 150)
    font_scale = 2
    font_color = (255, 255, 255)
    font_thickness = 4
    img_new = cv2.putText(img_new, f"{weight_result}kg", font_org, font, 
                    font_scale, font_color, font_thickness, cv2.LINE_AA)


def run(input_dir):
    cwd = os.getcwd()
    out_dir = os.path.join(cwd, 'test_output')

    # Build filepaths
    file_paths = {
        'base': os.path.basename(input_dir),
        'image': None,
        'lidar_2D': None,
        'lidar_3D': None
    }
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".ply"):
                if 'grid' in file:
                    file_paths['lidar_2D'] = os.path.join(root, file)
                elif 'original' in file:
                    file_paths['lidar_3D'] = os.path.join(root, file)
            if file.endswith(".jpeg"):
                file_paths['image'] = os.path.join(root, file)
            if file.endswith(".json"):
                file_paths['json'] = os.path.join(root, file)


    # Get segmentation result
    segmentation_results = get_segmentation_result(file_paths)

    # Get matching mask from results
    segmentation_box, segmentation_mask = get_segmentation_mask(segmentation_results, "cow")
    if segmentation_mask is None:
        print("ERROR: No mask found")
        sys.exit()

    # Get lidar points from inside mask data
    points_dict = get_points_inside_mask(segmentation_mask, file_paths)


    # Begin transforms
    pts = points_dict['1_mask_original']


    # Remove 0 value Z index values
    print("- Remove 0 Z values")
    print(f"{len(pts)} - # Points initial")
    # pts = (pts[:,:6]).astype(np.float32) # first 6 values
    pts = pts[~(pts[:,2]==0),:]


    # Rotate points
    print(f"- Rotate points")
    pts = get_rotated(pts, file_paths)
    points_dict['2_mask_rotate'] = pts


    # Remove all values with z-index outside std-deviation
    # TODO Review std deviation
    print(f"- Trim points by std deviation 1.5")
    pts = get_standard_deviation_trim(pts, 1.5)
    points_dict['3_mask_std'] = pts
    print(f"{len(pts)} - # Points after removing values outside standard deviation 1.5")

    print(f"- Trim points by std deviation 3.0")
    pts = get_standard_deviation_trim(pts, 3.0)
    points_dict['4_mask_std'] = pts
    print(f"{len(pts)} - # Points after removing values outside standard deviation 3.0")


    # Get random sample 1024 points
    print("- Random selection")
    pts = get_random_selection(pts, 1024)
    points_dict['4_mask_random'] = pts


    # Centralise
    print(f"- Centralise")
    pts = get_centralised_points(pts)
    points_dict['5_mask_centralise'] = pts


    # Z-order
    print("- Z-Order")
    pts = get_z_ordered_points(pts)
    points_dict['6_mask_zorder'] = pts


    print("- Get weight result")
    pts = np.float32(pts)
    pts_without_conf = (pts[:,:6]).astype(np.float32) # first 6 values
    weight_result = get_weigh_result(pts_without_conf)
    #get_weigh_result_pt(pts_without_conf)
    print(f"====== {weight_result}kg ======")

    if show_results or save_results:
        img_draw = get_draw_image(segmentation_box, segmentation_mask, file_paths, weight_result)

        if show_results:
            cv2.imshow("Image", img_draw)
            cv2.waitKey(0)
        if save_results:
            out_name = f"{file_paths['base']}_{seg_model_name}_{weight_result}kg.jpg"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, img_draw)


    # Save as various formats
    points_dict['7_mask_final'] = pts
    if save_results:
        print(f"Save")
        for build_name, points in points_dict.items():
            pd_points = pd.DataFrame(points)
            pd_points.columns = ['x', 'y', 'z', 'red', 'green', 'blue', 'confidence']
            ptcloud_x = PyntCloud(pd_points)
            ptcloud_x.points['red'] = np.asarray(points[:,3], dtype=np.uint8)
            ptcloud_x.points['green'] = np.asarray(points[:,4], dtype=np.uint8)
            ptcloud_x.points['blue'] = np.asarray(points[:,5], dtype=np.uint8)

            out_name = f"{file_paths['base']}_{build_name}_{seg_model_name}_{weight_result}kg.ply"
            out_path = os.path.join(out_dir, out_name)
            ptcloud_x.to_file(out_path)

    
def run_cli():
    cwd = os.getcwd()
    input_dir = os.path.join(cwd, 'test_data', "cow")

    if len(sys.argv) == 2:
        input_dir = sys.argv[1]

    run(input_dir)


if __name__ == "__main__":
    run_cli()