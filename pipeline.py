import os
import sys
import time

import numpy as np
import pandas as pd
from ultralytics import YOLO
from pyntcloud import PyntCloud
import cv2

seg_model = None
model_path = '/Users/daniel/Documents/Work/Jobs/Scanabull/models/yolo/mlpackage/yolo11l-seg.mlpackage'


def get_segmentation_result(file_paths):
    global seg_model
    seg_model = YOLO(model=model_path, task="segment")
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


def get_points_within_mask(segmentation_mask, file_paths, show=False, save=False):
    mask_points = np.int32([segmentation_mask])

    if show:
        img = cv2.imread(file_paths['image'])
        # img_640 = cv2.resize(img, (640, 640))
        # for mask, box in zip(result.masks.xyn, result.boxes):
        #cv2.polylines(img, points, True, (255, 0, 0), 1)
        cv2.fillPoly(img, mask_points, [255,0,0])
        cv2.imshow("Image", img)
        cv2.waitKey(0)

    # Get Point clouds as numpy
    # 2D Lidar is currently 1920x1440
    pointcloud_2D = PyntCloud.from_file(file_paths['lidar_2D'])
    pointcloud_2D_points = pointcloud_2D.points.to_numpy()

    # Get 3D Lidar Point cloud
    pointcloud_3D = PyntCloud.from_file(file_paths['lidar_3D'])
    pointcloud_3D_points = pointcloud_3D.points.to_numpy()

    cow_array = []

    #for row in pointcloud_2D_points:
    idx = 0
    for row in pointcloud_2D_points:
        #row_x = round(row[0])
        #row_y = round(row[1])
        #row_z = row[2]
        xy = (row[0], row[1])
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
        new_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        row[3] = new_rgb[0][0][0]
        row[4] = new_rgb[0][0][1]
        row[5] = new_rgb[0][0][2]

        pointcloud_3D_points[idx][3] = new_rgb[0][0][0]
        pointcloud_3D_points[idx][4] = new_rgb[0][0][1]
        pointcloud_3D_points[idx][5] = new_rgb[0][0][2]

        idx += 1
    
    cow_points = np.array(cow_array)

    return_obj = {
        'lidar_2D': pointcloud_2D_points,
        'lidar_3D': pointcloud_3D_points,
        'lidar_contained': cow_points
    }

    return return_obj


def run(required_detection_type, input_dir):

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

    for key, filepath in file_paths.items():
        if filepath is None:
            print(f"Cannot find file for {key}")
            sys.exit()

    # Get segmentation result
    segmentation_results = get_segmentation_result(file_paths)

    # Get matching mask from results
    show_results = False
    segmentation_mask = None
    min_confidence = 0.5
    if segmentation_results:
        for result in segmentation_results:
            if show_results:
                result.show()

            boxes = result.boxes
            masks = result.masks
            #probs = result.probs   # Probs object for classification outputs
            #obb = result.obb       # Oriented boxes object for OBB outputs

            for i, box in enumerate(boxes):
                confidence = box.conf
                detection_type = seg_model.names.get(box.cls.item())
                if confidence >= min_confidence and detection_type == required_detection_type:
                    # Relative positions
                    ## segmentation_mask = masks.xyn[i]
                    # Pixel positions
                    segmentation_mask = masks.xy[i]

                    break
    
    # Find points from mask inside lidar data
    points_dict = get_points_within_mask(segmentation_mask, file_paths)

    # Save as various formats
    cwd = os.getcwd()
    out_dir = os.path.join(cwd, 'test_output')
    for build_name, points in points_dict.items():
        pd_points = pd.DataFrame(points)
        pd_points.columns = ['x', 'y', 'z', 'red', 'green', 'blue', 'confidence']
        ptcloud_x = PyntCloud(pd_points)
        ptcloud_x.points['red'] = np.asarray(points[:,3], dtype=np.uint8)
        ptcloud_x.points['green'] = np.asarray(points[:,4], dtype=np.uint8)
        ptcloud_x.points['blue'] = np.asarray(points[:,5], dtype=np.uint8)

        out_name = f"{file_paths['base']}_{build_name}.ply"
        out_path = os.path.join(out_dir, out_name)
        ptcloud_x.to_file(out_path)

    
def run_cli():
    cwd = os.getcwd()
    required_detection_type = "cow"
    input_dir = os.path.join(cwd, 'test_data', 'cow')

    if len(sys.argv) == 2:
        required_detection_type = sys.argv[1]
        input_dir = os.path.join(cwd, 'test_data', required_detection_type)
    if len(sys.argv) == 3:
        required_detection_type = sys.argv[1]
        input_dir = sys.argv[2]

    run(required_detection_type, input_dir)


if __name__ == "__main__":
    run_cli()