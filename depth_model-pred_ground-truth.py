import cv2
import numpy as np
import pandas as pd
import ast
import os
import json
from shapely.geometry import Polygon , mapping

folder = os.listdir("runs")

# Iterate over the folders in the "runs" directory.
for count , base_dir in enumerate(folder):
    if base_dir == '.DS_Store': continue
    
    # Read the csv file 
    df = pd.read_csv(path + "//results_.csv")
    path = os.path.join("runs" , base_dir)  
    mask_path = os.path.join(path , "class//masks//") 
    grnd_path = "ground_labels//ground_labels//"
    rgb_image_path = os.path.join(path , "class//rgb_images//") 
    depth_image_path = os.path.join(path , "class//depth_images//") 
    
    # Read file lists
    mask_list = os.listdir(mask_path)
    grnd_list = os.listdir(grnd_path)
    rgb_image_list = os.listdir(rgb_image_path)
    depth_image_list = os.listdir(depth_image_path)
    
    # Iterate through corresponding files in each folder
    for i , (depth_img_file , mask_file , rgb_img_file , grnd_file)  in enumerate(zip(depth_image_list , mask_list , rgb_image_list , grnd_list)):
        img_name = os.path.basename(depth_img_file).split('.')[0]
        
        # Read depth image
        depth_image = cv2.imread(depth_image_path + depth_img_file, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            print(f"Error loading image: {depth_img_file}")
            continue
        
        # Extract depth values
        depth_values = depth_image

        # Convert the ground_depth_list in the csv file, which is in the string format to list
        list_result = ast.literal_eval(df['ground_depth_list'][i])
        k = int(df['height_mm'][i])
        
        # Get the average value of all the elements in the list
        list_avg = int(sum(list_result)/len(list_result))
        # Subtract the list average value with the height of the box
        depth_value_avg = list_avg - k
        
        mask_file_path = mask_path + mask_file
        # Read mask file
        with open(mask_file_path , 'r') as json_file:
            data = json.load(json_file)
            
        # Extract bounding box coordinates
        box_coordinates = data['box']
        min_x, min_y = max_x, max_y = box_coordinates[0]

        # Get the bounding box coordinates from the box coordinates 
        for x, y in box_coordinates:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        bounding_box = [(min_x, min_y), (max_x, max_y)]
        
        # Create a binary mask
        height, width = 720, 1280
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Add a padding of 20 to the bounding box coordinates and compare the depth of these pixels with the depth we get from the depth image 
        for y in range(min_y - 20 , max_y + 20):
            for x in range(min_x - 20 , max_x + 20):
                depth_value = depth_values[y, x]
                flag = int(k * (25/54))
                if depth_value in range(depth_value_avg - flag, depth_value_avg + flag):
                    mask[y, x] = 255
                    
        rgb_img_path = rgb_image_path + rgb_img_file 
        # Read rgb image
        rgb_image = cv2.imread(rgb_img_path)  
        img_height, img_width = rgb_image.shape[:2]
        
        bw_image = mask
        kernel = np.ones((10, 10), np.uint8)
        bw_image = cv2.dilate(bw_image, kernel, iterations=1)
        bw_image = cv2.erode(bw_image, kernel, iterations=1)
        contours, _ = cv2.findContours(mask , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        grnd_file_path = grnd_path + img_name + ".txt"
        # Read the ground truth txt file and store the values in a variable
        with open(grnd_file_path, 'r') as file2:
            lines = file2.readline()
    
            line = lines.strip()
            coordinates2 = line.split(' ')
            del coordinates2[0]
            polygon2 = [(float(coordinates2[i])*img_width,float(coordinates2[i+1])*img_height) for i in range(0,len(coordinates2),2)]
            polygon2 = Polygon(polygon2)
            vertices = [list(mapping(polygon2)["coordinates"][0])]
            vertices = np.array(vertices, np.int32)
        
        # Plot the contours on the rgb image
        cv2.drawContours(rgb_image, contours, -1, (0, 255, 0), 2) 
        # Plot the masks we get from the model predictions on the rgb image
        cv2.drawContours(rgb_image, [np.array(box_coordinates)], -1, (0, 0, 255), 2) 
        # Plot the ground truth masks on the rgb image
        cv2.polylines(rgb_image, [np.array(vertices, dtype=np.int32)], isClosed=True, color=(255, 255, 255), thickness=2) 
        
        output_path = "mask_grnd_depth_folder//"
        os.makedirs(output_path , exist_ok=True)
        
        # Save the output image
        cv2.imwrite(output_path + img_name + ".jpg" , rgb_image)
        print(f"Saved image {i + 1} from folder {count}")
    print(f"Successfully saved images in folder {count}")