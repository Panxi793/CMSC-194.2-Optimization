import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import argparse


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes.
    Boxes are in format [x, y, width, height]"""
    # Convert to [x1, y1, x2, y2] format
    box1_x1, box1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    box1_x2, box1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    
    box2_x1, box2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    box2_x2, box2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    # Calculate intersection area
    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    
    # Calculate IoU
    iou = intersection_area / (box1_area + box2_area - intersection_area)
    return iou


def calculate_object_displacement(obj1, obj2):
    """Calculate Euclidean distance between object centers."""
    return np.sqrt((obj1[0] - obj2[0])**2 + (obj1[1] - obj2[1])**2)


def has_similar_object_nearby(obj, prev_objects, discrimination_distance):
    """Check if the object is close to any object in the previous frame."""
    for prev_obj in prev_objects:
        if calculate_object_displacement(obj, prev_obj) < discrimination_distance:
            return True
    return False


def create_region_around_object(obj, image, region_size):
    """Create a cropped region around the object."""
    img_height, img_width = image.shape[:2]
    
    # Get object center and dimensions
    x_center, y_center, width, height = obj
    
    # Calculate region dimensions maintaining aspect ratio
    aspect_ratio = region_size[0] / region_size[1]
    region_width = min(region_size[0], img_width)
    region_height = min(region_size[1], img_height)
    
    # Calculate region boundaries
    x1 = max(0, int(x_center - region_width/2))
    y1 = max(0, int(y_center - region_height/2))
    
    # Adjust if region exceeds image boundaries
    if x1 + region_width > img_width:
        x1 = img_width - region_width
    if y1 + region_height > img_height:
        y1 = img_height - region_height
    
    # Ensure x1, y1 aren't negative
    x1 = max(0, x1)
    y1 = max(0, y1)
    
    # Extract the region
    region = image[y1:y1+region_height, x1:x1+region_width].copy()
    
    return region, (x1, y1, region_width, region_height)


def calculate_overlap_percentage(obj, region_coords):
    """Calculate what percentage of the object is within the region."""
    region_x, region_y, region_w, region_h = region_coords
    obj_x, obj_y, obj_w, obj_h = obj
    
    # Convert to [x1, y1, x2, y2] format
    obj_x1, obj_y1 = obj_x - obj_w/2, obj_y - obj_h/2
    obj_x2, obj_y2 = obj_x + obj_w/2, obj_y + obj_h/2
    
    region_x2, region_y2 = region_x + region_w, region_y + region_h
    
    # Calculate intersection area
    x_overlap = max(0, min(obj_x2, region_x2) - max(obj_x1, region_x))
    y_overlap = max(0, min(obj_y2, region_y2) - max(obj_y1, region_y))
    
    intersection_area = x_overlap * y_overlap
    obj_area = obj_w * obj_h
    
    return intersection_area / obj_area if obj_area > 0 else 0.0


def transform_bbox_to_cropped_region(obj, region_coords):
    """Transform object coordinates to the cropped region's coordinate system."""
    region_x, region_y, _, _ = region_coords
    obj_x, obj_y, obj_w, obj_h = obj
    
    # Adjust center coordinates
    new_x = obj_x - region_x
    new_y = obj_y - region_y
    
    return [new_x, new_y, obj_w, obj_h]


def blur_object_in_region(region, obj, blur_strength=15):
    """Apply Gaussian blur to an object in the region."""
    # Get object boundaries
    x, y, w, h = obj
    x1, y1 = int(x - w/2), int(y - h/2)
    x2, y2 = int(x + w/2), int(y + h/2)
    
    # Make sure coordinates are within region bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(region.shape[1], x2)
    y2 = min(region.shape[0], y2)
    
    # Create a mask for the object
    object_region = region[y1:y2, x1:x2]
    
    # Apply Gaussian blur
    if object_region.size > 0:
        blurred = cv2.GaussianBlur(object_region, (blur_strength, blur_strength), 0)
        region[y1:y2, x1:x2] = blurred
    
    return region


def load_yolo_labels(label_file, img_width, img_height):
    """Load YOLO format labels from a file.
    Returns list of [class_id, x, y, w, h] where x,y,w,h are absolute values."""
    if not os.path.exists(label_file):
        return []
    
    objects = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                # YOLO format: [x_center, y_center, width, height] normalized
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                objects.append([class_id, x_center, y_center, width, height])
    
    return objects


def save_yolo_labels(objects, output_file, img_width, img_height):
    """Save objects in YOLO format.
    Objects are in format [class_id, x, y, w, h] with absolute values."""
    with open(output_file, 'w') as f:
        for obj in objects:
            class_id, x, y, w, h = obj
            # Convert to normalized coordinates
            x_norm = x / img_width
            y_norm = y / img_height
            w_norm = w / img_width
            h_norm = h / img_height
            
            f.write(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")


def preprocess_dataset(dataset_path, output_path, config):
    """Main preprocessing function."""
    # Configuration
    region_size = config['region_size']
    discrimination_distance = config['discrimination_distance']
    min_area_percentage = config['min_area_percentage']
    key_frame_interval = config['key_frame_interval']
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels'), exist_ok=True)
    
    # Get image files sorted by frame number
    image_files = sorted([f for f in os.listdir(os.path.join(dataset_path, 'images')) 
                         if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    previous_objects = None
    frame_count = 0
    total_regions_created = 0
    
    # Process each frame in sequence
    for img_file in image_files:
        frame_count += 1
        img_path = os.path.join(dataset_path, 'images', img_file)
        label_path = os.path.join(dataset_path, 'labels', os.path.splitext(img_file)[0] + '.txt')
        
        # Load image and its objects
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        img_height, img_width = image.shape[:2]
        objects = load_yolo_labels(label_path, img_width, img_height)
        
        # Phase 1: Discard objects based on movement
        selected_objects = []
        
        # For key frames or the first frame, keep all objects
        is_key_frame = frame_count % key_frame_interval == 1
        print(f"Processing frame {frame_count}: {img_file} {'(KEY FRAME)' if is_key_frame else ''}")
        
        if is_key_frame or previous_objects is None:
            selected_objects = [obj for obj in objects]
        else:
            # For non-key frames, only keep objects that have moved
            for obj in objects:
                obj_pos = [obj[1], obj[2]]  # x, y positions
                if not has_similar_object_nearby(obj_pos, [[o[1], o[2]] for o in previous_objects], discrimination_distance):
                    selected_objects.append(obj)
        
        print(f"  Selected {len(selected_objects)} objects out of {len(objects)}")
        
        # Phase 2: Crop regions and relabel
        selectable_objects = set(range(len(objects)))  # Initially, all objects are selectable
        
        for obj_idx, obj in enumerate(selected_objects):
            obj_pos = [obj[1], obj[2], obj[3], obj[4]]  # x, y, width, height
            
            # Create region around object
            region, region_coords = create_region_around_object(obj_pos, image, region_size)
            region_height, region_width = region.shape[:2]
            
            # Process all objects within the region
            region_objects = []
            
            for i, region_obj in enumerate(objects):
                if i not in selectable_objects:
                    continue  # Skip already processed objects
                
                region_obj_pos = [region_obj[1], region_obj[2], region_obj[3], region_obj[4]]
                overlap = calculate_overlap_percentage(region_obj_pos, region_coords)
                
                if overlap >= 0.99:  # Object fully within region
                    # Transform object coordinates to region's coordinate system
                    new_obj = transform_bbox_to_cropped_region(region_obj_pos, region_coords)
                    region_objects.append([region_obj[0], new_obj[0], new_obj[1], new_obj[2], new_obj[3]])
                    selectable_objects.remove(i)  # Mark as not selectable
                    
                elif overlap > min_area_percentage:  # >50% of object in region
                    new_obj = transform_bbox_to_cropped_region(region_obj_pos, region_coords)
                    region_objects.append([region_obj[0], new_obj[0], new_obj[1], new_obj[2], new_obj[3]])
                    
                elif overlap > 0:  # <50% of object in region but still overlapping
                    # Blur the object in the region
                    new_obj = transform_bbox_to_cropped_region(region_obj_pos, region_coords)
                    region = blur_object_in_region(region, new_obj)
            
            # Save cropped region as new image if it contains objects
            if region_objects:
                region_filename = f"{os.path.splitext(img_file)[0]}_region{obj_idx}.jpg"
                region_img_path = os.path.join(output_path, 'images', region_filename)
                region_label_path = os.path.join(output_path, 'labels', f"{os.path.splitext(region_filename)[0]}.txt")
                
                cv2.imwrite(region_img_path, region)
                save_yolo_labels(region_objects, region_label_path, region_width, region_height)
                total_regions_created += 1
        
        previous_objects = objects
    
    print(f"\nPreprocessing complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total regions created: {total_regions_created}")
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset for object detection.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset with images/ and labels/ folders')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the preprocessed dataset')
    parser.add_argument('--region_width', type=int, default=640, help='Width of the cropped regions')
    parser.add_argument('--region_height', type=int, default=360, help='Height of the cropped regions')
    parser.add_argument('--discrimination_distance', type=int, default=10, help='Distance threshold for object movement')
    parser.add_argument('--min_area_percentage', type=float, default=0.5, help='Minimum percentage of object area to include')
    parser.add_argument('--key_frame_interval', type=int, default=7, help='Interval between key frames')
    
    args = parser.parse_args()
    
    config = {
        'region_size': (args.region_width, args.region_height),
        'discrimination_distance': args.discrimination_distance,
        'min_area_percentage': args.min_area_percentage,
        'key_frame_interval': args.key_frame_interval
    }
    
    preprocess_dataset(args.dataset_path, args.output_path, config)


if __name__ == '__main__':
    main()