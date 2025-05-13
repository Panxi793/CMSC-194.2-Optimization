import os
import re
import glob
import shutil

# Path to the YOLO labels folder to modify
input_labels_dir = "yolo_labels/uav0000143_02250_v" 
output_labels_dir = "fixed_labels/uav0000143_02250_v"

# Create output directory
os.makedirs(output_labels_dir, exist_ok=True)

# Class mapping function - only reduces class ID by 1 (from 1-10 to 0-9)
def fix_class_id(class_id):
    # Convert class_id to int
    class_id = int(class_id)
    
    # Only modify classes 1 and above, leave 0 as is
    if class_id > 0:
        return class_id - 1
    return class_id

# Regular expression to match YOLO format lines
# Format: class_id x_center y_center width height
yolo_pattern = re.compile(r'^(\d+)(\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+)$')

# Process all label files
for label_file in glob.glob(os.path.join(input_labels_dir, "*.txt")):
    # Get just the filename
    filename = os.path.basename(label_file)
    output_file = os.path.join(output_labels_dir, filename)
    
    # Process each file
    with open(label_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
                
            # Match YOLO format
            match = yolo_pattern.match(line)
            if match:
                class_id, rest_of_line = match.groups()
                # Fix class ID only
                new_class_id = fix_class_id(class_id)
                # Write updated line
                outfile.write(f"{new_class_id}{rest_of_line}\n")
            else:
                # If doesn't match pattern, copy as is
                outfile.write(line + "\n")
    
    print(f"Processed {filename}")

print(f"All labels processed. Updated files saved to {output_labels_dir}") 