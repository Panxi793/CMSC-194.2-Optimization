import os
import csv

# Set image size (replace with your actual image resolution)
IMAGE_WIDTH = 1904
IMAGE_HEIGHT = 1071
input_file = "uav0000342_04692_v.txt"
output_dir = "yolo_labels"

os.makedirs(output_dir, exist_ok=True)

with open(input_file, 'r') as f:
    reader = csv.reader(f)
    frame_dict = {}

    for row in reader:
        frame_id = int(row[0])
        x, y, w, h = map(float, row[2:6])
        class_id = int(row[7])

        x_center = (x + w / 2) / IMAGE_WIDTH
        y_center = (y + h / 2) / IMAGE_HEIGHT
        w_norm = w / IMAGE_WIDTH
        h_norm = h / IMAGE_HEIGHT

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
        frame_dict.setdefault(frame_id, []).append(yolo_line)

for frame_id, lines in frame_dict.items():
    output_path = os.path.join(output_dir, f"{frame_id:07d}.txt")
    with open(output_path, 'w') as f:
        f.writelines(lines)

print(f"Converted to YOLO format in '{output_dir}'")
