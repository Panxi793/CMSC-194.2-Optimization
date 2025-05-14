#!/usr/bin/env python3
import os
import subprocess
import sys
import yaml
import shutil
import glob

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.yaml')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

def check_yolov5_repo():
    """Check if YOLOv5 repository exists, clone if it doesn't"""
    repo_path = os.path.join(os.path.dirname(SCRIPT_DIR), 'yolov5_repo')
    
    if not os.path.exists(repo_path):
        print("YOLOv5 repository not found. Cloning...")
        subprocess.run(
            ['git', 'clone', 'https://github.com/ultralytics/yolov5.git', repo_path],
            check=True
        )
        # Install requirements
        subprocess.run(
            ['pip', 'install', '-r', os.path.join(repo_path, 'requirements.txt')],
            check=True
        )
    else:
        print("YOLOv5 repository found.")
    
    return repo_path

def load_config():
    """Load configuration from the YAML file"""
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
    return config

def find_datasets():
    """Find dataset directories and verify they exist"""
    # Preprocessed dataset for training
    train_root = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'datasets', 'preprocessed_dataset'))
    train_dirs = glob.glob(os.path.join(train_root, 'dataset*'))
    
    # Original dataset for validation
    val_root = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'datasets', 'drone_dataset'))
    val_dirs = glob.glob(os.path.join(val_root, 'dataset*'))
    
    if not train_dirs:
        raise FileNotFoundError(f"No dataset directories found in {train_root}")
    
    if not val_dirs:
        raise FileNotFoundError(f"No validation dataset directories found in {val_root}")
    
    return train_dirs, val_dirs

def create_dataset_yaml():
    """Create a proper YOLOv5 dataset YAML file"""
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Get dataset directories
    train_dirs, val_dirs = find_datasets()
    
    # Load class names from config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        class_names = config['names']
        nc = config['nc']
    
    # Create data.yaml content
    data_yaml = {
        'train': [],  # Will be populated with actual paths
        'val': [],    # Will be populated with actual paths
        'nc': nc,
        'names': class_names
    }
    
    # Populate train with preprocessed image paths
    for d in train_dirs:
        image_dir = os.path.join(d, 'images')
        if os.path.isdir(image_dir):
            data_yaml['train'].append(image_dir)
    
    # Populate val with original image paths
    for d in val_dirs:
        image_dir = os.path.join(d, 'images')
        if os.path.isdir(image_dir):
            data_yaml['val'].append(image_dir)
    
    # Verify paths exist
    if not data_yaml['train'] or not data_yaml['val']:
        raise FileNotFoundError("Could not find valid image directories in the dataset")
        
    # Write the YAML file
    data_yaml_path = os.path.join(DATA_DIR, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created dataset configuration at {data_yaml_path}")
    print(f"Train paths: {data_yaml['train']}")
    print(f"Val paths: {data_yaml['val']}")
    print(f"Number of classes: {nc}")
    print(f"Class names: {class_names}")
    
    return data_yaml_path

def train_model(config, repo_path):
    """Train YOLOv5m model with specified configuration"""
    # Create data.yaml file with proper format
    data_yaml_path = create_dataset_yaml()
    
    # Prepare command arguments
    cmd = [
        'python', os.path.join(repo_path, 'train.py'),
        '--img', str(config['img-size'][0]),
        '--batch', str(config['batch-size']),
        '--epochs', str(config['epochs']),
        '--data', data_yaml_path,
        '--weights', 'yolov5m.pt',  # Use YOLOv5m model
        '--project', RESULTS_DIR,
        '--name', 'model_B_preprocessed',
        '--exist-ok'
    ]
    
    # Execute training
    print(f"Starting training with command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    print("=== YOLOv5 Training for Model B (Preprocessed VisDrone2019) ===")
    
    # Check for YOLOv5 repo
    repo_path = check_yolov5_repo()
    
    # Load configuration
    config = load_config()
    
    # Train the model
    train_model(config, repo_path)
    
    print("Training completed for Model B.")
    
if __name__ == "__main__":
    main() 