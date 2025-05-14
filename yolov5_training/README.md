# YOLOv5 Training and Comparison Framework

This framework enables training and comparative evaluation of YOLOv5m models on original and preprocessed VisDrone2019 datasets.

## Folder Structure

```
yolov5_training/
├── model_A/                  # Original VisDrone2019 dataset model
│   ├── config.yaml           # Model configuration
│   ├── train.py              # Training script
│   ├── data/                 # Data configuration
│   └── results/              # Training results
├── model_B/                  # Preprocessed VisDrone2019 dataset model
│   ├── config.yaml           # Model configuration
│   ├── train.py              # Training script
│   ├── data/                 # Data configuration
│   └── results/              # Training results
├── yolov5_repo/              # YOLOv5 repository (cloned automatically)
├── compare_results.py        # Script to compare model performance
├── evaluate_small_objects.py # Script to evaluate small object detection
├── comparison_results/       # Generated comparison reports and visualizations
└── small_object_evaluation/  # Small object detection evaluation results
```

## Training Configuration

Both models use the YOLOv5m architecture with the following settings:
- Epochs: 30
- Batch size: Auto-determined
- Image size: 640x640
- Learning rate: Default YOLOv5 settings
- Optimizer: SGD

## Getting Started

### Prerequisites

- Python 3.7+
- Pip package manager
- Git

The required packages will be installed automatically during the first run.

### Training the Models

1. To train the model on the original VisDrone2019 dataset:

```bash
cd yolov5_training/model_A
python train.py
```

2. To train the model on the preprocessed VisDrone2019 dataset:

```bash
cd yolov5_training/model_B
python train.py
```

Each script will:
- Clone the YOLOv5 repository if it doesn't exist
- Install required dependencies
- Train the YOLOv5m model with the specified configuration
- Save results to the respective results directory

### Evaluating the Models

After training both models, you can evaluate small object detection performance:

```bash
cd yolov5_training
python evaluate_small_objects.py
```

This will:
- Run YOLOv5 validation on both models
- Focus analysis on small object detection
- Save results to the small_object_evaluation directory

### Comparing the Results

To generate a comprehensive comparison between the two models:

```bash
cd yolov5_training
python compare_results.py
```

This will:
- Analyze performance metrics for both models
- Generate visualizations comparing key metrics
- Create a detailed report with conclusions
- Save all outputs to the comparison_results directory

## Metrics

The comparison includes:
- mAP@0.5 (primary metric)
- Precision and Recall
- Training time
- Small object detection performance

## Notes

- The evaluation uses original images from the VisDrone2019 subset for validation
- Results are saved in separate directories for easy comparison
- The visualization includes charts comparing metrics and training times 