# CMSC-194.2-Optimization
Replicating a study titled "Optimization algorithm to reduce training time for deep learning computer vision algorithms using large image datasets with tiny objects" by Sergio Bemposta Rosende, Javier Fernández-Andrés, and Javier Sánchez-Soriano

### I. Introduction

High-resolution drone footage presents new opportunities and challenges in computer vision, particularly for detecting tiny objects like pedestrians, cars, and bicycles. However, the training of deep learning models with such video-based datasets is computationally expensive, especially when high frame rates produce highly redundant data. 

This study replicates the optimization approach of Rosende et al. (2023), focusing on using videos from the VisDrone2019-VID dataset. By pre-processing video frames to discard redundant objects and crop relevant regions, we aim to significantly reduce training time without affecting model performance. This study applies the method to a smaller, selected set of videos to verify its effectiveness even at a reduced scale.

---

### II. Methodology

This study follows a two-phase pre-processing pipeline adapted specifically for video sequences:

#### A. Dataset Preparation

- **Source:** VisDrone2019-VID dataset
- **Subset:** Select 2–5 video sequences (≈2,000–5,000 frames total).
- **Resolution:** FullHD (1920×1080).
- **Annotations:** Provided as `.txt` files per video, already compatible with YOLO format (some reformatting may be needed).

#### B. Phase 1: Object Discarding Based on Video Frames

1. **Video Frame Extraction:**  
   - Extract all frames of the selected videos (if needed) at 1 FPS or 2 FPS (to simulate low-frame drone recordings).

2. **Sequential Object Tracking:**  
   - Mark every Nth frame (e.g., every 7th) as a **key frame** → keep all objects.
   - For frames between key frames:
     - Compare object bounding boxes to previous frame.
     - If object displacement <10 pixels → discard (static object).
     - If displacement ≥10 pixels → retain (moving object).
   - Update labels accordingly.

#### C. Phase 2: Cropping Around Objects

1. For each selected object:
   - Center a 640x640 crop around it.
   - Make sure crop stays inside image boundaries.
2. Include inside cropped image:
   - Objects fully inside (100%) → keep as labels.
   - Objects partially inside (≥50%) → keep.
   - Objects less than 50% inside → **blur** or ignore.

3. Save cropped frames and YOLO annotations.
