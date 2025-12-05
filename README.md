Malaria Parasite Quantitation and Classification with YOLOv8
This project demonstrates the use of YOLOv8 for automated detection, quantitation, and classification of malaria parasites in microscopic blood smear images. The pipeline covers dataset preparation, model training, and comprehensive evaluation on both internal validation and external generalization datasets.

Table of Contents
Setup
Dataset
Model Training
Evaluation
Internal Validation (Quantitation)
External Validation (Classification)
Results Summary
Inference Performance
Setup
1. Requirements
First, install the necessary Python libraries. A requirements.txt file is provided for easy installation:

pip install -r requirements.txt
Key libraries include ultralytics (for YOLOv8), torch, opencv-python, pandas, matplotlib, and scikit-learn.

2. Google Colab Specifics
This notebook is designed to run in Google Colab, leveraging GPU acceleration and Google Drive for persistent storage of datasets and trained models.

Mount Google Drive: The notebook automatically mounts your Google Drive to /content/drive to access input datasets and save model outputs.
GPU Check: Verifies Ultralytics installation and CUDA (GPU) availability.
Dataset
1. Source Data
The raw dataset consists of microscopic blood smear images along with PASCAL VOC XML annotation files, specifying bounding box locations of 'Trophozoite' stage malaria parasites.

2. Local Copy
For faster processing, the dataset is copied from Google Drive (/content/drive/MyDrive/malaria_quantitation/full_dataset) to a local Colab directory (/content/local_malaria_data).

3. Preparation for YOLOv8
The XML annotations are converted to YOLO format (.txt files), and the data is split into 80% training and 20% validation sets. A dataset.yaml configuration file is generated for YOLOv8 training.

4. Ground Truth Visualization
A random image from the training set is visualized with its ground truth bounding box annotations to ensure correct data preparation.

Model Training
The project uses the YOLOv8s (small) model architecture. The model is trained with the following parameters:

Model: yolov8s.pt (pretrained weights)
Dataset: /content/malaria_dataset/dataset.yaml
Epochs: 100
Image Size: 1280x1280 pixels
Augmentations: Aggressive augmentations including mosaic (1.0), mixup (0.1), and degrees (180) are applied to improve generalization.
Early Stopping: patience=15 is used to prevent overfitting.
Output: Training results and model weights are saved to Google Drive (/content/drive/MyDrive/malaria_quantitation/malaria_models/yolov8s_malaria_final).
Evaluation
Internal Validation (Quantitation)
The best-trained model is evaluated on the internal validation set to assess its ability to accurately count parasites.

Inference: The model predicts bounding boxes on validation images.
Quantification Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R-squared (R2), Mean Absolute Percentage Error (MAPE), and accuracy (exact match, within +/- 1, within +/- 5 counts) are calculated to quantify counting performance.
Visualization: Scatter plots show actual vs. predicted counts and prediction bias.
External Validation (Classification)
An external dataset (/content/drive/MyDrive/malaria_quantitation/Val_small_dataset), categorized into 'Uninfected' and 'Infected' folders, is used to test the model's generalization capabilities for classifying infection status.

Data Preparation: The external validation dataset is copied locally.
Inference & Classification: For each image, the model predicts parasite counts. Images with predicted_count > 0 are classified as 'Infected', otherwise 'Uninfected'.
Classification Metrics: Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix are computed for the binary classification task.
Error Analysis: Detailed analysis identifies false positive and false negative images.
Visualization: Histograms and bar plots illustrate the distribution of predicted counts for 'Uninfected' and 'Infected' images.

Training Performance Metrics:
Precision
Recall
mAP50
mAP50-95

Quantification Performance (Internal Validation):
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R-squared (R2) Score
MAPE (on non-zero images)
Exact Match Accuracy
Accuracy (within +/- 1 count)
Accuracy (within +/- 5 counts)

Classification Performance (External Validation):
Accuracy
Precision (for 'Infected')
Recall (Sensitivity, for 'Infected')
F1-Score (for 'Infected')

Confusion Matrix:
True Negatives:
False Positives: 
False Negatives: 
True Positives:

Inference Performance
The model demonstrates efficient inference speed on the current hardware:
Approximate Inference Speed: 
