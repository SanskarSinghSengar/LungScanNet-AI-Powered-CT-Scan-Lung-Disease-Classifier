# LungScanNet-AI-Powered-CT-Scan-Lung-Disease-Classifier
This project uses a CNN to classify chest CT scans into four categories—three lung diseases and normal—providing confidence scores and aiding in fast, accurate medical diagnosis.

# Overview
This project uses Convolutional Neural Networks (CNNs) to classify chest CT scan images into four categories:

->Adenocarcinoma

->Large Cell Disease

->Squamous Cell Disease

->Normal

The system predicts the health condition of a patient based on CT scan images, providing a confidence score for each prediction. This can assist doctors in making fast and data-backed medical decisions.

# Tech Stack
Python 3

TensorFlow / Keras – Deep learning model

Scikit-learn – Metrics (confusion matrix, precision, recall, F1-score)

Matplotlib – Plotting results and samples

Pillow (PIL) – Image processing

KaggleHub – Dataset download from Kaggle

# Dataset
We use the Chest CT Scan Images dataset from Kaggle:
https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images


# How to Get the Kaggle API Token
Create a free account on Kaggle.

Go to your Kaggle Account → API → Click Create New API Token.

A file named kaggle.json will be downloaded.

Upload this file into your working directory (e.g., Colab or local environment).

# How to Download the Dataset
python
Copy
Edit
import kagglehub

# Download dataset
path = kagglehub.dataset_download("mohamedhanyyy/chest-ctscan-images")
print(" Dataset downloaded to:", path)
Working of the Model
Data Loading & Preprocessing – Images are resized and normalized.

Model Architecture – A CNN extracts features and classifies the image.

Training – The model learns from the training set and validates on the validation set.

Evaluation – The test set is used to calculate accuracy, precision, recall, F1-score, and confusion matrix.

Prediction – The user can upload a CT scan image, and the model outputs the predicted disease or “No Disease” with a confidence score.

# Usage
Download the dataset using the Kaggle API token.

Train the CNN model on the dataset.

Evaluate performance with metrics and visualizations.

Upload a CT scan image to get predictions and confidence scores.

# License
This project is intended for educational and research purposes only and should not be used as a substitute for professional medical diagnosis.
