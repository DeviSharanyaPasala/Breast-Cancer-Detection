# Breast Cancer Detection Using Deep Learning

---

## Table of Contents

- [Overview](#overview)
- [Project Objective](#project-objective)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Evaluation Metrics](#evaluation-metrics)
- [Methodology](#methodology)
- [Results Summary](#results-summary)
- [Visual Results](#visual-results)
- [Technical Summary](#technical-summary)
- [Why This Project Is Relevant](#why-this-project-is-relevant)
- [How to Run the Project](#how-to-run-the-project)
- [Key Features](#key-features)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [License](#license)

---

## Overview

This project focuses on the early detection of breast cancer using deep learning techniques applied to mammography images. The goal is to evaluate how different convolutional neural network (CNN) architectures perform when trained on the same dataset and preprocessing pipeline, rather than relying on a single model.

The project emphasizes reproducibility, proper evaluation, and model comparison, which are essential when applying machine learning to healthcare-related problems.

---

## Project Objective

Breast cancer is one of the most common cancers worldwide. Early and accurate detection can significantly improve treatment outcomes.

The objective of this project is to:
- Classify breast cancer images as **Benign** or **Malignant**
- Compare different deep learning models
- Evaluate model performance using reliable metrics

---

## Dataset

- **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)**
- Mammography images labeled as **benign** or **malignant**
- Images resized and normalized for consistency across models

Due to licensing restrictions, the dataset cannot be redistributed through this repository.

Dataset sources:
- https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset  
- https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM  

After downloading, organize the images into train, validation, and test folders as described in the notebook.

---

## Models Implemented

The following pre-trained CNN architectures were fine-tuned using transfer learning:

- VGG16  
- VGG19  
- ResNet50  
- ResNet101  
- InceptionV3  
- InceptionResNetV2  
- MobileNet  
- DenseNet121  

All models were trained using identical data splits and hyperparameter settings to ensure a fair comparison.

---

## Evaluation Metrics

Model performance was measured using:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC Score  

These metrics are especially important for medical image classification, where both correctness and sensitivity matter.

---

## Methodology

1. Data preprocessing including resizing, normalization, and stratified splitting  
2. Transfer learning with frozen base layers during initial training  
3. Binary classification using binary cross-entropy loss  
4. Model evaluation using multiple performance metrics  

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|---------|----------|--------|---------|--------|
| DenseNet121 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 |
| MobileNet | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 |
| ResNet101 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |
| ResNet50 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 |

DenseNet121 achieved the most consistent performance across all evaluation metrics, indicating strong feature reuse and generalization on mammography images.

---

## Visual Results

<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/1d1f46da-bc51-4d65-a2e0-ddf0ba3281e7" />


---

## Project Workflow (GIF)

*High-level workflow of the project pipeline:*

- Data loading and preprocessing  
- Model training using transfer learning  
- Evaluation and comparison  


---

## Technical Summary

- **Task:** Binary image classification (Benign vs Malignant)
- **Data Type:** Mammography images
- **Learning Type:** Supervised learning
- **Models:** Pre-trained CNNs with transfer learning
- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Frameworks:** TensorFlow / Keras
- **Evaluation Focus:** Balanced metric selection for medical reliability

This section highlights the technical depth of the project for recruiters and data science professionals.

---

## Why This Project Is Relevant

- Demonstrates comparative model evaluation rather than single-model training  
- Highlights responsible metric selection for medical image classification  
- Reflects real-world machine learning workflows used in healthcare analytics  

---

## How to Run the Project

1. Clone the repository:

       git clone https://github.com/DeviSharanyaPasala/Breast-Cancer-Detection.git

2. Navigate to the project directory:

       cd Breast-Cancer-Detection

3. Install the required dependencies:

       pip install -r requirements.txt

4. Open and run the Jupyter Notebook:

       jupyter notebook FINAL_CODE.ipynb

---

## Key Features

- Breast cancer detection using deep learning
- Comparison of multiple CNN architectures
- Use of transfer learning for improved performance
- Evaluation using standard medical metrics
- Complete end-to-end machine learning pipeline

---

## Future Improvements

- Add explainability techniques such as Grad-CAM
- Deploy the trained model as a web application
- Improve performance using data augmentation
- Create dashboards for model comparison

---

## Author

**Devi Sharanya Pasala**  
Graduate student in Information Science with a focus on Data Analytics and Artificial Intelligence.

- GitHub: https://github.com/DeviSharanyaPasala  
- LinkedIn: https://www.linkedin.com/in/devi-sharanya/

---

## License

This project is intended for academic and learning purposes.
