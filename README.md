## Breast Cancer Detection Using Deep Learning

### Overview
This project focuses on the early detection of breast cancer using deep learning techniques applied to mammography images. The goal is to evaluate how different convolutional neural network (CNN) architectures perform when trained on the same dataset and preprocessing pipeline, rather than relying on a single model.

The project emphasizes reproducibility, proper evaluation, and model comparison, which are essential when applying machine learning to healthcare-related problems.

---

### Dataset
- **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)**
- Mammography images labeled as **benign** or **malignant**
- Images were resized and normalized for consistency across models

Due to licensing restrictions, the dataset cannot be redistributed through this repository.
You can request and download the dataset from the official source:

🔗 [https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)<img width="468" height="45" alt="image" src="https://github.com/user-attachments/assets/cd42ea11-eab8-4485-b497-12a91fd7dd97" />


After downloading, organize the images into train/validation/test folders as described in the notebook.
---

### Models Implemented
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

### Methodology
1. Data preprocessing including resizing, normalization, and stratified splitting  
2. Transfer learning with frozen base layers during initial training  
3. Binary classification using binary cross-entropy loss  
4. Model evaluation using multiple performance metrics  

---

### Results Summary
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|---------|----------|--------|---------|--------|
| DenseNet121 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 |
| MobileNet | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 |
| ResNet101 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |
| ResNet50 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 |

DenseNet121 achieved the most consistent performance across all evaluation metrics, indicating strong feature reuse and generalization on mammography images.

---

### Why This Project Is Relevant
- Demonstrates comparative model evaluation rather than single-model training  
- Highlights responsible metric selection for medical image classification  
- Reflects real-world machine learning workflows used in healthcare analytics  

---

### How to Run
```bash
git clone https://github.com/DeviSharanyaPasala/Breast-Cancer-Detection.git
cd Breast-Cancer-Detection
pip install -r requirements.txt
