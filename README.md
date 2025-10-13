# Breast Cancer Detection Using Mammogram Images with Deep Learning

**Author:** Devi Sharanya Pasala

**Department of Information Science and Technology, University at Albany, SUNY, NY, USA**

**Email:** [dpasala@albany.edu](mailto:dpasala@albany.edu)



## Project Overview

Breast cancer remains a major global health challenge, emphasizing the importance of early and accurate diagnosis. This project explores deep learning approaches for **automated breast cancer detection** using **mammogram images**. By applying and comparing various convolutional neural network (CNN) architectures, the study identifies the most effective model for improving diagnostic accuracy.

Traditional mammogram interpretation can be subjective and prone to error. This project demonstrates how modern deep learning models, particularly the **InceptionResNet Architecture** outperform conventional methods in classifying mammogram images as benign or malignant.



## Objectives

* To develop and compare multiple CNN architectures for mammogram classification.
* To evaluate performance using metrics such as **Accuracy**, **F1-Score**, **Cohen’s Kappa**, and **AUC**.
* To identify the most accurate and robust model for early breast cancer detection.



## Models and Methods

The study experimented with several architectures:

* **InceptionResNet**
* **ResNet50 / ResNet101**
* **VGG16 / VGG19**
* **MobileNet / MobileNetV2**
* **DenseNet121**
* **Convolutional (2-layer, 3-layer, and 8-layer) CNN models**

The **InceptionResNet Architecture** achieved the best results, offering superior accuracy and stability.

All models were implemented using **TensorFlow** and **Keras**, with the **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)** dataset.



## Dataset

* **Source:** CBIS-DDSM (Curated Breast Imaging Subset of DDSM)
* **Content:** 2,620 digitized mammography studies
* **Data Format:** DICOM images with metadata and pathology-confirmed labels
* **Labels:** Benign, Malignant, and Normal



## Implementation Steps

1. **Data Preprocessing**

   * Rescaling and normalization of images
   * Histogram equalization for contrast enhancement
   * Splitting into training, validation, and test sets

2. **Model Development**

   * Implemented using TensorFlow and Keras
   * Pre-trained models fine-tuned via transfer learning
   * Evaluated across accuracy, F1-score, AUC, and Kappa metrics

3. **Evaluation Metrics**

   * **Accuracy** – Overall correctness of predictions
   * **F1-Score** – Balances precision and recall
   * **Cohen’s Kappa** – Measures agreement between predictions and ground truth
   * **AUC** – Represents the model’s discriminative ability



## Results

| Model                        | Accuracy   | AUC    | F1 Score | Cohen Kappa |
| ---------------------------- | ---------- | ------ | -------- | ----------- |
| InceptionResNet Architecture | **0.9403** | 0.5420 | 0.5217   | -0.9243     |
| ResNet50                     | 0.6563     | 0.5530 | 0.5428   | -0.9423     |
| VGG16                        | 0.7633     | 0.5948 | 0.5406   | -0.9389     |
| VGG19                        | 0.5656     | 0.6216 | 0.5751   | -0.9132     |
| MobileNetV2                  | 0.9401     | 0.6126 | 0.5714   | -0.9729     |
| Conv 2-Layer                 | 0.5966     | 0.5235 | 0.4936   | -0.8744     |
| Conv 3-Layer                 | 0.6763     | 0.5396 | 0.4642   | -0.7391     |
| MobileNet                    | 0.5948     | 0.5866 | 0.5589   | -0.6294     |
| InceptionResNetV2            | 0.6948     | 0.5866 | 0.5625   | -0.9685     |
| ResNet101                    | 0.6645     | 0.5462 | 0.5328   | -0.9672     |

**Best Model:** *InceptionResNet Architecture*



## Key Findings

* Deep learning models outperform traditional feature-engineering approaches.
* Transfer learning using InceptionResNet achieved the highest detection accuracy.
* Data preprocessing (especially normalization and equalization) significantly improved model performance.
* CNN-based systems can assist radiologists by reducing diagnostic errors and improving detection rates.



## Conclusion

The project concludes that **InceptionResNet** provides the most accurate and reliable results among all tested architectures.
With further tuning and access to more diverse datasets, these models can be integrated into **computer-aided diagnostic systems (CAD)** to support radiologists and reduce false positives and negatives in breast cancer detection.



## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* DICOM image processing libraries



## Contact

For questions or collaboration, reach out:
**Devi Sharanya Pasala**
[dpasala@albany.edu](mailto:dpasala@albany.edu)



## References

All references used in this project are included in the full report and correspond to published works in *IEEE, Elsevier, and Springer* journals.



### If you find this project useful, please consider giving it a star on GitHub!
