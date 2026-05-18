# Breast Cancer Detection — MLOps Pipeline

Early detection of breast cancer from mammogram images using deep learning, wrapped in a full MLOps pipeline with experiment tracking, a REST API, and automated CI/CD.

## Results

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| DenseNet121 | 0.97 | 0.97 | 0.97 |
| MobileNet | 0.96 | 0.96 | 0.96 |
| ResNet101 | 0.95 | 0.95 | 0.95 |
| ResNet50 | 0.94 | 0.94 | 0.94 |

DenseNet121 performed best across all metrics and is used for inference.

## Project Structure
## MLOps Pipeline

Every push to main triggers:
1. **Run Tests** — pytest against the Flask API
2. **Build Docker Image** — builds and validates the container
3. **Validate MLOps Setup** — confirms MLflow and training script are working

## Flask API

The model is served via a Flask API with three endpoints:

- `GET /health` — returns healthy status
- `GET /ready` — returns ready status if model is loaded
- `POST /predict` — accepts an image file and returns benign/malignant prediction with confidence score

## Training

Training uses DenseNet121 with transfer learning on the CBIS-DDSM dataset. MLflow logs every run including hyperparameters, validation accuracy, and the saved model artifact.

```bash
DATASET_PATH=data EPOCHS=10 python train.py
```

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```

Or with Docker:

```bash
docker build -t breast-cancer-detection .
docker run -p 5000:5000 breast-cancer-detection
```

## Dataset

CBIS-DDSM (Curated Breast Imaging Subset of DDSM) — mammography images labeled benign or malignant.

- https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
- https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM

## Author

Devi Sharanya Pasala
- GitHub: https://github.com/DeviSharanyaPasala
- LinkedIn: https://www.linkedin.com/in/devisharanya
