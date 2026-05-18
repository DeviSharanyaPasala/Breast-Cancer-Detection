import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import app

def test_health():
    client = app.test_client()
    response = client.get('/health')
    assert response.status_code == 200

def test_ready_no_model():
    client = app.test_client()
    response = client.get('/ready')
    assert response.status_code in [200, 503]

def test_predict_no_file():
    client = app.test_client()
    response = client.post('/predict')
    assert response.status_code in [400, 503]
