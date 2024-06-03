from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# FastAPI örneği oluştur
app = FastAPI()

# Modeli yükle
model = joblib.load('model.pkl')

# Girdi verisi için Pydantic modelini oluştur
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Ana sayfa
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris classification API"}

# Tahmin için endpoint oluştur
@app.post("/predict")
def predict(iris: IrisFeatures):
    data = np.array([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}

from fastapi import FastAPI, Query
import joblib
import numpy as np

app = FastAPI()

# Modeli yükle
model = joblib.load('model.pkl')

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris classification API"}

@app.get("/predict")
def predict(
    sepal_length: float = Query(...),
    sepal_width: float = Query(...),
    petal_length: float = Query(...),
    petal_width: float = Query(...)
):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
