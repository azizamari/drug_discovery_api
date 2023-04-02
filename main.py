from fastapi import FastAPI, File
import json
from fastapi.middleware.cors import CORSMiddleware
from utils_functions import get_model
model = get_model()
app = FastAPI(
    title="Drug Discovery model inference API",
    description="""""",
    version="0.0.1",
)
origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pydantic import BaseModel


class input_data(BaseModel):
    canonical_smiles: str

@app.post("/predict")
async def predict_class(data: input_data):
    return {"result": data.canonical_smiles}