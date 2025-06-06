from fastapi import FastAPI
from pydantic import BaseModel
from .pipeline import OnlineDisasterMessagePipeline

app = FastAPI()

# Instantiate the pipeline once
pipeline = OnlineDisasterMessagePipeline()

# Define the input model
class MessageInput(BaseModel):
    text: str

@app.post("/predict")
def predict_message(input_data: MessageInput):
    result = pipeline.predict(input_data.text)
    return result
