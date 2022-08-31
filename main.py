
from pydantic import BaseModel
from fastapi import FastAPI
from transformers import BertForSequenceClassification, BertTokenizer
import torch

app = FastAPI()

class request_body(BaseModel):
    snippet: str

# initialize our model and tokenizer
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

def sentiment(tokens):
    # get output logits from the model
    output = model(**tokens)
    # convert to probabilities
    probs = torch.nn.functional.softmax(output[0], dim=-1)
    pred = torch.argmax(probs)
    return pred.item()

@app.get("/healthcheck")
def read_root():
    return {"Hello": "world"}
    
@app.post("/predict")
def predict(data: request_body):
    txt = data.snippet
    tokens = tokenizer.encode_plus(txt, max_length=512, truncation=True, padding='max_length',
                               add_special_tokens=True, return_tensors='pt')
    pred = sentiment(tokens)
    return {"snippet": txt, "pred": pred}