from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np 
import torch
import random
import torch.nn as nn
import joblib

# start a fast api app
app = FastAPI(title="Air Quality Prediction API",
              description= "It predicts NO2 concentration over 72 time steps using the LSTM model.")

#load scaler 
scaler = joblib.load("scaler.pkl") 
input_size = scaler.n_features_in_ 

# MODEL parameters
hidden_size = 64 # lstm cells
num_layers = 2 
dropout_rate = 0.2 # prevent overfitting
output_size = 1 # target variable: "NO2(GT)"

# LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first= True, dropout= dropout)
        
        self.fc = nn.Linear(hidden_size, output_size)  # predict the last time step

    def forward(self, X):
        out, _ = self.lstm(X) # lstm output out = (batch, seq_len, hidden)
        out = out[:, -1, :] # only the last time step
        out = self.fc(out)
        return out


# create model 
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate)    
model.load_state_dict(torch.load("lstm_model.pt"))
model.eval() # evaluation mode

# data structure format
class SequenceInput(BaseModel): 
    sequence: list[list[float]]  # 2D array: 72 time step x 7 features


# API endpoint
@app.post("/predict")
def predict(data: SequenceInput):

    seq = data.sequence # take the sequence

    # control
    if len(seq) != 72 or len(seq[0] != input_size):
        raise HTTPException(status_code= 400, detail= "Invalid data size. Expected data size is 72 x 7. ")

    try:
        # to numpy array then to pytorch tensor
        X = np.array(seq).reshape(1, 72, input_size)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # predict with the model
        with torch.no_grad():
            prediction = model(X_tensor).numpy()[0][0] 

        # denormalization
        dummy = np.zeros((1, scaler.n_features_in_))  # all features
        prediction = dummy[0][0]
        inv_pred = scaler.inverse_transform(dummy[0][0])

        # json format return
        return {"predicted_NO2": round(inv_pred, 2)}
    except Exception as e:
        raise HTTPException(status_code= 500, detail= str(e))