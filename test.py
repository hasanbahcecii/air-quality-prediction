import torch
import torch.nn as nn
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# numpy array to pytorch tensor format
X_test = torch.tensor(X_test, dtype= torch.float32)
y_test = torch.tensor(y_test, dtype= torch.float32).view(-1, 1)

print(y_test.shape)

# load scaler object for normalization
scaler = joblib.load("scaler.pkl")


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


# MODEL parameters
input_size = X_test.shape[2] 
hidden_size = 64 # lstm cells
num_layers = 2 
dropout_rate = 0.2 # prevent overfitting
output_size = 1 # target variable: "NO2(GT)"



# create model 
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate)    
model.load_state_dict(torch.load("lstm_model.pt"))
model.eval() # test mode

# prediction
with torch.no_grad():
    predictions = model(X_test)

# convert predictions to numpy array
y_pred = predictions.numpy()
y_true = y_test.numpy()

print("y_pred shape:", y_pred.shape)
print("y_true shape:", y_true.shape)

# denormalization
dummy_pred = np.zeros((len(y_pred), scaler.n_features_in_))
dummy_pred[:, 0] = y_pred[:, 0]

dummy_true = np.zeros((len(y_true), scaler.n_features_in_))
dummy_true[:, 0] = y_true[:, 0]

# inverse transform
inv_y_pred = scaler.inverse_transform(dummy_pred)[:, 0]
inv_y_true = scaler.inverse_transform(dummy_true)[:, 0]

# real vs prediction comparison
plt.figure()
plt.plot(inv_y_true, label = "Real NO2", color= "blue")
plt.plot(inv_y_pred, label = "Predicted NO2", color = "red", alpha = 0.5)
plt.title("Real vs Prediction NO2")
plt.xlabel("Time Step")
plt.ylabel("NO2")
plt.legend()
plt.show()

# performance metric evaluation
mae = mean_absolute_error(inv_y_pred, inv_y_true)
mse = mean_squared_error(inv_y_pred, inv_y_true)

print(f"MAE: {mae}")
print(f"MSE: {mse}")