import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# load training data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

# numpy array to pytorch tensor
X_train = torch.tensor(X_train, dtype= torch.float32)
y_train = torch.tensor(y_train, dtype= torch.float32).unsqueeze(1)

# MODEL parameters
input_size = X_train.shape[2] # input variable number (7 features)
hidden_size = 64 # lstm cells
num_layers = 2 
dropout_rate = 0.2 # prevent overfitting
output_size = 1 # target variable: "NO2(GT)"
learning_rate = 0.001 # lr
num_epochs = 100

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

# loss funtion and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# training loop
loss_list = []

for epoch in range(num_epochs):
    model.train() # training mode
    output = model(X_train) # take the predictions
    loss = criterion(output, y_train) # calculate loss

    optimizer.zero_grad() 
    loss.backward() # backpropagation
    optimizer.step() # learning: update weights

    loss_list.append(loss.item()) # save loss

    if(epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1} / {num_epochs}], Loss: {loss.item()}")


# loss graph
plt.figure()
plt.plot(loss_list)
plt.title("Training Loss Graph - MSE")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

torch.save(model.state_dict(), "lstm_model.pt")
print("Model saved successfully.")