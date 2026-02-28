import requests
import numpy as np

# load test dataset 
X_test = np.load("X_test.npy")

# take samples for test
sample_index = 0
sample = (X_test[sample_index]) # (72, 7) data

# generate test data
print(sample.tolist())

# make it suitable for json
# fast api: list[list[float]]
payload={
    "sequence": sample.tolist() # numpy to list
}

# api url
url = "http://127.0.0.1:8000/predict"

# post request
try:
    response = requests.post(url, json= payload)
    if response.status_code == 200:
        result = response.json()
        print(f"Predicted NO2: {result}")
    else:
        print(f"Error code: {response.status_code}")    
        print(f"Error message: {response.text}")

except requests.exceptions.ConnectionError:
    print("The API server is not responding. Please, start FastAPI !")       