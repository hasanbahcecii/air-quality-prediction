import streamlit as st
import numpy as np
import requests

# header and description
st.title("NO2 Prediction")
st.markdown("This application estimates the NO2 level in the air using environmental data from the past 72 hours.")

if st.button("Predict with existing sample data"):
    try:
        x_test = np.load("X_test.npy")
        example = x_test[0].tolist()

        # send API request
        response = requests.post("http://127.0.0.1:8000/predict", json = {"sequence": example})

        if response.status_code == 200:
            result = response.json()

            st.success(f"Predicted NO2: {result["predicted_NO2"]}")
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")    

    except FileNotFoundError:
        st.error("X_test.npy file not found!")
    except requests.exceptions.ConnectionError:
        st.error("API connection error.")            

# manual input user interface
with st.expander("Predict with your own data"):
    st.markdown("Data format: 72 time step, for each step 7 features")

    custom_input = st.text_area("Input your data in JSON format. e.g. [[0.2, 0.3, ...] [...], [...] ]")

    if st.button("Make manual prediction"):
        try:
            parsed = eval(custom_input)

            if len(parsed) == 72 and len(parsed[0]) == 7:
                response = requests.post("http://127.0.0.1:8000/predict", json = {"sequence": parsed})

                if response.status_code == 200:
                    result = response.json()

                    st.success(f"Predicted NO2: {result["predicted_NO2"]}")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")    
            else:
                st.warning("Please, enter the data in 72 time steps, with 7 features for each step.")
        except Exception as e:
            st.error(f"Input could not be parsed {e}")