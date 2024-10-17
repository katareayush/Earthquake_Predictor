import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/prediction', methods=['POST'])
def prediction():
    data1 = float(request.form['latitude'])
    data2 = float(request.form['longitude'])
    data3 = float(request.form['depth'])
    
    # Prepare input for the model
    arr = np.array([[data1, data2, data3]])
    
    # Make prediction
    output = model.predict(arr)

    # Convert the output to a readable format
    output_value = output[0]

    return render_template('prediction.html', p=output_value)
    
print("Model Created Sucessfully")

# Example input data
test_input = np.array([[29.06, 77.42, 5]])  # Replace with actual values
predicted_magnitude = model.predict(test_input)

print(f"Predicted Magnitude: {predicted_magnitude[0]}")


