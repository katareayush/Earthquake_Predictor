from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Home routes
@app.route('/')
@app.route('/home')
def home():
    return render_template('homepage.html')

@app.route('/error')
def error():
    return render_template('error.html')

@app.route('/aboutproject')
def aboutproject():
    return render_template('aboutproject.html')

@app.route('/review')
def review():
    return render_template('review.html')

@app.route('/sourcecode')
def sourcecode():
    return render_template('sourcecode.html')

@app.route('/creator')
def creator():
    return render_template('creator.html')

@app.route('/location')
def location():
    return render_template('location.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    try:
        data1 = int(float(request.form['a']))  # Parameter 1 (Latitude)
        data2 = int(float(request.form['b']))  # Parameter 2 (Longitude)
        data3 = int(float(request.form['c']))  # Parameter 3 (Depth)
    except ValueError:
        return render_template('error.html', message="Invalid input")

    arr = np.array([[data1, data2, data3]])
    output = model.predict(arr)[0]  # Predicted magnitude
    output_str = str(np.round(output, 2))
    print(f"Predicted Risk Score: {output}")
    

    prediction_categories = {
        (float('-inf'), 4): 'No',
        (4, 6): 'Low',
        (6, 8): 'Moderate',
        (8, 9): 'High',
        (9, float('inf')): 'Very High'
    }

    risk_score = None
    for (lower, upper), label in prediction_categories.items():
        if lower <= output < upper:
            risk_score = label
            break

    return render_template('prediction_results.html', magnitude=output_str, risk_score=risk_score)

if __name__ == "__main__":
    app.run(debug=True)
