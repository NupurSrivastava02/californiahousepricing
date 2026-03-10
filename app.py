import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load model and scaler
regression = pickle.load(open("regmodel.pkl", "rb"))
scaler = pickle.load(open("scaling.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["data"]

    features = np.array([[
        data["MedInc"],
        data["HouseAge"],
        data["AveRooms"],
        data["AveBedrms"],
        data["Population"],
        data["AveOccup"],
        data["Latitude"],
        data["Longitude"]
    ]], dtype=float)

    new_data = scaler.transform(features)
    output = regression.predict(new_data)

    return jsonify({
        "predicted_house_value": float(output[0]),
        "unit": "$100k"
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = regression.predict(final_input)[0]
    output = round(output * 100000, 2)
    return render_template(
        'home.html',
        prediction_text="Predicted House Value: ${}".format(output)
    )
if __name__=="__main__":
    app.run(debug=True, port=5001)