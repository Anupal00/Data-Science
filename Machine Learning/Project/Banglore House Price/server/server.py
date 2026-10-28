from flask import Flask, request, jsonify
import util
app = Flask(__name__)

@app.route('/hello')
def hello():
    return "Hello World!"

@app.route('/get_location_names')
def get_location_names():
    response = jsonify({
        "locations": util.get_location_names()
    })
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    total_sqft = float(request.json["total_sqft"])
    location = request.json["location"]
    bhk = int(request.json["bhk"])
    bath = int(request.json["bath"])

    response = jsonify({
        "estimated_sqft": util.get_estimated_price(total_sqft, location, bhk, bath)
    })

    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

if __name__ == '__main__':
    print("Starting flask server for Banglore House Price prediction...")
    util.load_saved_artifacts()
    app.run()