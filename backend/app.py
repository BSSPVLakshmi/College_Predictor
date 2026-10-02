from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor_api import predict_colleges, get_available_districts, get_available_branches

app = Flask(__name__)
CORS(app)

@app.route("/options", methods=["GET"])
def options():
    return jsonify({
        "districts": get_available_districts(),
        "branches": get_available_branches()
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    try:
        results = predict_colleges(
            rank=data["rank"],
            category=data["category"],
            gender=data["gender"],
            districts=data.get("districts", "ALL"),
            branches=data.get("branches", "ALL"),
            sort_choice=data.get("sort_choice", "1"),
            show_only_safe_target=data.get("show_only_safe_target", "no"),
        )
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)