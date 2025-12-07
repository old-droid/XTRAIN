
from flask import Flask, request, jsonify
import numpy as np
from run_model import ModelRunner

app = Flask(__name__)

# Cache for loaded models
models = {}

def get_model_runner(model_type):
    """
    Loads a model runner for the given model type and caches it.
    """
    if model_type not in models:
        print(f"Loading model of type: {model_type}")
        models[model_type] = ModelRunner(model_type=model_type)
    return models[model_type]

@app.route("/infer/<model_type>", methods=["POST"])
def infer(model_type):
    """
    Perform inference on the specified model.
    """
    if not request.json or 'data' not in request.json:
        return jsonify({"error": "Missing 'data' in request body"}), 400

    try:
        model_runner = get_model_runner(model_type)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    data = np.array(request.json['data'])

    # Perform inference
    # Note: The input data shape should match the model's expected input shape.
    # This is a simplified example.
    try:
        # The forward method of the model is called through the runner's model attribute
        if model_runner.model:
            output = model_runner.model.forward(data)
            # Convert numpy array to list for JSON serialization
            output = output.tolist()
            return jsonify({"prediction": output})
        else:
            return jsonify({"error": "Model is not loaded"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3434, debug=True)
