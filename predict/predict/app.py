# Flask
from flask import Flask, request, jsonify, render_template
from predict.predict import run
import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model outside of the route
model = None

# Prediction route
@app.route('/', methods=['GET', 'POST'])
def predict():
    global model
    # /Users/thibaultgillard/Documents/EPF/5A/Poc_to_Prod/poc-to-prod-capstone/train/data/artefacts/2023-12-12-14-50-05
    if model is None:
        model = run.TextPredictionModel.from_artefacts("/Users/thibaultgillard/Documents/EPF/5A/Poc_to_Prod/poc-to-prod-capstone/train/data/artefacts/2023-12-12-14-50-05")

    try:
        if request.method == 'POST':
            # Get text input from the form
            text = request.form['text']

            # Perform predictions using the loaded model
            predictions = model.predict([text], top_k=5)

            # Get the labels for all top_k predictions
            index_return = [model.labels_index_inv[pred[0]] for pred in predictions]

            return jsonify({"predictions": index_return})

        return render_template('predict.html')

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({"error": "Prediction failed. Check server logs for details."}), 500

    # Running the Flask app
if __name__ == "__main__":
        app.run(host='0.0.0.0', port=5001, debug=True)
