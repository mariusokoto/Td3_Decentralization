from flask import Flask, jsonify, request
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the Iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(
    data=np.c_[iris['data'], iris['target']],
    columns=iris['feature_names'] + ['target']
)

# Split data into training and testing sets
X = iris_df.drop('target', axis=1)
y = iris_df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)

# Scale the features
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def hello_world():
    return 'Currently running. Make a prediction request at /predict'

@app.route('/predict', methods=["GET"])
def predict():
    try:
        # Extract feature values from the request
        features = [
            float(request.args.get("sepal_length")),
            float(request.args.get("sepal_width")),
            float(request.args.get("petal_length")),
            float(request.args.get("petal_width")),
        ]

        # Transform the features
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)

        # Make a prediction and return probabilities
        probabilities = model.predict_proba(features)

        return jsonify({
            "status": "success",
            "probability_scores": probabilities[0].tolist(),
            "class_labels": iris.target_names.tolist()
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# Run the application
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
