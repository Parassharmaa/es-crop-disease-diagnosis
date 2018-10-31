import os
from flask import Flask, render_template, request, send_from_directory
from neural_kb import predict
from fact_base import get_facts
import json

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_disease():
    file = request.files['image']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(f)
    disease_id, label, prediction = predict(f)

    data = get_facts(disease_id)

    return render_template("predict.html", data=data, image=file.filename)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=8085)
