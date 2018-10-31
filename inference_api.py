import os
from flask import Flask, render_template, request
from neural_kb import predict
from fact_base import get_facts
import json

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    file.save(f)

    disease_id, label, prediction = predict(f)

    return json.dumps(get_facts(disease_id))


if __name__ == "__main__":
    app.run(host='0.0.0.0')
