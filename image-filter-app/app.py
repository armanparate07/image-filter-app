from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os
from filters import apply_filter

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    filter_type = request.form['filter']
    
    # Read image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return 'Failed to read image'
    
    output = apply_filter(img, filter_type)

    # Save and return image
    output_path = 'static/output.jpg'
    cv2.imwrite(output_path, output)
    return render_template('result.html', image_path=output_path)

if __name__ == '__main__':
    app.run(debug=True)
