from pyexpat import model
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from models.eye_detection_model import augment_data, load_model, predict_disease
from predict import predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image
        file = request.files['image']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['/Users/kanik/eye-detection-ui/upload_folder'], filename)
        file.save(file_path)

        # Process the image using your eye detection model
        predicted_label = predict_disease(file_path, model)

        # Render the results on the HTML template
        return render_template('results.html', predicted_label=predicted_label)

    return render_template('index.html')

if __name__ == '__main__':
    # Create the upload directory if it doesn't exist
    os.makedirs(app.config['/Users/kanik/eye-detection-ui/upload_folder'], exist_ok=True)
    app.run(debug=True)