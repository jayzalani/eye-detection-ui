# EyeDx: Diabetic Eye Disease Detection

## Overview
EyeDx is a deep learning-based project aimed at detecting diabetic eye diseases such as Diabetic Retinopathy, Cataracts, and Glaucoma from retinal images. The model leverages **EfficientNetB3** as a feature extractor and employs transfer learning to classify images into different eye disease categories.

## Problem Statement
Diabetes is a leading cause of blindness among working-age adults, primarily due to complications like:
- **Diabetic Retinopathy**: Damage to the retina's small blood vessels due to high blood sugar levels.
- **Cataracts**: Clouding of the eye lens, leading to impaired vision.
- **Glaucoma**: Damage to the optic nerve, often due to increased eye pressure.

Regular screening and early diagnosis can prevent severe vision loss and blindness. **EyeDx** provides an AI-powered solution for automated detection, assisting medical professionals in early intervention.

## Dataset
The dataset consists of retinal images categorized into four classes:
- **Normal**
- **Diabetic Retinopathy**
- **Cataract**
- **Glaucoma**

Each class contains approximately **1000 images**, sourced from various medical datasets, including:
- **IDRiD** (Indian Diabetic Retinopathy Image Dataset)
- **Ocular Recognition datasets**
- **HRF (High-Resolution Fundus Images)**

## Model Architecture
EyeDx employs **EfficientNetB3** as the backbone model with the following architecture:
- Pretrained **EfficientNetB3** (weights: `imagenet`, `include_top=False`)
- Global Average Pooling Layer
- Fully Connected Layer (512 neurons, ReLU activation, L2 regularization)
- Output Layer (Softmax activation, L2 regularization)

The model is compiled using:
```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
```

## Installation & Usage
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install tensorflow numpy pandas matplotlib opencv-python
```

### Training the Model
1. Load and preprocess the dataset.
2. Initialize and compile the model.
3. Train the model using the following command:
   ```python
   model.fit(train_data, epochs=20, validation_data=val_data)
   ```
4. Evaluate the model:
   ```python
   model.evaluate(test_data)
   ```

### Inference
To make predictions on new images:
```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('eyedx_model.h5')
image = cv2.imread('sample_image.jpg')
image = cv2.resize(image, (256, 256))
image = np.expand_dims(image, axis=0)

prediction = model.predict(image)
predicted_class = np.argmax(prediction)
print(f'Predicted Class: {predicted_class}')
```

## Results & Performance
- **Accuracy**: The model achieves competitive accuracy in classifying retinal diseases.
- **Loss & Metrics**: Evaluation metrics indicate effective learning and generalization.
- **Confusion Matrix**: Used for analyzing misclassifications.

## Future Improvements
- Optimize model architecture for better performance.
- Incorporate explainability methods (Grad-CAM) for visualizing model decisions.
- Deploy the model as a web app for real-time diagnosis.

## Contributors
- **Jay Zalani**

## License
This project is licensed under the **MIT License**.

