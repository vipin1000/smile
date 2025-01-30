import cv2
import dlib
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from CNN_MODEL import build_smile_model  

# Load face detector from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess the image by detecting the face, cropping it, resizing, and normalizing.
    """
    # Read image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    
    # If no face detected, return None
    if len(faces) == 0:
        return None
    
    # Take the first face detected
    face = faces[0]
    
    # Crop the face region
    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
    face_image = image[y:y+h, x:x+w]
    
    # Resize the face to the target size
    face_image = cv2.resize(face_image, target_size)
    
    # Normalize the pixel values to [0, 1]
    face_image = face_image / 255.0
    
    return face_image

def load_data_from_folders(smiling_folder, not_smiling_folder, target_size=(224, 224)):
    images = []
    labels = []
    
    # Load smiling images (label=1)
    for filename in os.listdir(smiling_folder):
        image_path = os.path.join(smiling_folder, filename)
        image = preprocess_image(image_path, target_size)
        if image is not None:
            images.append(image)
            labels.append(1)  # Label for smiling images
    
    # Load not smiling images (label=0)
    for filename in os.listdir(not_smiling_folder):
        image_path = os.path.join(not_smiling_folder, filename)
        image = preprocess_image(image_path, target_size)
        if image is not None:
            images.append(image)
            labels.append(0)  # Label for not smiling images
    
    return np.array(images), np.array(labels)





smiling_folder = 'data'
not_smiling_folder = 'data2'

# Load images and labels
X, y = load_data_from_folders(smiling_folder, not_smiling_folder)

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and compile the model
model = build_smile_model()

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the model after training
model.save('smile_detection_model.h5')

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model on validation data
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")


# Example: Predict smile percentage for a new image
def predict_smile(image_path):
    image = preprocess_image(image_path)
    if image is not None:
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        prediction = model.predict(image)
        # Return the prediction as 0 (not smiling) or 1 (smiling)
        return 'Smiling' if prediction[0][0] > 0.5 else 'Not Smiling'
    else:
        return None

# Test the model with a new image
predicted_label = predict_smile('new_face_image.jpg')
print(f"Predicted Label: {predicted_label}")
