

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_smile_model(input_shape=(224, 224, 3)):
    """
    Builds and returns the CNN model.
    """
    model = Sequential()
    
    # Add convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten the output of convolutions
    model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout for regularization
    
    # Output layer for smile percentage (regression)
    model.add(Dense(1, activation='sigmoid'))  # Linear output for regression
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['mae'])
    
    return model

