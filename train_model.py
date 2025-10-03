import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# --- Model & Data Constants ---
WINDOW_SIZE = 30
NUM_CHANNELS = 4  # x, y, z, magnitude
INPUT_SHAPE = (WINDOW_SIZE, NUM_CHANNELS)
RECORD_STRIDE = 20

def prepare_data(gestures):
    """Prepares data for training."""
    inputs = []
    outputs = []

    def process_gesture(gesture_samples, label):
        if len(gesture_samples) < WINDOW_SIZE:
            return
        for i in range(0, len(gesture_samples) - WINDOW_SIZE + 1, RECORD_STRIDE):
            window = gesture_samples[i : i + WINDOW_SIZE]
            inputs.append(window)
            outputs.append([label])

    process_gesture(gestures.get("0", []), 0)
    process_gesture(gestures.get("1", []), 1)

    if not inputs:
        return None, None

    input_tensor = np.array(inputs, dtype=np.float32)
    output_tensor = np.array(outputs, dtype=np.float32)

    return input_tensor, output_tensor

def train_model(data_path="clapping-data.json", tflite_path="clap_model.tflite"):
    """Loads data, defines, trains, and exports the model."""
    print(f"Loading data from {data_path}...")
    try:
        with open(data_path, "r") as f:
            gestures = json.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please export the data from the HTML file and place it in the same directory.")
        return

    print("Preparing training data...")
    input_tensor, output_tensor = prepare_data(gestures)

    if input_tensor is None:
        print("Not enough data to form a complete window for training.")
        return

    print(f"Prepared {len(input_tensor)} training samples.")

    # Define the 1D CNN model architecture
    model = Sequential([
        Conv1D(filters=16, kernel_size=5, activation='relu', input_shape=INPUT_SHAPE),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    print("\nTraining model...")
    model.fit(input_tensor, output_tensor, epochs=50, shuffle=True, validation_split=0.2)

    print("\nTraining complete! Converting to TensorFlow Lite...")

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Model saved successfully to {tflite_path}")
    print("To use the model, you will need to quantize it and convert it to a C array.")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 1D CNN model for clapping detection.")
    parser.add_argument("--data_path", type=str, default="clapping-data.json",
                        help="Path to the training data JSON file (default: clapping-data.json)")
    parser.add_argument("--output_path", type=str, default="clap_model.tflite",
                        help="Path to save the output TFLite model (default: clap_model.tflite)")
    args = parser.parse_args()

    train_model(data_path=args.data_path, tflite_path=args.output_path)
