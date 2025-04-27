# Sign Language Identification Application using CNN

## Overview
The Sign Language Recognition Android Application enables real-time translation of American Sign Language (ASL) gestures into text using deep learning models optimized for mobile deployment.  
The system integrates **TensorFlow Lite** models for hand detection and gesture classification with **OpenCV**-based live camera processing, delivering efficient and accurate on-device inference without requiring an internet connection.

---

## Key Features
- **Real-Time Recognition**: Live detection and classification of hand gestures using dual TensorFlow Lite models.
- **Efficient Deep Learning Backbone**: Gesture classification powered by an EfficientNetB0 feature extractor.
- **Robust Performance**: Supports recognition of **24+ alphabet signs** under diverse lighting conditions.
- **Low-Latency Inference**: Achieves sub-100ms prediction latency per frame for smooth user experience.
- **Interactive Interface**: Allows users to combine recognized letters into full words and sentences.
- **Fully On-Device**: No server or internet dependency post-installation, ensuring data privacy and offline usability.

---

## Tech Stack
- **Mobile Development**: Android (Java)
- **Machine Learning**: TensorFlow Lite, EfficientNetB0, Python (Keras/TensorFlow)
- **Computer Vision**: OpenCV for Android

---

## How It Works
1. **Hand Detection**: The application first identifies the hand region within each video frame using a custom object detection model (`hand_model.tflite`).
2. **Gesture Classification**: The detected hand region is then passed to a separate classifier model (`sign_language_model.tflite`) to predict the corresponding ASL letter.
3. **Result Display and Sentence Formation**: Recognized letters are shown live, and users can construct words or sentences by adding individual characters through the app interface.

---

## Future Enhancements
- Expand the classification model to support dynamic signs and full-word detection.
- Integrate real-time text-to-speech (TTS) features for auditory output.
- Improve model robustness across varied hand shapes, backgrounds, and environmental conditions.

