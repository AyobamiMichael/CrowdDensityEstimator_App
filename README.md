🎯 Project Title: Face Counting and Crowd Density Estimation using MTCNN
🔍 Overview
This project addresses the problem of estimating the number of faces in an image, with the added goal of classifying crowd density as Sparse, Medium, or Dense. Initially approached as a regression task, we shifted strategies after empirical evaluation showed that using ground-truth bounding box annotations from the WIDER FACE dataset provided more reliable face counts than a trained regressor. We ultimately adopted the MTCNN (Multi-task Cascaded Convolutional Networks) approach for real-time face detection due to its robustness, accuracy, and speed in varying crowd scenarios.

🧠 Why MTCNN?
MTCNN is a popular face detection framework because:

It combines face detection and facial landmark localization, making it suitable for fine-grained analysis.

It performs well across scales, poses, and lighting conditions.

It runs in real-time, enabling its use in live applications like webcam feeds.

It's pretrained and optimized, saving time and training resources.

💡 How It Works
A user uploads an image or uses their webcam to capture one.

The app runs MTCNN to detect faces and count bounding boxes.

Based on the count:

Sparse: 1–10 faces

Medium: 11–50 faces

Dense: 51+ faces

Results are overlaid directly on the image, offering a visual interpretation of the density.

🛠️ Tech Stack
Model: facenet-pytorch MTCNN

Dataset: WIDER FACE (for validation and benchmarking)

Framework: Python, Streamlit

Libraries: OpenCV, PIL, Torch, Matplotlib

Deployment: Streamlit (local or cloud-hosted)

🌍 Real-Life Applications
Surveillance & Public Safety
Detect unusually dense crowds in public areas to trigger alerts.

Event Management
Monitor real-time foot traffic and optimize crowd control in concerts, rallies, etc.

Retail Analytics
Gauge customer distribution across zones in malls or stores.

Transportation Hubs
Analyze crowd density in airports or stations to deploy personnel dynamically.

Smart Cities
Integrated with CCTV systems, the app can be part of intelligent urban monitoring.

Pandemic Safety Enforcement
Identify when crowd limits are exceeded to enforce health protocols.

✅ Impact
This project demonstrates how a lightweight, pretrained architecture like MTCNN can replace heavyweight regression-based models when data annotation is reliable. It bridges computer vision with social and safety applications, showing how academic tools can translate into real-world solutions.

