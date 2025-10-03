# Work Study Analysis with Computer Vision

This project is a Python application that utilizes **Computer Vision** and **Artificial Intelligence** techniques to analyze work processes from video recordings. It specifically tracks the movements of employees in environments like factories, warehouses, or offices, providing critical data for efficiency and process improvement studies.

## 🚀 Features

* **Employee Detection and Tracking:** Uses the **YOLOv8** model for person detection and the **DeepSORT** algorithm to track each worker with a unique ID.
* **Work Duration Analysis:** Calculates the total number of frames for each employee ID and estimates the **working duration in seconds** based on the video's FPS (Frames Per Second) value.
* **Movement Heatmap:** Monitors the employees' locations within the video to generate a **Heatmap** that visualizes the areas where they spend the most time. This helps identify bottlenecks or unnecessary movements.
* **User Interface:** Features a modern and user-friendly interface built with the **CustomTkinter** library.
* **Reporting:** Provides the capability to save the analysis results (working durations and heatmap image) to a user-specified location.

## 🛠️ Technologies Used

* **Python:** The core programming language for the project.
* **YOLOv8 (Ultralytics):** A fast and lightweight model used for object detection.
* **DeepSORT:** The algorithm employed for tracking the detected objects (people).
* **OpenCV (`cv2`):** Used for video reading, processing, and displaying results.
* **NumPy / SciPy:** Used for mathematical operations, especially for applying a **Gaussian Filter** to the Heatmap data.
* **Matplotlib / PIL:** Used for generating and saving the Heatmap visualization.
* **CustomTkinter (`ctk`):** Used to create the modern and customizable desktop graphical user interface (GUI).
