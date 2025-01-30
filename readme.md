# Life2O2

Life2O2 is a multimodal system that integrates audio and video inputs with a Partial Differential Equation (PDE) based neural network to generate dynamic visualizations and audio outputs. Utilizing real-time data from your microphone and webcam, Life2O2 processes the information through a neural PDE framework, offering an interactive and immersive experience.

# Features

Real-Time Audio Processing: Captures and processes audio input from your microphone.
Live Video Integration: Streams and processes live video feed from your webcam.
Neural PDE Framework: Utilizes a neural network-based PDE to manipulate and integrate multimodal data.
Dynamic Visualization: Visualizes the PDE field in real-time using Matplotlib.
Interactive GUI: Provides a simple Tkinter-based interface to select audio devices and camera.
Installation

# Clone the Repository

git clone https://github.com/anttiluode/life2o2.git
cd life2o2
Create a Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies

pip install -r requirements.txt

# Run the Application

python main.py

Configure Devices

Upon launching, a GUI will appear allowing you to select your audio input/output devices and webcam index.
Select the desired devices and click Start.
Interact with the System

The PDE visualization will appear in a Matplotlib window.
The system will process audio and video inputs in real-time, reflecting changes in the visualization and audio output.
Exit

Close the Matplotlib window to stop the application gracefully.

# Requirements

Python 3.7 or higher

See requirements.txt for Python package dependencies.

Dependencies

numpy
torch
sounddevice
opencv-python
matplotlib
License

MIT License
