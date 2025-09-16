Real-Time Streaming Studio - Production Build (Dual-Pod Architecture)
This application provides real-time face-swapping and voice-changing, outputting the result to virtual devices for use in other applications like Zoom or Skype.

Architectural Overview
The application is a client-server system that connects to two separate, dedicated processing pods: one for video (face-swapping) and one for audio (voice-changing). This ensures optimal performance, scalability, and resilience.

Key Features
Minimalist UI: Designed for non-technical users with a clean black, white, and gray theme.

Dual Processing Modes:

Local GPU: Utilizes the user's local hardware by running two separate server processes in the background.

RunPod Cloud: Automatically provisions and connects to two serverless GPU pods on RunPod for users without required hardware.

Intelligent Model Management: Each server (local or cloud) independently downloads the models it needs, with status updates reflected in the UI.

Push-to-Talk: Audio is only processed while the spacebar is held, saving resources.

Dynamic Settings: Users can select camera sources, target faces, and voices in real-time.

How to Run the Application
Install Dependencies:

Create and activate a Python virtual environment.

Install all required packages from requirements.txt:

pip install -r requirements.txt

Install Virtual Devices (One-time setup for end-users):

Virtual Camera: Install a backend driver like the OBS Studio Virtual Camera.

Virtual Audio: Install a virtual audio cable like VB-AUDIO Virtual Cable.

Docker (For RunPod Mode):

Build the server Docker image provided in server/Dockerfile.

Push the image to a container registry (e.g., Docker Hub).

Update the server_image path in config.json to point to your image.

Run the App:

python main.py

Project Structure
main.py: (Updated) Main GUI application. Manages UI, state, and coordinates the connection to both servers.

config.json: Central configuration for models, voices, and server settings.

state_manager.py: (Updated) Centralized state, now tracking the status and endpoints of both servers.

services/:

media_service.py: (Updated) Now handles two separate WebRTC connections to the video and audio servers.

runpod_service.py: (Updated) Manages the lifecycle of two distinct RunPod pods.

local_server_manager.py: (Unchanged) Still correctly starts two separate local server processes.

ui/app_ui.py: (Unchanged)

utils/: (Unchanged)

server/: (Unchanged) The generic server code is run with different parameters for each pod.

requirements.txt: (Unchanged)
