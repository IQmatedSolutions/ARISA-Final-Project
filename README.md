# ARISA-Final-Project

Repository contains all configurations, scripts and final YOLO model deployed as part of the "Real Time Home Security System Powered by Computer Vision AI model" project.
The project is a successful proof of concept for use of a computer vision model on edge devices with very limited computational power. 
Use of a CI/CD pipeline and centralized code base allows fast training and deploying of a model in a controlled manner. This system automatically
fetches a custom dataset from an AWS S3 bucket, trains a YOLOv5nu model, and exports the final model to the efficient NCNN format for on-device inference.
The process is orchestrated by GitHub Actions and executed on a dedicated, GPU-enabled runner, which significantly reduces the time required for model training.
The use of Raspberry Pi devices opens the door to interference with external components and makes the whole solution much more robust.
Connection with components such as buzzer, siren, power switch or even servomotors could make the household more secure.

## Structure of the repository
- **.github** - configuration of CI/CD pipelines
- **DSML** - directory containing Python scripts necessary to preprocess and train the YOLO5nu model on custom dataset
- **Scripts** - Python script created to handle the computer vision model inference on Raspberry Pi
- **models** - final version of the YOLO5nu model trained on custom dataset and exported to NCNN framework

## CI/CD Architecture
The implemented solution uses an AWS S3 bucket for storage and a GitHub GPU runner for compute.
Newest version of the model is being pushed to the repository as part of training pipeline and finally deployed to Raspberry Pi.
Deployment is done with use of SSH connection.
<img width="880" height="877" alt="image" src="https://github.com/user-attachments/assets/23edb18c-34b9-4d07-9d0d-4d4776571e3a" />

## Real-time analysis logic
A Python script was developed to analyze real-time video frames with the following logic.
<img width="1239" height="540" alt="image" src="https://github.com/user-attachments/assets/96550ba6-1f2d-4b13-876b-fd9d7958036b" />

## Model evaluation
The model evaluation was conducted as part of the automated training process. The YOLOv5 framework offers a variety of metrics 
but Mean Average Precision (mAP) is considered the most comprehensive and standard metric for evaluating object detection models in computer vision.
<img width="897" height="257" alt="image" src="https://github.com/user-attachments/assets/c0f23768-6542-4486-83d1-aaf6a9aaed7a" />

## Legal and Ethical Issues
The project actively addressed potential legal and ethical concerns related to data privacy. Given that the camera system is intended
for home security and may capture images of people, the solution is explicitly limited to private use. Furthermore, the system is designed
to only record video in the event of a security breach. To meet legal requirements and ensure transparency, the monitored area is clearly
marked with proper signage as suggested by legal guidelines. This approach demonstrates a commitment to responsible AI development by prioritizing
user consent, privacy, and transparency, even for a non-critical application. External datasets used for solution capabilities use CC BY 4.0
license which allows to use the material for any purpose, even commercially.
https://creativecommons.org/licenses/by/4.0/
