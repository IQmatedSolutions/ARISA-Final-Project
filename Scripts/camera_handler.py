# importing required libraries 
import cv2 
import time 
import threading 
from picamera2 import Picamera2
import argparse
import os
from ultralytics import YOLO
import numpy as np

# defining a helper class for implementing multi-threaded processing 
class WebcamStream:
    def __init__(self, stream_id=0):
        self.stream_id = stream_id
        self.buzzer_pin_id = 23
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--model', help='Path to Yolo model')
        self.args = self.parser.parse_args()
        self.model_path = self.args.model
        if not os.path.exists(self.model_path):
            print('ERROR: Model path is invalid or model was not found!')
            exit(0)

        self.model = YOLO(self.model_path, task='detect')
        self.labels = self.model.names
        self.bbox_colors = [(164,120,87),(64,148,228),(93,97,209),(178,182,133),(88,159,106),(96,202,231),
                             (159,124,168),(169,162,241),(98,118,150),(172,176,184)]

        self.avg_frame_rate = 0
        self.frame_rate_buffer = []
        self.fps_avg_len = 200

        self.vcap = Picamera2()
        max_size = self.vcap.camera_properties['PixelArraySize']
        self.vcap.configure(self.vcap.create_video_configuration(main={"format": 'RGB888', "size": (640, 480)},raw={'size': (max_size[0], max_size[1])}))
        self.vcap.start()

        self.frame = self.vcap.capture_array()
        if self.frame is None:
            print('[Exiting] No more frames to read')
            exit(0)

        self.stopped = False
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True

    def start(self):
        self.t.start()

    def calculate_risk(self,classname):
        if classname in ('Kokos', 'Konrad','Magda', 'Delivery-Man'):
            risk = 0
        elif classname == 'fire':
            risk = 0.7
        elif classname in ('Threat', 'gun', 'knife'):
            risk = 1
        elif classname == 'Person' and classname not in ('Threat', 'fire', 'gun', 'knife'):
            risk = 0.5
        return risk

    def activate_alarm(self, message):
        # Send message to homeowners about risk
        import requests
        phone = os.environ['PHONE']
        api_key = os.environ['APIKEY']
        requests.request(method="GET",url=f"https://api.callmebot.com/whatsapp.php?phone={phone}&text={message}&apikey={api_key}")

        ### SOUND A BUZZER ###
        # Set up the GPIO mode to use BCM numbering
        # BCM refers to the Broadcom SOC channel numbers
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)

        # Define the GPIO pin connected to the buzzer signal line (e.g., GPIO 23)
        GPIO.setup(self.buzzer_pin_id, GPIO.OUT)

        print("Buzzer test starting...")

        try:
            while True:
                # Turn the buzzer ON (set the pin HIGH)
                # Note: Some buzzers are "active-low" and will sound with GPIO.LOW
                GPIO.output(self.buzzer_pin_id, GPIO.HIGH)
                print("Beep ON")
                time.sleep(0.5)  # Wait for 0.5 seconds

                # Turn the buzzer OFF (set the pin LOW)
                GPIO.output(self.buzzer_pin_id, GPIO.LOW)
                print("Beep OFF")
                time.sleep(0.5)  # Wait for 0.5 seconds

        except KeyboardInterrupt:
            # Cleanup on Ctrl+C exit
            print("Exiting and cleaning up GPIO...")
            GPIO.cleanup()

    def update(self):
        start_time = time.perf_counter()
        while not self.stopped:
            t_start = time.perf_counter()
            # Capture the new frame
            new_frame = self.vcap.capture_array()

            # Analyze and annotate the new frame
            results = self.model(new_frame, verbose=False)
            detections = results[0].boxes
            object_count = 0

            annotated_frame = new_frame.copy()

            for i in range(len(detections)):
                xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy

                classidx = int(detections[i].cls.item())

                if classidx in self.labels:
                    classname = self.labels[classidx]
                else:
                    classname = "Unknown"

                risk = self.calculate_risk(classname)

                conf = detections[i].conf.item()

                if conf > 0.5:
                    color = self.bbox_colors[classidx % len(self.bbox_colors)]
                    cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), color, 2)

                    label = f'{classname}: {int(conf * 100)}%; Risk score: {int(risk * 100)}%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(ymin, labelSize[1] + 10)

                    cv2.rectangle(annotated_frame, (xmin, label_ymin - 7), (xmin + labelSize[0], label_ymin + baseLine), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, label, (xmin, label_ymin + baseLine), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    object_count += 1
                    if risk >= 0.7:
                        message = f'There+was+a+{classname}+object+observed+in+camera.+Alarm+activation+process+started!'
                        self.activate_alarm(message)

            # 💡 Correct FPS calculation happens here
            t_stop = time.perf_counter()
            frame_rate_calc = float(1/(t_stop - t_start))

            if len(self.frame_rate_buffer) >= self.fps_avg_len:
                self.frame_rate_buffer.pop(0)
            self.frame_rate_buffer.append(frame_rate_calc)

            self.avg_frame_rate = np.mean(self.frame_rate_buffer)

            cv2.putText(annotated_frame, f'FPS: {self.avg_frame_rate:0.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f'Number of objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)

            self.lock.acquire()
            self.frame = annotated_frame
            self.lock.release()

    def read(self):
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()
        return frame

    def stop(self):
        self.stopped = True
        self.t.join()
        self.vcap.stop()

# initializing and starting multi-threaded webcam capture input stream 
webcam_stream = WebcamStream(stream_id=0)
webcam_stream.start()

num_frames_processed = 0
start = time.time()

while True:
    frame = webcam_stream.read()

    cv2.imshow('YOLO detection results', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    num_frames_processed += 1

end = time.time()
webcam_stream.stop()

elapsed = end - start
fps = num_frames_processed / elapsed 
print(f"FPS: {fps:.2f}, Elapsed Time: {elapsed:.2f}, Frames Processed: {num_frames_processed}")

cv2.destroyAllWindows()
