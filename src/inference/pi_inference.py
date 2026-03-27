import cv2
import threading
import time
import paho.mqtt.client as mqtt
from paho.mqtt import client as mqtt_client
from ultralytics import YOLO
import logging  
'''
This code simulates a real-time video processing pipeline on a Raspberry Pi,
where one thread continuously captures frames from an RTSP stream (the "Receptionist") 
and another thread processes the latest frame with an AI model. 
The Receptionist ensures that the AI always works with the most recent frame, 
effectively implementing a "Drop Frame" strategy to avoid lag.
'''


# Configuration for logging: timestamp, log level, and message
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class ChassisScanner:
    '''
    Docstring for ChassisScanner
    This class encapsulates the functionality of a chassis scanner system that captures video frames from an RTSP stream and publishes results via MQTT.
    It consists of two main components:
    
    1. Reception (getting the latest frame): Continuously captures frames from the RTSP stream and updates the latest frame in a thread-safe manner.
    2. MQTT (sending back the results): Connects to an MQTT broker (a broker is a server that facilitates communication between clients) to publish the results of the AI analysis.
    
    
    The system is designed to ensure that the AI always processes the most recent frame, implementing a "Drop Frame" to minimize latency.
    The drop frame means that if the AI is still processing a previous frame when a new frame arrives, the older frame will be discarded and replaced with the new one.
    '''
    def __init__(self, rtsp_url, mqtt_host):
        # Reception
        self.cap = cv2.VideoCapture(rtsp_url)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        
        #  MQTT 
        self.mqtt_client = mqtt.Client(mqtt_client.CallbackAPIVersion.VERSION2,"RPi_Scanner_Chassis")
        try:
            self.mqtt_client.connect(mqtt_host, 1883)
            logging.info(f"Broker MQTT connected on: {mqtt_host}")
        except Exception as e:
            logging.error(f"MQTT Connection failed : {e}")
        # Model
        self.model = YOLO(model_path, task='detect')
        
    def start_receptionist(self): 
        """Start the receptionist thread that continuously captures frames from the RTSP stream."""
        thread = threading.Thread(target=self._update_loop, daemon=True)
        thread.start()
        return self

    def _update_loop(self):
        """Permanent Drop Frame if a new one arrives"""
        while self.running:
            ret, new_frame = self.cap.read()
            if not ret:
                logging.warning("Warning: Unable to read frame from RTSP stream.")
                time.sleep(1)
                continue
            with self.lock:
                self.frame = new_frame

    def get_latest_frame(self):
        """The buffer with last image"""
        with self.lock:
            return self.frame

    def publish_result(self, status):
        """Send results using MQTT"""
        topic = "usine/ligne1/scanner/resultat"
        self.mqtt_client.publish(topic, status)
        logging.info(f"Result Sent : {status}")

    def stop(self):
        """When the program is stopped, release resources and disconnect MQTT."""
        self.running = False
        self.cap.release()
        self.mqtt_client.disconnect()
        cv2.destroyAllWindows()

# Main 
if __name__ == "__main__":
    PC_IP = "192.168.1.10" 
    RTSP_URL = f"rtsp://{PC_IP}:8554/chassis"
    model_path="best_openvino_model"
    scanner = ChassisScanner(RTSP_URL, PC_IP).start_receptionist()
    print("initialization done")

    try:
        while True:
            img = scanner.get_latest_frame()

            if img is not None:
                cv2.imwrite("debug_pi.jpg",img)
                print(f"format d'image {img.shape}, type: {img.dtype} \n")
                img_resized = cv2.resize(img, (640, 640))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_normalized = img_rgb.astype("float32") / 255.0
                results = scanner.model(img_normalized,conf=0.05, verbose=False)
                names = scanner.model.names
                detections = []
                if len(results[0].boxes)>0:
                    for box in results[0].boxes:
                        class_id = int(box.cls[0])
                        class_name = names[class_id]
                        confidence = float(box.conf[0])
                        detections.append(f"{class_name}, (cnfidence:{confidence:.2f})")
                    verdict = ", ".join(detections)
                else:
                    verdict = "no object"
                print(f"verdict:{verdict}")
                scanner.publish_result(verdict)
                
            if cv2.waitKey(1) & 0xFF == ord('q'): # Quit if 'q' is pressed
                break

    except KeyboardInterrupt:
        logging.info("Stpping...")
    finally:
        scanner.stop()
