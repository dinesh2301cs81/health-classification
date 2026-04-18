from flask import Flask, render_template, Response, jsonify
import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# -------- CAMERA + MODEL --------
class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 320)
        self.cap.set(4, 240)

        self.interpreter = tflite.Interpreter(model_path="model.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.latest_prediction = "None"

    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            return None

        frame = self.run_model(frame)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def run_model(self, frame):
        input_shape = self.input_details[0]['shape']
        h, w = input_shape[1], input_shape[2]

        img = cv2.resize(frame, (w, h))
        img = np.expand_dims(img, axis=0).astype('float32') / 255.0

        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        pred = int(np.argmax(output))

        self.latest_prediction = str(pred)

        cv2.putText(frame, f"Pred: {pred}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        return frame

camera = Camera()

# -------- SENSOR (SIMULATED) --------
def get_heart():
    return 70 + int(time.time()) % 10

def get_temp():
    return round(36 + (time.time() % 5)*0.2, 2)

# -------- ROUTES --------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    def gen():
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    return jsonify({
        "heart_rate": get_heart(),
        "temperature": get_temp(),
        "prediction": camera.latest_prediction,
        "timestamp": time.strftime("%H:%M:%S")
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
