import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

sensor_model = tflite.Interpreter(model_path="sensor_model.tflite")
image_model = tflite.Interpreter(model_path="image_model.tflite")

fever_model = tflite.Interpreter(model_path="fever_model.tflite")
spo2_model = tflite.Interpreter(model_path="spo2_model.tflite")
hr_model = tflite.Interpreter(model_path="hr_model.tflite")
meta_model = tflite.Interpreter(model_path="meta_model.tflite")

models = [sensor_model, image_model, fever_model, spo2_model, hr_model, meta_model]

for m in models:
    m.allocate_tensors()


classes = ['Asthma', 'Diabetes Mellitus', 'Healthy', 'Heart Disease', 'Hypertension']


def run_model(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    return interpreter.get_tensor(output_details[0]['index'])


def get_sensor_data():
    spo2 = float(input("Enter SpO2: "))
    temp = float(input("Enter Temperature: "))
    hr = float(input("Enter Heart Rate: "))
    return spo2, temp, hr


def get_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Camera Error")
        return None
    
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img.astype(np.float32), axis=0)
    
    return img


def main():
    
    spo2, temp, hr = get_sensor_data()
    img = get_image()
    
    if img is None:
        return
    
    sensor_input = np.array([[spo2, temp, hr]], dtype=np.float32)
    
    # Step 2: Run Basic Models
    sensor_out = run_model(sensor_model, sensor_input)
    image_out = run_model(image_model, img)
    
    # Combine score (simple average)
    score = (sensor_out[0][1] + image_out[0][1]) / 2
    
    print("Health Score:", score)
    
  
    if score >= 0.5:
        print("⚠️ Abnormal detected → Running detailed analysis...")
        
        # Run sub-models
        fever_out = run_model(fever_model, np.array([[temp, hr]], dtype=np.float32))
        spo2_out  = run_model(spo2_model, np.array([[spo2]], dtype=np.float32))
        hr_out    = run_model(hr_model, np.array([[hr]], dtype=np.float32))
        
        # Combine outputs
        meta_input = np.concatenate([
            fever_out, spo2_out, hr_out
        ], axis=1).astype(np.float32)
        
        # Final prediction
        meta_out = run_model(meta_model, meta_input)
        pred = np.argmax(meta_out)
        
        print("Predicted Disease:", classes[pred])
    
    else:
        print("✅ Normal Condition")


if __name__ == "__main__":
    main()