from flask import Flask, render_template, request
import cv2, numpy as np, os, pickle, base64, json
from datetime import datetime
from tensorflow.keras.models import load_model, Model
from werkzeug.utils import secure_filename
from gradcam import make_gradcam_heatmap, overlay_heatmap

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HISTORY_FILE = 'history.json'

print("Loading model...")
model      = load_model('model/best_weights.h5')
feat_model = Model(model.inputs, model.layers[-4].output)
labels     = pickle.load(open('labels.pkl', 'rb'))
rf         = pickle.load(open('model/rf_model.pkl', 'rb'))
print("Model loaded! Labels:", labels)

risk_profiles = {
    "road":     {"score":2,  "level":"Low Risk",     "color":"#1D9E75", "action":"Safe to land. Proceed with standard descent protocol."},
    "grass":    {"score":3,  "level":"Low-Med Risk", "color":"#BA7517", "action":"Conditionally safe. Reduce descent speed, check for soft ground."},
    "forest":   {"score":9,  "level":"Critical",     "color":"#A32D2D", "action":"ABORT — dense canopy detected. Redirect immediately."},
    "building": {"score":7,  "level":"High Risk",    "color":"#D85A30", "action":"Avoid — structural hazards and legal restrictions apply."},
    "water":    {"score":10, "level":"Critical",     "color":"#A32D2D", "action":"ABORT — water contact causes irreversible drone damage."},
    "rocky":    {"score":7,  "level":"High Risk",    "color":"#D85A30", "action":"Landing gear damage likely. Seek alternate flat terrain."},
}


def log_prediction(entry):
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            history = []
    history.insert(0, entry)
    history = history[:50]
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/history')
def history():
    records = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                records = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            records = []
    return render_template('history.html', records=records)


@app.route('/predict', methods=['POST'])
def predict():
    filename = None

    if 'image_data' in request.form and request.form['image_data']:
        image_data = request.form['image_data']
        header, encoded = image_data.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        filename = "camera_capture.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(img_bytes)
        source = "Live Camera"

    elif 'image' in request.files:
        file     = request.files['image']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        source = "Image Upload"

    else:
        return "No image provided", 400

    img         = cv2.imread(filepath)
    img_resized = cv2.resize(img,(96,96)).reshape(1,96,96,3).astype('float32')/255
    features    = feat_model.predict(img_resized)
    cnn_probs   = model.predict(img_resized)[0]
    rf_result   = rf.predict(features)[0]
    label       = labels[rf_result]
    confidence  = float(np.max(cnn_probs)) * 100
    risk        = risk_profiles.get(label, {"score":5,"level":"Unknown","color":"gray","action":"Manual verification needed."})

    top3_idx = np.argsort(cnn_probs)[::-1][:3]
    top3     = [(labels[i], round(float(cnn_probs[i])*100,1)) for i in top3_idx if i < len(labels)]

    low_confidence = confidence < 60

    heatmap_filename = "heatmap_" + filename
    heatmap_filepath = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
    try:
        heatmap = make_gradcam_heatmap(img_resized, model, "Conv_1")
        overlay_heatmap(filepath, heatmap, heatmap_filepath)
        heatmap_available = True
        print("Grad-CAM heatmap generated successfully")
    except Exception as e:
        print("Grad-CAM failed:", repr(e))
        heatmap_available = False

    log_prediction({
        "timestamp":  datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
        "source":     source,
        "terrain":    label.upper(),
        "confidence": round(confidence,1),
        "risk_score": risk['score'],
        "risk_level": risk['level'],
        "risk_color": risk['color'],
        "image_file": filename,
        "low_confidence": low_confidence
    })

    return render_template('result.html',
        terrain    = label.upper(),
        confidence = round(confidence,1),
        risk_score = risk['score'],
        risk_level = risk['level'],
        risk_color = risk['color'],
        action     = risk['action'],
        image_file = filename,
        top3       = top3,
        low_confidence    = low_confidence,
        heatmap_available = heatmap_available,
        heatmap_file      = heatmap_filename
    )


if __name__ == '__main__':
    app.run(debug=True)