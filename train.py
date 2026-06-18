import os, cv2, numpy as np, matplotlib.pyplot as plt
import seaborn as sns, pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

os.makedirs("model", exist_ok=True)

# ---------- 1. Load Dataset with augmentation ----------
path = "LandingDataset"
labels, X, Y = [], [], []

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(('.jpg','.jpeg','.png')) and 'Thumbs' not in file:
            img = cv2.imread(os.path.join(root, file))
            if img is None: continue
            img = cv2.resize(img, (96, 96))
            X.append(img)
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name)
            Y.append(labels.index(name))

print("Labels found:", labels)
print("Total images:", len(X))

X = np.asarray(X).astype('float32') / 255
Y = to_categorical(np.asarray(Y))

# ---------- 2. Train-Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y)
print("Train size:", X_train.shape[0])
print("Test size :", X_test.shape[0])

# ---------- 3. Data Augmentation ----------
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)
datagen.fit(X_train)

# ---------- 4. MobileNetV2 Model ----------
base = MobileNetV2(include_top=False, weights='imagenet',
                   input_shape=(96,96,3), pooling='avg')
for layer in base.layers:
    layer.trainable = False

model = Sequential([
    base,
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------- 5. Train ----------
checkpoint = ModelCheckpoint('model/best_weights.h5',
                             save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy',
                           patience=8, restore_best_weights=True)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# ---------- 6. Evaluate CNN ----------
pred = np.argmax(model.predict(X_test), axis=1)
true = np.argmax(y_test, axis=1)

print("\n========== CNN MODEL RESULTS ==========")
print(f"Accuracy  : {accuracy_score(true,pred)*100:.2f}%")
print(f"Precision : {precision_score(true,pred,average='macro')*100:.2f}%")
print(f"Recall    : {recall_score(true,pred,average='macro')*100:.2f}%")
print(f"F1 Score  : {f1_score(true,pred,average='macro')*100:.2f}%")

# ---------- 7. Hybrid Random Forest ----------
feat_model = Model(model.inputs, model.layers[-4].output)
features   = feat_model.predict(X)
Y_flat     = np.argmax(Y, axis=1)

Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(
    features, Y_flat, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(Xf_tr, yf_tr)
rf_pred = rf.predict(Xf_te)

print("\n========== HYBRID MODEL RESULTS ==========")
print(f"Accuracy  : {accuracy_score(yf_te,rf_pred)*100:.2f}%")
print(f"Precision : {precision_score(yf_te,rf_pred,average='macro')*100:.2f}%")
print(f"Recall    : {recall_score(yf_te,rf_pred,average='macro')*100:.2f}%")
print(f"F1 Score  : {f1_score(yf_te,rf_pred,average='macro')*100:.2f}%")

# ---------- 8. Confusion Matrix ----------
cm = confusion_matrix(yf_te, rf_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix — MobileNetV2 + Random Forest")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ---------- 9. Training Curves ----------
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],     label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Accuracy"); plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'],     label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Loss"); plt.legend()
plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

# ---------- 10. Save ----------
pickle.dump(labels, open('labels.pkl','wb'))
pickle.dump(rf,     open('model/rf_model.pkl','wb'))
print("\nModel and labels saved successfully!")
print("Labels:", labels)