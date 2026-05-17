# train.py
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# --- IMPLEMENTASI EKSPLISIT BACKPROPAGATION MANUAL (NUMPY) ---
class BackpropagationManual:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        np.random.seed(42)
        # Menggunakan He/Xavier initialization agar bobot tidak meledak
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.lr = lr

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1) # Aktivasi ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def backward(self, X, y, output):
        m = X.shape[0]
        error = output - y.reshape(-1, 1)
        
        dW2 = np.dot(self.a1.T, error) / m
        db2 = np.sum(error, axis=0, keepdims=True) / m
        
        da1 = np.dot(error, self.W2.T)
        dz1 = da1 * (self.z1 > 0) # Turunan ReLU
        
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update dengan Clip Gradient untuk mencegah nilai ekstrem
        for grad in [dW2, db2, dW1, db1]:
            np.clip(grad, -1.0, 1.0, out=grad)
            
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def fit(self, X, y, epochs=500):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        return self.forward(X).flatten()

# --- PIPELINE PREPROCESSING & TRAINING ---
print("⏳ Memulai pipeline pembersihan data...")
df = pd.read_csv('dataset.csv')

def clean_harga(val):
    if not isinstance(val, str): return np.nan
    val = val.lower().replace('rp', '').replace('.', '').replace(',', '.').strip()
    if 'miliar' in val:
        return float(val.replace('miliar', '').strip()) * 1_000_000_000
    elif 'juta' in val:
        return float(val.replace('juta', '').strip()) * 1_000_000
    return np.nan

def clean_luas(val):
    if not isinstance(val, str): return np.nan
    try: return float(val.lower().replace('m²', '').replace('m2', '').replace('.', '').replace(',', '.').strip())
    except: return np.nan

df['Harga_Clean'] = df['Harga'].apply(clean_harga)
df['LT_Clean'] = df['Luas Tanah'].apply(clean_luas)
df['LB_Clean'] = df['Luas Bangunan'].apply(clean_luas)

df_clean = df[['LT_Clean', 'LB_Clean', 'Kamar Tidur', 'Kamar Mandi', 'Harga_Clean']].dropna()

X = df_clean[['LT_Clean', 'LB_Clean', 'Kamar Tidur', 'Kamar Mandi']]
y = df_clean['Harga_Clean']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi Fitur (X)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 🔥 Solusi Utama: Normalisasi Target (y) agar angka miliaran tidak meledakkan gradien
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

print("🧠 Memulai pelatihan model (ANN Scikit-Learn & Backpropagation Manual)...")

# 1. Model ANN Scikit-Learn
model_ann = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, learning_rate_init=0.01)
model_ann.fit(X_train_scaled, y_train_scaled)

# 2. Model Backpropagation NumPy Manual
model_backprop = BackpropagationManual(input_size=4, hidden_size=32, output_size=1, lr=0.01)
model_backprop.fit(X_train_scaled, y_train_scaled, epochs=500)

# Evaluasi Nilai Metrik (Prediksi dikembalikan ke skala rupiah asli dulu sebelum dihitung r2_score)
metrics = {}
for name, model in [('Artificial Neural Network (ANN)', model_ann), 
                    ('Backpropagation Model (NumPy)', model_backprop)]:
    preds_scaled = model.predict(X_test_scaled)
    # Kembalikan angka ke skala milyar asli
    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    metrics[name] = {'R2': round(max(0.0, r2), 4), 'MAE': round(mae, 2)}

print("📊 Hasil Evaluasi Model:")
print(json.dumps(metrics, indent=4))

# Simpan semua scaler dan objek model biner
with open('model_ann.pkl', 'wb') as f: pickle.dump(model_ann, f)
with open('model_backprop.pkl', 'wb') as f: pickle.dump(model_backprop, f)
with open('scaler_X.pkl', 'wb') as f: pickle.dump(scaler_X, f)
with open('scaler_y.pkl', 'wb') as f: pickle.dump(scaler_y, f)

with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

print("✅ Pipeline Sukses! Semua file model berhasil diekspor tanpa error overflow.")