Berikut adalah konversi dokumen LaTeX ke dalam format Markdown. Markdown tidak mendukung semua fitur LaTeX seperti tabel kompleks, gambar, atau kode dengan gaya khusus secara langsung, tetapi saya akan menyederhanakannya agar sesuai dengan kemampuan Markdown sambil mempertahankan struktur dan konten utama.

---

# Prediksi Kurs Rupiah terhadap Yen menggunakan RNN dan LSTM

**Penulis:** Muhammad Farhan Arya Wicaksono  
**NRP:** 5054231011  
**Laporan Tugas Mata Kuliah:** *Deep Learning*  

---

## Pendahuluan

Laporan ini membahas dan membandingkan hasil prediksi kurs Rupiah terhadap Yen dengan menggunakan model **RNN (Recurrent Neural Network)** dan **LSTM (Long Short-Term Memory)**. Model *Deep Learning* berbasis sekuens ini digunakan untuk menangkap pola dalam data *time series* dan dapat memperkirakan nilai tukar di masa depan.

---

## Metodologi

### Data dan Preprocessing

Data diunduh dari [satudata.kemendag.go.id](https://satudata.kemendag.go.id/data-informasi/perdagangan-dalam-negeri/nilai-tukar) berupa kurs dari beberapa negara di dunia dari periode 2001 hingga 2025, dengan data kurs disajikan dalam format bulanan.

Data mencakup berbagai mata uang seperti USD, JPY, GBP, dll., tetapi untuk analisis ini, hanya data nilai tukar JPY (Jepang) yang digunakan. Kolom "Tahun" dan "Bulan" digabung menjadi satu kolom waktu, dan fitur mata uang relevan dipilih untuk analisis lebih lanjut.

- **Training:** Januari 2001 - Desember 2022  
- **Testing:** Januari 2023 - Desember 2023  

**Preprocessing:**
- Normalisasi menggunakan `MinMaxScaler`
- Pembuatan *sequence* (window = 12 bulan)

### Arsitektur Model

Pembuatan model dilakukan dengan mencoba berbagai konfigurasi (*tuning*) dari jumlah **neuron**, **layer**, **dropout**, **epoch**, **optimizer**, dan **learning rate**. Proses *tuning* diterapkan pada model **RNN** dan **LSTM** dengan konfigurasi serupa untuk membandingkan performa secara adil.

Berikut adalah konfigurasi model yang digunakan:

| Model | Hidden Sizes       | Dropout | Learning Rate | Epochs | Optimizer |
|-------|--------------------|---------|---------------|--------|-----------|
| 1     | [16]              | 0.1     | 0.001         | 10     | Adam      |
| 2     | [32]              | 0.1     | 0.0008        | 15     | Adam      |
| 3     | [64, 128]         | 0.25    | 0.0005        | 25     | Adam      |
| 4     | [64, 128, 256]    | 0.3     | 0.0003        | 30     | RMSprop   |
| 5     | [128, 256]        | 0.35    | 0.0002        | 35     | AdamW     |
| 6     | [128, 256, 512]   | 0.4     | 0.0001        | 40     | Adam      |
| 7     | [256, 512, 1024]  | 0.45    | 0.00005       | 50     | AdamW     |
| 8     | [512, 1024, 2048] | 0.5     | 0.00001       | 60     | Adam      |
| 9     | [128, 256, 512]   | 0.3     | 0.0003        | 30     | SGD       |
| 10    | [64, 128, 256, 512] | 0.4   | 0.0001        | 50     | Adam      |

---

## Implementasi Kode

### Import Library

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import torch.optim as optim
```

### Formatting Data

```python
data = pd.read_excel('kurs_rupiah.xlsx')
hasil = []
current_year = 2025

def ubah_format(listData):
    global current_year
    if isinstance(listData[0], int) or (isinstance(listData[0], str) and listData[0].isdigit()):
        current_year = str(listData[0])
    else:
        hasil.append([str(current_year) + "-" + listData[0], str(listData[1])])

for data in JAPAN.iloc[1:].values:
    ubah_format(data)

clean = pd.DataFrame(hasil, columns=['Tahun', 'JPY'])
clean['JPY'] = clean['JPY'].str.replace(',', '.').astype(float)
clean.to_csv('jpy.csv', index=False)
```

**Output contoh:**
```
Tahun         JPY
2025-Januari  10.52363
2024-Desember 10.23625
2024-November 10.45301
...
2001-Januari  8.13149
```

### Preparing and Preprocessing Data

```python
# Set CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv('jpy.csv')

# Convert month names to English
bulan_indo_ke_eng = {
    "Januari": "January", "Februari": "February", "Maret": "March",
    "April": "April", "Mei": "May", "Juni": "June",
    "Juli": "July", "Agustus": "August", "September": "September",
    "Oktober": "October", "November": "November", "Desember": "December"
}

def convert_to_datetime(date_str):
    tahun, bulan_indo = date_str.split('-')
    bulan_eng = bulan_indo_ke_eng[bulan_indo]
    return datetime.strptime(f"{tahun}-{bulan_eng}", "%Y-%B")

df["tanggal_datetime"] = df["Tahun"].apply(convert_to_datetime)
df['tanggal_datetime'] = pd.to_datetime(df['tanggal_datetime'])
df.set_index('tanggal_datetime', inplace=True)
df = df.sort_index()

# Normalize data
scaler = MinMaxScaler()
df['JPY'] = scaler.fit_transform(df[['JPY']])

# Split data
train_data = df.loc['2001-01-01':'2022-12-01']
test_data = df.loc['2022-01-01':'2023-12-01']

# Create sequences
sequence_length = 12
def create_sequences(data, sequence_length):
    sequences, labels = [], []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]
        label = data[i+sequence_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

train_sequences, train_labels = create_sequences(train_data['JPY'].values, sequence_length)
test_sequences, test_labels = create_sequences(test_data['JPY'].values, sequence_length)

# Convert to PyTorch tensors
X_train = torch.tensor(train_sequences, dtype=torch.float32).unsqueeze(-1).to(device)
y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(-1).to(device)
X_test = torch.tensor(test_sequences, dtype=torch.float32).unsqueeze(-1).to(device)
y_test = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(-1).to(device)
```

### Pembuatan Model

```python
# RNN Model
class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout):
        super(CustomRNN, self).__init__()
        self.num_layers = len(hidden_sizes)
        self.rnn_layers = nn.ModuleList()
        self.rnn_layers.append(nn.RNN(input_size, hidden_sizes[0], batch_first=True))
        for i in range(len(hidden_sizes) - 1):
            self.rnn_layers.append(nn.RNN(hidden_sizes[i], hidden_sizes[i + 1], batch_first=True))
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for rnn in self.rnn_layers:
            x, _ = rnn(x)
            x = self.dropout(x)
        out = self.fc(x[:, -1, :])
        return out

# LSTM Model
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout):
        super(CustomLSTM, self).__init__()
        self.num_layers = len(hidden_sizes)
        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(nn.LSTM(input_size, hidden_sizes[0], batch_first=True))
        for i in range(len(hidden_sizes) - 1):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i], hidden_sizes[i + 1], batch_first=True))
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)
        out = self.fc(x[:, -1, :])
        return out
```

### Model Konfigurasi dan Training

```python
configs = [
    {"hidden_sizes": [16], "dropout": 0.1, "lr": 0.001, "epochs": 10, "optimizer": "Adam"},
    {"hidden_sizes": [32], "dropout": 0.1, "lr": 0.0008, "epochs": 15, "optimizer": "Adam"},
    {"hidden_sizes": [64, 128], "dropout": 0.25, "lr": 0.0005, "epochs": 25, "optimizer": "Adam"},
    {"hidden_sizes": [64, 128, 256], "dropout": 0.3, "lr": 0.0003, "epochs": 30, "optimizer": "RMSprop"},
    {"hidden_sizes": [128, 256], "dropout": 0.35, "lr": 0.0002, "epochs": 35, "optimizer": "AdamW"},
    {"hidden_sizes": [128, 256, 512], "dropout": 0.4, "lr": 0.0001, "epochs": 40, "optimizer": "Adam"},
    {"hidden_sizes": [256, 512, 1024], "dropout": 0.45, "lr": 0.00005, "epochs": 50, "optimizer": "AdamW"},
    {"hidden_sizes": [512, 1024, 2048], "dropout": 0.5, "lr": 0.00001, "epochs": 60, "optimizer": "Adam"},
    {"hidden_sizes": [128, 256, 512], "dropout": 0.3, "lr": 0.0003, "epochs": 30, "optimizer": "SGD"},
    {"hidden_sizes": [64, 128, 256, 512], "dropout": 0.4, "lr": 0.0001, "epochs": 50, "optimizer": "Adam"}
]
```

### Model Implementasi dan Training Loop

```python
for idx, config in enumerate(configs):
    loss_rnn_list, loss_lstm_list = [], []
    hidden_sizes = config["hidden_sizes"]
    dropout = config["dropout"]
    lr = config["lr"]
    epochs = config["epochs"]
    optimizer_name = config["optimizer"]

    model_rnn = CustomRNN(input_size=1, hidden_sizes=hidden_sizes, output_size=1, dropout=dropout).to(device)
    model_lstm = CustomLSTM(input_size=1, hidden_sizes=hidden_sizes, output_size=1, dropout=dropout).to(device)

    optimizer_rnn = optim.Adam(model_rnn.parameters(), lr=lr) if optimizer_name == "Adam" else optim.RMSprop(model_rnn.parameters(), lr=lr)
    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=lr) if optimizer_name == "Adam" else optim.RMSprop(model_lstm.parameters(), lr=lr)

    criterion = nn.MSELoss()

    for epoch in tqdm(range(epochs), desc=f"Model {idx+1} Training Progress"):
        optimizer_rnn.zero_grad()
        optimizer_lstm.zero_grad()
        output_rnn = model_rnn(X_train)
        output_lstm = model_lstm(X_train)
        loss_rnn = criterion(output_rnn, y_train)
        loss_lstm = criterion(output_lstm, y_train)
        loss_rnn.backward()
        loss_lstm.backward()
        optimizer_rnn.step()
        optimizer_lstm.step()
        loss_rnn_list.append(loss_rnn.item())
        loss_lstm_list.append(loss_lstm.item())
```

### Evaluasi dan Hasil

```python
model_rnn.eval()
model_lstm.eval()

with torch.no_grad():
    pred_rnn = model_rnn(X_test)
    pred_lstm = model_lstm(X_test)

mse_rnn = criterion(pred_rnn, y_test).item()
mse_lstm = criterion(pred_lstm, y_test).item()

avg_loss_rnn = sum(loss_rnn_list) / len(loss_rnn_list)
avg_loss_lstm = sum(loss_lstm_list) / len(loss_lstm_list)
```

### Visualisasi dan Plotting

```python
y_test_inv = scaler.inverse_transform(y_test.cpu().reshape(-1, 1))
y_pred_rnn_inv = scaler.inverse_transform(pred_rnn.cpu().reshape(-1, 1))
y_pred_lstm_inv = scaler.inverse_transform(pred_lstm.cpu().reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual', color='black')
plt.plot(y_pred_rnn_inv, label='RNN Prediction', color='blue')
plt.plot(y_pred_lstm_inv, label='LSTM Prediction', color='red')
plt.title('RNN vs LSTM Predictions')
plt.xlabel('Time')
plt.ylabel('JPY Value')
plt.legend()
plt.show()
```

---


# Hasil

![alt text](<assets/hasil percobaan.png>)

![alt text](<assets/plot loss and accuracy.png>)

## Analisis

### Perbandingan Final RNN Loss vs Final LSTM Loss
- LSTM umumnya menunjukkan nilai *loss* lebih rendah dibandingkan RNN, kecuali pada model 1 dan 7.
- Secara keseluruhan, LSTM lebih stabil dalam mengurangi *loss*.

### Pengaruh Jumlah Layer dan Ukuran Lapisan Tersembunyi
- Model dengan 2-3 lapisan (Model 2 dan 3) memberikan kinerja terbaik dengan MSE lebih rendah.
- Ukuran lapisan besar tidak selalu lebih baik (contoh: Model 8).

### Pengaruh Dropout dan Learning Rate
- *Dropout* tinggi (Model 7 dan 8) membantu LSTM, tetapi tidak selalu untuk RNN.
- *Learning rate* rendah (Model 2 dan 5) cenderung baik, terutama pada LSTM.

### Perbandingan Test MSE RNN vs Test MSE LSTM
- LSTM memiliki MSE lebih rendah pada sebagian besar model.
- Optimizer AdamW menunjukkan hasil baik pada kedua model.

### Perbandingan Optimizer
- AdamW terbaik dalam mengurangi MSE dibandingkan SGD.
- Adam dan AdamW efektif, sementara SGD kurang efektif.

### Pengaruh Epochs
- Epoch tinggi (Model 6 dan 7) baik untuk LSTM, tetapi tidak selalu untuk RNN.
- Epoch banyak memungkinkan pembelajaran lebih baik, tapi risiko *overfitting* meningkat.

### Model Terbaik
- Model dengan 2 *hidden layer* ([128, 256]), AdamW, *dropout* 0.35, dan 35 epoch menghasilkan MSE terendah.
- LSTM secara konsisten lebih baik dibandingkan RNN.

---

## Kesimpulan

- **LSTM lebih unggul** dalam akurasi dan stabilitas dibandingkan RNN.
- **AdamW** adalah optimizer terbaik untuk mengurangi MSE.
- Hasil terbaik diperoleh dengan **LSTM menggunakan AdamW** dan konfigurasi model yang lebih dalam.

---

Catatan: Markdown tidak mendukung penyisipan gambar langsung seperti LaTeX. Untuk visualisasi (gambar hasil dan plot), Anda perlu menyisipkannya secara terpisah di platform yang mendukung (misalnya, GitHub atau Jupyter Notebook).