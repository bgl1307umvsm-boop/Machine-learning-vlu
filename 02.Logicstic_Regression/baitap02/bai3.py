import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[2], [4], [6], [8], [10], [12], [14], [16], [18], [20]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

def du_doan(gio):
    x = np.array([[gio]])
    xac_suat = model.predict_proba(x)[0][1]
    nhan = 1 if xac_suat >= 0.5 else 0

    print(f"{gio} giờ -> xác suất qua môn: {xac_suat:.4f}, nhãn: {nhan}")

for gio in [3, 8, 12.89, 18, 26]:
    du_doan(gio)