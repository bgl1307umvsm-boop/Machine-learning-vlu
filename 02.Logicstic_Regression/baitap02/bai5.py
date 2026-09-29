import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

thu_muc_hien_tai = Path(__file__).resolve().parent
file_csv = thu_muc_hien_tai.parent / "data" / "sinh_vien.csv"

df = pd.read_csv(file_csv)

X = df[["gio_on_tap"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]

nguong_tot_nhat = 0
f1_tot_nhat = 0

print("NGƯỠNG\tF1")

for nguong in np.arange(0.05, 1.0, 0.05):

    y_pred = (y_prob >= nguong).astype(int)

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print(f"{nguong:.2f}\t{f1:.4f}")

    if f1 > f1_tot_nhat:
        f1_tot_nhat = f1
        nguong_tot_nhat = nguong

print("\nNgưỡng có F1 cao nhất:", round(nguong_tot_nhat, 2))
print("F1 cao nhất:", round(f1_tot_nhat, 4))

if abs(nguong_tot_nhat - 0.5) < 0.001:
    print("Nhận xét: Ngưỡng tốt nhất bằng 0.5.")
else:
    print("Nhận xét: Ngưỡng tốt nhất không bằng 0.5.")