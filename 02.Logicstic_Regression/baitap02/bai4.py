import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

y_pred = model.predict(X_test)

TP = 0
TN = 0
FP = 0
FN = 0

for thuc_te, du_doan in zip(y_test, y_pred):
    if thuc_te == 1 and du_doan == 1:
        TP += 1
    elif thuc_te == 0 and du_doan == 0:
        TN += 1
    elif thuc_te == 0 and du_doan == 1:
        FP += 1
    elif thuc_te == 1 and du_doan == 0:
        FN += 1

accuracy = (TP + TN) / (TP + TN + FP + FN)

precision = TP / (TP + FP) if (TP + FP) != 0 else 0

recall = TP / (TP + FN) if (TP + FN) != 0 else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) != 0
    else 0
)

print("TP =", TP)
print("TN =", TN)
print("FP =", FP)
print("FN =", FN)

print("Accuracy =", accuracy)
print("Precision =", precision)
print("Recall =", recall)
print("F1 =", f1)