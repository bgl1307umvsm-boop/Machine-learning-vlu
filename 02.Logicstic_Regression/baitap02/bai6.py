import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

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

precision_1 = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

recall_1 = recall_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

precision_0 = precision_score(
    y_test,
    y_pred,
    pos_label=0,
    zero_division=0
)

recall_0 = recall_score(
    y_test,
    y_pred,
    pos_label=0,
    zero_division=0
)

print("=== LỚP 1: QUA MÔN ===")
print("Precision =", precision_1)
print("Recall =", recall_1)

print("\n=== LỚP 0: RỚT MÔN ===")
print("Precision =", precision_0)
print("Recall =", recall_0)

print("\nNhận xét:")
print("Hai bộ số khác nhau vì khi đổi lớp dương,")
print("TP, FP, FN cũng đổi ý nghĩa.")
print("Mô hình và dữ liệu không thay đổi,")
print("nhưng lớp đang được quan tâm đã thay đổi.")