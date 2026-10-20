
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("sinh_vien_mo_rong.csv")

X = df[["gio_on", "diem_giua_ky", "chi_tieu"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model1 = KNeighborsClassifier(n_neighbors=7)
model1.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)
acc1 = accuracy_score(y_test, y_pred1)

print("Accuracy truoc khi doi:", round(acc1, 4))

df["chi_tieu"] = df["chi_tieu"] / 1000

X = df[["gio_on", "diem_giua_ky", "chi_tieu"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model2 = KNeighborsClassifier(n_neighbors=7)
model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)
acc2 = accuracy_score(y_test, y_pred2)

print("Accuracy sau khi doi:", round(acc2, 4))
