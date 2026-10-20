
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("hai_vang_trang.csv")

X = df[["x1", "x2"]]
y = df["nhan"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

for k in [1, 13, 101]:
    model1 = KNeighborsClassifier(n_neighbors=k)
    model1.fit(X_train, y_train)

    y_pred1 = model1.predict(X_test)
    acc1 = accuracy_score(y_test, y_pred1)

    model2 = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance"
    )
    model2.fit(X_train, y_train)

    y_pred2 = model2.predict(X_test)
    acc2 = accuracy_score(y_test, y_pred2)

    print("k =", k)
    print("Khong trong so:", round(acc1, 4))
    print("Co trong so:", round(acc2, 4))
    print("-------------------------")
