
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

for metric in ["euclidean", "manhattan"]:
    best_k = 1
    best_acc = 0

    for k in range(1, 32, 2):
        model = KNeighborsClassifier(
            n_neighbors=k,
            metric=metric
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(metric, "k =", k, "accuracy =", round(acc, 4))

        if acc > best_acc:
            best_acc = acc
            best_k = k

    print("Khoang cach:", metric)
    print("k tot nhat:", best_k)
    print("Accuracy tot nhat:", round(best_acc, 4))
    print("-------------------------")
