
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("gia_nha.csv")

X = df[["dien_tich", "so_phong", "tuoi_nha"]]
y = df["gia"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

best_k = 1
best_mae = float("inf")

for k in range(1, 21):
    model = KNeighborsRegressor(n_neighbors=k)

    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring="neg_mean_absolute_error"
    )

    mae = -scores.mean()

    print("k =", k, "MAE =", round(mae, 4))

    if mae < best_mae:
        best_mae = mae
        best_k = k

print("k tot nhat:", best_k)
print("MAE kiem dinh cheo:", round(best_mae, 4))

model_knn = KNeighborsRegressor(n_neighbors=best_k)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)

mae_knn = mean_absolute_error(y_test, y_pred_knn)

model_linear = LinearRegression()
model_linear.fit(X_train, y_train)
y_pred_linear = model_linear.predict(X_test)

mae_linear = mean_absolute_error(y_test, y_pred_linear)

print("MAE KNN:", round(mae_knn, 4))
print("MAE duong thang:", round(mae_linear, 4))

if mae_knn < mae_linear:
    print("KNN tot hon duong thang")
else:
    print("KNN khong tot hon duong thang")
