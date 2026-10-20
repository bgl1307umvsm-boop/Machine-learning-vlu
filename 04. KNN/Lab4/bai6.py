
import numpy as np
import time
from sklearn.neighbors import KNeighborsRegressor

rng = np.random.default_rng(0)

for n in [1000, 10000, 100000]:
    X_train = rng.random((n, 2))
    y_train = X_train[:, 0] + X_train[:, 1]

    X_test = rng.random((1000, 2))

    model = KNeighborsRegressor(n_neighbors=5)

    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - start

    print("So mau:", n)
    print("Thoi gian fit:", round(fit_time, 4), "giay")
    print("Thoi gian predict:", round(predict_time, 4), "giay")
    print("-------------------------")
