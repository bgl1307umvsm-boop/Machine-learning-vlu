
import numpy as np
from collections import Counter

def knn_du_doan(X_hoc, y_hoc, X_moi, k):
    khoang_cach = []

    for i in range(len(X_hoc)):
        kc = np.sqrt(np.sum((X_hoc[i] - X_moi) ** 2))
        khoang_cach.append(kc)

    chi_so = np.argsort(khoang_cach)[:k]
    nhan = y_hoc[chi_so]

    dem = Counter(nhan)
    return dem.most_common(1)[0][0]


# Du lieu vi du de kiem tra
X_hoc = np.array([
    [1, 2],
    [2, 3],
    [7, 8],
    [8, 9]
])

y_hoc = np.array([0, 0, 1, 1])

X_moi = np.array([2, 2])
k = 3

ket_qua = knn_du_doan(X_hoc, y_hoc, X_moi, k)
print("Nhan du doan:", ket_qua)
