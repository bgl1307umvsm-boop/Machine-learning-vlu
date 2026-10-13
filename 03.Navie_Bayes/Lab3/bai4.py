from pathlib import Path
import pandas as pd


def tim_file(ten):
    thu_muc = Path(__file__).resolve().parent
    for duong_dan in [thu_muc.parent / ten, thu_muc / ten,
                      thu_muc.parent / "data" / ten, thu_muc / "data" / ten]:
        if duong_dan.is_file():
            return duong_dan
    raise FileNotFoundError(f"Khong tim thay {ten}. Hay de CSV trong thu muc 03.Navie_Bayes.")


def doc_tin_nhan():
    # header=None de khong lam mat dong du lieu dau neu file khong co tieu de
    df = pd.read_csv(tim_file("tin_nhan.csv"), header=None, encoding="utf-8-sig")
    if df.shape[1] != 2:
        raise ValueError(f"tin_nhan.csv can co 2 cot (noi dung, nhan); tim thay {df.shape[1]} cot.")
    df.columns = ["noi_dung", "nhan"]
    # Neu dong dau la tieu de thi bo di
    if str(df.iloc[0, 1]).strip().lower() not in ("0", "1", "0.0", "1.0"):
        df = df.iloc[1:].copy()
    df["noi_dung"] = df["noi_dung"].fillna("").astype(str)
    df["nhan"] = pd.to_numeric(df["nhan"], errors="raise").astype(int)
    if set(df["nhan"].unique()) != {0, 1}:
        raise ValueError("Cot nhan phai chua ca hai gia tri 0 (thuong) va 1 (rac).")
    return df.reset_index(drop=True)

import math
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Viet bo dem va bo phan loai tu dau, KHONG dung MultinomialNB cho bo loc tu viet.
# Regex nay trung voi quy tac tach tu mac dinh cua CountVectorizer.
QUY_TAC_TU = re.compile(r"(?u)\b\w\w+\b")

def tach_tu(van_ban):
    return QUY_TAC_TU.findall(str(van_ban).lower())

class NaiveBayesTuViet:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, van_ban, nhan):
        self.dem_lop = Counter(int(y) for y in nhan)
        self.dem_tu = {0: Counter(), 1: Counter()}
        for text, lop in zip(van_ban, nhan):
            self.dem_tu[int(lop)].update(tach_tu(text))
        self.tu_vung = set(self.dem_tu[0]) | set(self.dem_tu[1])
        self.tong_tu = {lop: sum(self.dem_tu[lop].values()) for lop in (0, 1)}
        self.tong_mau = len(nhan)
        return self

    def predict_one(self, van_ban):
        dem_van_ban = Counter(t for t in tach_tu(van_ban) if t in self.tu_vung)
        diem = {}
        for lop in (0, 1):
            diem[lop] = math.log(self.dem_lop[lop] / self.tong_mau)
            mau_so = self.tong_tu[lop] + self.alpha * len(self.tu_vung)
            for tu, so_lan in dem_van_ban.items():
                xs = (self.dem_tu[lop][tu] + self.alpha) / mau_so
                diem[lop] += so_lan * math.log(xs)
        return 1 if diem[1] > diem[0] else 0

    def predict(self, van_ban):
        return [self.predict_one(text) for text in van_ban]

df = doc_tin_nhan()
X_train, X_test, y_train, y_test = train_test_split(
    df["noi_dung"], df["nhan"], test_size=0.25,
    random_state=17, stratify=df["nhan"]
)

bo_loc = NaiveBayesTuViet(alpha=1.0).fit(X_train.tolist(), y_train.tolist())
pred_tu_viet = bo_loc.predict(X_test.tolist())

vec = CountVectorizer()
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)
model = MultinomialNB(alpha=1.0)
model.fit(Xtr, y_train)
pred_sklearn = model.predict(Xte)

acc_tu_viet = accuracy_score(y_test, pred_tu_viet)
acc_sklearn = accuracy_score(y_test, pred_sklearn)

print("BAI 4 - NAIVE BAYES TU VIET")
print(f"Tong tin nhan: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
print(f"So tu trong tu vung: {len(bo_loc.tu_vung)}")
print(f"Accuracy tu viet:       {acc_tu_viet:.4f}")
print(f"Accuracy MultinomialNB: {acc_sklearn:.4f}")
print(f"Chenh lech:             {abs(acc_tu_viet - acc_sklearn):.4f}")
print(f"So du doan khac nhau:   {sum(a != b for a, b in zip(pred_tu_viet, pred_sklearn))}")
print("NHAN XET: Hai mo hinh cung dem tu, lam tron Laplace alpha=1")
print("va cong log xac suat, nen ket qua phai rat gan nhau (thuong trung nhau).")
