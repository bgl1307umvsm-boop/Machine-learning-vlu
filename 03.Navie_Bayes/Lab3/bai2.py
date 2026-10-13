
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

print("BAI 2: TU NAO TO CAO MANH NHAT")
print("------------------------------------")

thu_muc = Path(__file__).resolve().parent

cac_duong_dan = [
    thu_muc / "tin_nhan.csv",
    thu_muc.parent / "tin_nhan.csv",
    thu_muc / "data" / "tin_nhan.csv",
    thu_muc.parent / "data" / "tin_nhan.csv"
]

duong_dan = next(
    (p for p in cac_duong_dan if p.exists()),
    None
)

if duong_dan is None:
    raise FileNotFoundError("Khong tim thay tin_nhan.csv")

df = pd.read_csv(
    duong_dan,
    header=None,
    encoding="utf-8-sig"
)

if df.shape[1] != 2:
    raise ValueError("File CSV phai co 2 cot: noi dung va nhan")

df.columns = ["noi_dung", "nhan"]

nhan_dau = str(df.iloc[0]["nhan"]).strip().lower()

if nhan_dau not in ["0", "1", "0.0", "1.0", "spam", "ham", "rac", "thuong", "rác", "thường"]:
    df = df.iloc[1:].copy()

df["nhan"] = df["nhan"].astype(str).str.strip().str.lower()

df["nhan"] = df["nhan"].replace({
    "spam": "1",
    "rac": "1",
    "rác": "1",
    "ham": "0",
    "thuong": "0",
    "thường": "0",
    "0.0": "0",
    "1.0": "1"
})

df["nhan"] = df["nhan"].astype(int)

X_text = df["noi_dung"].fillna("").astype(str)
y = df["nhan"]

print("Tong so tin nhan:", len(df))
print("Tin nhan thuong:", (y == 0).sum())
print("Tin nhan rac:", (y == 1).sum())

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_text)

mo_hinh = MultinomialNB()
mo_hinh.fit(X, y)

tu_vung = vectorizer.get_feature_names_out()

idx_0 = list(mo_hinh.classes_).index(0)
idx_1 = list(mo_hinh.classes_).index(1)

hieu = (
    mo_hinh.feature_log_prob_[idx_1]
    - mo_hinh.feature_log_prob_[idx_0]
)

thu_tu = hieu.argsort()

top_rac = thu_tu[-5:][::-1]

top_thuong = thu_tu[:5]

print("\n5 TU TO CAO TIN NHAN RAC MANH NHAT:")
for i in top_rac:
    print(f"{tu_vung[i]}: {hieu[i]:.4f}")

print("\n5 TU CHO THAY TIN NHAN THUONG MANH NHAT:")
for i in top_thuong:
    print(f"{tu_vung[i]}: {hieu[i]:.4f}")

print("\n------------------------------------")
print("HOAN THANH BAI 2!")
