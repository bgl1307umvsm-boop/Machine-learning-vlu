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

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = doc_tin_nhan()
X_train, X_test, y_train, y_test = train_test_split(
    df["noi_dung"], df["nhan"], test_size=0.25,
    random_state=17, stratify=df["nhan"]
)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("BAI 3 - DOI THAM SO LAM TRON")
print(f"Tong tin nhan: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
print(f"{'alpha':>10} {'accuracy':>12} {'precision':>12} {'recall':>12}")
for alpha in [0.001, 0.1, 1.0, 10.0, 100.0]:
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train_vec, y_train)
    du_doan = model.predict(X_test_vec)
    acc = accuracy_score(y_test, du_doan)
    pre = precision_score(y_test, du_doan, pos_label=1, zero_division=0)
    rec = recall_score(y_test, du_doan, pos_label=1, zero_division=0)
    print(f"{alpha:>10.3f} {acc:>12.4f} {pre:>12.4f} {rec:>12.4f}")

print("\nNHAN XET: Alpha rat lon lam P(tu | lop) tien gan nhau giua cac tu trong cung lop;")
print("cac tu mat dan kha nang phan biet. Mo hinh co the du doan kem hon,")
print("nhung khong dam bao accuracy hay recall luon giam theo alpha.")
