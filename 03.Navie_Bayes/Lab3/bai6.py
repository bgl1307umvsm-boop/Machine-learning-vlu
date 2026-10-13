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
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

df = doc_tin_nhan()
thuong = df[df["nhan"] == 0].copy()
rac = df[df["nhan"] == 1].copy()
if len(thuong) != 180 or len(rac) < 20:
    raise ValueError(f"De yeu cau 180 tin thuong va it nhat 20 tin rac; "
                     f"file hien co {len(thuong)} thuong va {len(rac)} rac.")

rac_20 = rac.sample(n=20, random_state=17)
bo_moi = pd.concat([thuong, rac_20], ignore_index=True)
X_train, X_test, y_train, y_test = train_test_split(
    bo_moi["noi_dung"], bo_moi["nhan"], test_size=0.25,
    random_state=17, stratify=bo_moi["nhan"]
)
vec = CountVectorizer()
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)
model = MultinomialNB(alpha=1.0)
model.fit(Xtr, y_train)
pred = model.predict(Xte)
acc = accuracy_score(y_test, pred)
rec = recall_score(y_test, pred, pos_label=1, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
print("BAI 6 - LOP HIEM VA XAC SUAT TIEN NGHIEM")
print(f"Tap moi: {len(bo_moi)} tin (thuong={len(thuong)}, rac=20)")
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Accuracy:       {acc:.4f}")
print(f"Recall lop rac: {rec:.4f}")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Accuracy neu luon du doan thuong: {(y_test == 0).mean():.4f}")
print("NHAN XET: Do lop thuong chiem 90%, accuracy co the cao")
print("ngay ca khi bo loc bo sot nhieu tin rac. Recall lop 1")
print("= TP / (TP + FN) moi do truc tiep ty le tin rac duoc phat hien.")
