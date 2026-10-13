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
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score

# File sinh_vien.csv phai co tieu de cot.
df = pd.read_csv(tim_file("sinh_vien.csv"), encoding="utf-8-sig")
df.columns = df.columns.str.strip()

hai_cot = ["gio_on", "diem_giua_ky"]
thieu = [c for c in hai_cot if c not in df.columns]
if thieu:
    raise ValueError(f"Thieu cot {thieu}. Cot dang co: {df.columns.tolist()}")

# Tu nhan dien cot nhan, khong tu y lay cot cuoi vi de gay sai ket qua.
cot_nhan_co_the = ["qua_mon", "ket_qua", "nhan", "label", "dau_mon", "dat", "target", "passed"]
cot_nhan = next((c for c in cot_nhan_co_the if c in df.columns), None)
if cot_nhan is None:
    raise ValueError(f"Chua biet cot nhan sinh vien. Cac cot: {df.columns.tolist()}. "
                     "Hay sua bien cot_nhan trong code theo file cua ban.")

X = df[hai_cot].apply(pd.to_numeric, errors="raise")
y = df[cot_nhan]
if X.isnull().any().any() or y.isnull().any():
    raise ValueError("Du lieu co gia tri trong. Can xu ly truoc khi huan luyen.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
print("BAI 5 - CHON SAI BIEN THE NAIVE BAYES")
print(f"So mau: {len(df)} | Cot nhan: {cot_nhan}")

try:
    multi = MultinomialNB()
    multi.fit(X_train, y_train)
    acc_multi = accuracy_score(y_test, multi.predict(X_test))
    print(f"MultinomialNB: {acc_multi:.4f} (chay duoc)")
except ValueError as e:
    print("MultinomialNB bao loi:", e)

gauss = GaussianNB()
gauss.fit(X_train, y_train)
acc_gauss = accuracy_score(y_test, gauss.predict(X_test))
print(f"GaussianNB:    {acc_gauss:.4f}")
print("NHAN XET: MultinomialNB mo hinh hoa so lan dem/tu suat khong am;")
print("gio_on va diem_giua_ky la bien so do luong, khong phai so lan dem.")
print("Vi vay, MultinomialNB sai ve gia dinh du no van co the chay va co accuracy.")
print("GaussianNB dung gia dinh phan phoi Gauss cho tung dac trung lien tuc.")
