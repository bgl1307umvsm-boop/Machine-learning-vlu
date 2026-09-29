import pandas as pd
from pathlib import Path

thu_muc_hien_tai = Path(__file__).resolve().parent

file_csv = thu_muc_hien_tai.parent / "data" / "sinh_vien.csv"

df = pd.read_csv(file_csv)

nhom_7 = df[df["diem_giua_ky"] >= 7]
nhom_con_lai = df[df["diem_giua_ky"] < 7]

print("Số bạn có điểm giữa kỳ từ 7 trở lên:", len(nhom_7))
print("Tỷ lệ qua môn nhóm >= 7:", nhom_7["qua_mon"].mean())
print("Tỷ lệ qua môn nhóm < 7:", nhom_con_lai["qua_mon"].mean())

if nhom_7["qua_mon"].mean() > nhom_con_lai["qua_mon"].mean():
    print("Nhận xét: Điểm giữa kỳ có khả năng phân biệt hai nhóm.")
else:
    print("Nhận xét: Điểm giữa kỳ chưa phân biệt rõ hai nhóm.")