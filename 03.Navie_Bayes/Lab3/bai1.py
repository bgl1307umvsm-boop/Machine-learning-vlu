nhom = ["Hoc deu", "Hoc don", "It hoc"]
p_nhom = [0.50, 0.30, 0.20]
p_qua = [0.95, 0.70, 0.30]

tong = sum(p_nhom[i] * p_qua[i] for i in range(3))

print("BAI 1: BAYES VOI BA NGUYEN NHAN")

for i in range(3):
    ket_qua = p_nhom[i] * p_qua[i] / tong
    print(f"{nhom[i]}: {ket_qua:.2%}")

print(f"Tong xac suat: {sum(p_nhom[i] * p_qua[i] / tong for i in range(3)):.2%}")
