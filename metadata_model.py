from rfdetr import RFDETRSmall

# Load model dari checkpoint (ganti dengan path ke file ema.pth)
model = RFDETRSmall(pretrain_weights="yourweight.pth")  # Ganti dengan path model .pth

# Menampilkan atributs dari model yang berhubungan dengan kelas atau label
print("Atribut model yang berkaitan dengan class/label:")
for attr in dir(model):
    if 'class' in attr.lower() or 'label' in attr.lower():
        print(f"{attr}: {getattr(model, attr, 'Tidak ditemukan')}")

# Mengecek jumlah kelas jika atribut 'num_classes' tersedia
if hasattr(model, 'num_classes'):
    print(f"Jumlah kelas pada model: {model.num_classes}")