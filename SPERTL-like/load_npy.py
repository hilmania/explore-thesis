import numpy as np

# Ganti 'nama_file_anda.npy' dengan path file Anda
nama_file = 'nama_file_anda.npy'

# Gunakan np.load() untuk memuat file
try:
    data = np.load(nama_file)

    # Tampilkan data yang sudah dimuat
    print("File berhasil dimuat!")
    print("Bentuk (shape) data:", data.shape)
    print("Tipe data:", data.dtype)
    print("\nIsi data (beberapa elemen pertama):")
    print(data)

except FileNotFoundError:
    print(f"Error: File '{nama_file}' tidak ditemukan.")
except Exception as e:
    print(f"Terjadi error saat memuat file: {e}")
