# Adjoin/Adjugate 3x3 — Penjelasan & Cara Pakai

Skrip `minkofad.py` membantu Anda mempelajari dan menghitung komponen-komponen klasik aljabar linear untuk matriks 3x3 secara interaktif dan rinci:

- Minor (setiap elemen memakai determinan 2x2: ad - bc)
- Kofaktor (menambahkan pola tanda (-1)^(i+j))
- Adjoin/Adjugate (`adj(A) = C^T`)
- Determinan (ekspansi kofaktor pada baris pertama)
- Invers (`A^{-1} = (1/det(A)) * adj(A)`) jika determinan ≠ 0

File: `adjoin/minkofad.py`

## Fitur

- CLI interaktif: input 3 baris angka, dipisah spasi/koma.
- Output berurutan dan edukatif (menunjukkan submatriks 2x2, nilai minor, tanda kofaktor, transpose, hingga determinan dan invers).
- Tampilan matriks rapi dan rata kolom.
- Hasil invers ditampilkan dalam 2 bentuk:
  - Pecahan rasional (pakai `fractions.Fraction`) — tepat jika determinan bulat.
  - Desimal — nyaman dibaca, dengan presisi yang wajar.
- Utilitas pemformatan angka agar 1.999999 ≈ 2 dicetak sebagai `2`.

## Prasyarat

- Python 3.8+
- NumPy

Instal dependensi:

```bash
pip install numpy
```

## Menjalankan (CLI interaktif)

Jalankan skrip secara langsung:

```bash
python adjoin/minkofad.py
```

Contoh sesi dengan matriks 3x3:

```
Masukkan matriks 3x3 baris per baris (pisahkan angka dengan spasi/koma).
Baris 1: 1 2 3
Baris 2: 0 1 4
Baris 3: 5 6 0

Matriks A =
[ 1  2  3 ]
[ 0  1  4 ]
[ 5  6  0 ]

1) Minor M (pakai det 2x2: ad - bc)
...
Matriks Minor (M) =
[ -24  -20   -5 ]
[ -18  -15   -4 ]
[   5    4    1 ]

2) Kofaktor C (C[i,j] = (-1)^(i+j) * M[i,j])
...
Matriks Kofaktor (C) =
[ -24   20   -5 ]
[  18  -15    4 ]
[   5   -4    1 ]

3) Adjoin (Adjugate) adj(A) = C^T
...
Adjoin / Adjugate adj(A) =
[ -24   18    5 ]
[  20  -15   -4 ]
[  -5    4    1 ]

4) Determinan det(A) (ekspansi baris 1)
...
Determinannya det(A) = 1

5) Invers A^{-1} = (1/det(A)) * adj(A)
Bentuk pecahan (rasional):
[ -24   18    5 ]
[  20  -15   -4 ]
[  -5    4    1 ]

Bentuk desimal:
[ -24.00000000   18.00000000    5.00000000 ]
[  20.00000000  -15.00000000   -4.00000000 ]
[  -5.00000000    4.00000000    1.00000000 ]
```

> Catatan encoding: Jika terminal Anda menampilkan karakter aneh pada simbol/operator, itu biasanya karena encoding. Fungsi tetap berjalan dengan benar.

## Menggunakan sebagai Modul (API)

Anda bisa mengimpor fungsi-fungsinya bila ingin digunakan secara terprogram (tanpa CLI):

```python
import numpy as np
from adjoin.minkofad import (
    minor_matrix,
    cofactor_matrix,
    adjoint,
    determinant_from_cofactor_row1,
    pretty_matrix,
)

A = np.array([
    [1, 2, 3],
    [0, 1, 4],
    [5, 6, 0],
], dtype=float)

# 1) Minor dari A (khusus 3x3)
M = minor_matrix(A)

# 2) Kofaktor dari M (khusus 3x3)
C = cofactor_matrix(M)

# 3) Adjoin/Adjugate: transpose dari C
Adj = adjoint(C)

# 4) Determinan via ekspansi baris pertama
detA = determinant_from_cofactor_row1(A, C)

# 5) Invers bila det(A) ≠ 0
if abs(detA) > 1e-12:
    A_inv = Adj / detA
    print(pretty_matrix(A_inv, decimals=8))
```

Ringkasan fungsi:

- `minor_matrix(A: np.ndarray, verbose=False) -> np.ndarray`  
  Mengembalikan matriks minor 3x3 dari A. Menjelaskan langkah (submatriks 2x2 dan det 2x2) bila `verbose=True`.

- `cofactor_matrix(M: np.ndarray, verbose=False) -> np.ndarray`  
  Mengembalikan matriks kofaktor 3x3 dari M. Bila `verbose=True` menampilkan pola tanda dan perhitungan setiap elemen.

- `adjoint(C: np.ndarray, verbose=False) -> np.ndarray`  
  Mengembalikan adjoin/adjugate sebagai transpose dari matriks kofaktor `C`.

- `determinant_from_cofactor_row1(A: np.ndarray, C: np.ndarray, verbose=False) -> float`  
  Menghitung determinan dengan ekspansi kofaktor pada baris pertama: `a11*C11 + a12*C12 + a13*C13`.

- `pretty_matrix(M, decimals=4) -> str`  
  Menghasilkan string matriks yang rapi (kolom rata), cocok untuk dicetak.

- `to_fraction_matrix(M, denom_limit=999999) -> list[list[Fraction]]` dan `print_fraction_matrix(F)`  
  Utilitas untuk menampilkan matriks sebagai pecahan rasional yang rapi.

> Catatan: Skrip ini memang dioptimalkan untuk matriks 3x3. Fungsi `minor_matrix` dan `cofactor_matrix` mengasumsikan ukuran 3x3 (lihat inisialisasi bentuk `(3,3)` di kode). Untuk ukuran umum, perlu refaktor ringan ke versi generik `n x n`.

## Detail Perhitungan

- Minor per elemen dihitung dari submatriks 2x2 yang diambil dengan menghapus baris ke-`i` dan kolom ke-`j` dari A, lalu `det2x2 = a*d - b*c`.
- Kofaktor memakai pola tanda `(-1)^(i+j)` dikalikan nilai minor yang sesuai.
- Adjoin/Adjugate adalah transpose dari matriks kofaktor: `adj(A) = C^T`.
- Determinan dihitung dari ekspansi kofaktor pada baris pertama.
- Invers dihitung sebagai `A^{-1} = (1/det(A)) * adj(A)` bila `det(A) ≠ 0`.
- Untuk det(A) bilangan bulat, representasi pecahan invers akan tepat (bukan aproksimasi floating).

## Penanganan Input & Pemformatan

- Input baris berisi 3 angka (float diperbolehkan), dipisah spasi atau koma.
- Angka yang sangat dekat dengan bilangan bulat akan dicetak sebagai bilangan bulat agar rapi (misal 1.999999 → 2).
- Jumlah desimal pada tampilan dapat diatur lewat `decimals` pada `pretty_matrix`/`matrix_to_str`.

## Keterbatasan & Saran

- Khusus 3x3: fungsi-fungsi inti mengasumsikan ukuran 3x3. Untuk generalisasi ke `n x n`, perlu mengganti loop/turunan minor agar mengikuti ukuran dinamis.
- Encoding terminal tertentu bisa membuat simbol terlihat aneh; ini hanya visual.
- Angka sangat besar/kecil dapat memperlihatkan efek pembulatan floating; gunakan pecahan bila ingin presisi eksak.

## Lisensi

Belum ditentukan. Tambahkan lisensi sesuai kebutuhan proyek Anda.

