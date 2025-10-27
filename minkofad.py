#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from fractions import Fraction

# ========== Util Format ========== #
def _fmt_scalar(x, decimals=4, int_tol=1e-9):
    """Format angka agar ramah dibaca: integer tanpa .0, desimal dipangkas nol, dukung Fraction."""
    if isinstance(x, Fraction):
        return f"{x.numerator}/{x.denominator}" if x.denominator != 1 else f"{x.numerator}"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    # Floats
    try:
        xi = int(round(float(x)))
        if abs(float(x) - xi) < int_tol:
            return str(xi)
    except Exception:
        pass
    s = f"{float(x):.{decimals}f}"
    # pangkas nol ekor dan titik
    s = s.rstrip('0').rstrip('.') if '.' in s else s
    return s

def matrix_to_str(M, decimals=4, int_tol=1e-9):
    """Pretty print matriks menjadi baris yang rapi dan rata kolom."""
    M = np.asarray(M)
    rows_str = []
    col_w = [0] * M.shape[1]
    rough = []
    for i in range(M.shape[0]):
        r = []
        for j in range(M.shape[1]):
            s = _fmt_scalar(M[i, j], decimals=decimals, int_tol=int_tol)
            r.append(s)
            col_w[j] = max(col_w[j], len(s))
        rough.append(r)
    for r in rough:
        pads = [s.rjust(col_w[j]) for j, s in enumerate(r)]
        rows_str.append("[ " + "  ".join(pads) + " ]")
    return "\n".join(rows_str)

def fmt2x2(M, decimals=4):
    """Format matriks 2x2 sebagai string (rapi)."""
    return matrix_to_str(M, decimals=decimals)

def indent_block(s: str, prefix: str = "     ") -> str:
    """Tambahkan indent prefix ke setiap baris dalam string block."""
    return "\n".join(prefix + line for line in s.splitlines())

# ========== Operasi Matriks Dasar ========== #
def submatrix(A, i, j):
    """Hapus baris i dan kolom j (0-based)."""
    return np.delete(np.delete(A, i, axis=0), j, axis=1)

def minor_matrix(A, verbose=False):
    """Matriks minor untuk A 3x3 (tiap elemen adalah det submatriks 2x2)."""
    M = np.empty((3, 3), dtype=float)
    if verbose:
        print("\n1) Minor M (pakai det 2x2: ad - bc)")
        print("   Rumus: jika S = [[a, b],[c, d]] maka det(S) = ad - bc")
    for i in range(3):
        for j in range(3):
            S = submatrix(A, i, j)
            a, b, c, d = S[0, 0], S[0, 1], S[1, 0], S[1, 1]
            det2 = a * d - b * c
            M[i, j] = det2
            if verbose:
                print(f"\n   M[{i+1},{j+1}] -> hapus baris {i+1}, kolom {j+1}:")
                print("     Submatriks 2x2 (S) =")
                print(indent_block(fmt2x2(S), prefix="     "))
                print(
                    f"     det(S) = a·d - b·c = ({_fmt_scalar(a)})*({_fmt_scalar(d)}) - ({_fmt_scalar(b)})*({_fmt_scalar(c)}) "
                    f"= {_fmt_scalar(a*d)} - {_fmt_scalar(b*c)} = {_fmt_scalar(det2)}"
                )
    return M

def cofactor_matrix(M, verbose=False):
    """Matriks kofaktor: C[i,j] = (-1)^(i+j) * M[i,j]."""
    C = np.empty_like(M)
    if verbose:
        print("\n2) Kofaktor C (C[i,j] = (-1)^(i+j) · M[i,j])")
        # Tampilkan pola tanda dalam bentuk matriks s_ij = (-1)^(i+j)
        S = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]], dtype=int)
        print("   Pola tanda s_ij = (-1)^(i+j):")
        print(indent_block(matrix_to_str(S, decimals=0), prefix="   "))

    # Siapkan detail untuk perapihan kolom saat verbose
    details = []
    max_val_w = 0  # lebar nilai M dan C
    for i in range(3):
        for j in range(3):
            sign = -1 if ((i + j) % 2) else 1
            value_m = float(M[i, j])
            value_c = sign * value_m
            C[i, j] = value_c

            if verbose:
                m_str = _fmt_scalar(value_m)
                c_str = _fmt_scalar(value_c)
                s_str = "+1" if sign > 0 else "-1"
                max_val_w = max(max_val_w, len(m_str), len(c_str))
                details.append((i + 1, j + 1, s_str, m_str, c_str))

    if verbose:
        print("   Rincian (setiap elemen):")
        for (ii, jj, s_str, m_str, c_str) in details:
            print(
                "   "
                + f"C[{ii},{jj}] = ({s_str:>2}) · M[{ii},{jj}] = ({s_str:>2}) · {m_str.rjust(max_val_w)} = {c_str.rjust(max_val_w)}"
            )
    return C

def adjoint(C, verbose=False):
    """Adjoin/Adjugate = transpose dari kofaktor."""
    Adj = C.T
    if verbose:
        print("\n3) Adjoin (Adjugate) adj(A) = C^T")
        print("   Transpose kofaktor: baris <-> kolom")
    return Adj

def determinant_from_cofactor_row1(A, C, verbose=False):
    """Determinanan via ekspansi kofaktor pada baris-1: a11*C11 + a12*C12 + a13*C13."""
    detA = A[0, 0] * C[0, 0] + A[0, 1] * C[0, 1] + A[0, 2] * C[0, 2]
    if verbose:
        print("\n4) Determinan det(A) (ekspansi baris 1)")
        print("   det(A) = a11·C11 + a12·C12 + a13·C13")
        print(
            "   "
            + f"= {_fmt_scalar(A[0,0])}*{_fmt_scalar(C[0,0])} + {_fmt_scalar(A[0,1])}*{_fmt_scalar(C[0,1])} + {_fmt_scalar(A[0,2])}*{_fmt_scalar(C[0,2])} "
            + f"= {_fmt_scalar(detA)}"
        )
    return detA

def pretty_matrix(M, decimals=4):
    return matrix_to_str(M, decimals=decimals)

def to_fraction_matrix(M, denom_limit=999999):
    F = [[Fraction(x).limit_denominator(denom_limit) for x in row] for row in M]
    return F

def print_fraction_matrix(F):
    """Cetak matriks pecahan (list of list of Fraction) dengan kolom rata."""
    widths = [0, 0, 0]
    str_rows = []
    for row in F:
        srow = []
        for j, x in enumerate(row):
            s = f"{x.numerator}/{x.denominator}" if x.denominator != 1 else f"{x.numerator}"
            widths[j] = max(widths[j], len(s))
            srow.append(s)
        str_rows.append(srow)
    for row in str_rows:
        padded = [s.rjust(widths[j]) for j, s in enumerate(row)]
        print("[ " + "  ".join(padded) + " ]")

# ========== Bagian Invers: langkah rinci ========== #
def print_inverse_steps(A, Adj, detA, decimals=8, denom_limit=999999):
    """
    Cetak langkah rinci untuk invers:
    - A^{-1} = (1/det(A)) · adj(A)
    - Tampilkan faktor 1/det(A) (pecahan & desimal)
    - Tampilkan per-elemen: (A^{-1})_ij = adj_ij / det(A) (pecahan & desimal)
    - Cetak matriks invers bentuk pecahan dan desimal
    - Verifikasi A·A^{-1} dan A^{-1}·A (opsional)
    """
    print("\n5) Invers A^{-1} = (1/det(A)) · adj(A)")

    # 5.1 Faktor skalar 1/det(A)
    print("\n   5.1) Faktor skalar 1/det(A)")
    det_int = int(round(detA))
    if abs(detA - det_int) < 1e-12:
        # det(A) bulat → tampilkan pecahan eksak 1/det_int
        frac_factor = Fraction(1, det_int)
        print(f"        det(A) = {_fmt_scalar(detA)} → 1/det(A) = {frac_factor.numerator}/{frac_factor.denominator}")
    else:
        # det(A) tidak bulat → aproksimasi pecahan
        frac_factor = Fraction(detA).limit_denominator()
        frac_factor = Fraction(1, 1) / frac_factor
        print(f"        det(A) ≈ {_fmt_scalar(detA)} → 1/det(A) ≈ {frac_factor.numerator}/{frac_factor.denominator}")
    print(f"        (desimal) 1/det(A) ≈ {1.0/detA:.10f}")

    # 5.2 Tampilkan adj(A)
    print("\n   5.2) Matriks adjoin adj(A) yang dikalikan oleh faktor 1/det(A):")
    print(indent_block(pretty_matrix(Adj, decimals=4), prefix="        "))

    # 5.3 Hitung per-elemen sebagai pecahan & desimal
    print("\n   5.3) Per elemen: (A^{-1})_ij = adj_ij / det(A)")
    Ainv_frac = []
    Ainv_float = np.empty_like(Adj, dtype=float)
    for i in range(3):
        row_frac = []
        for j in range(3):
            aij = Adj[i, j]
            if abs(detA - det_int) < 1e-12:
                # det bulat → gunakan Fraction(adj_ij, det_int)
                fij = Fraction(int(round(aij)), det_int)
            else:
                # det tidak bulat → aproksimasi pecahan
                fij = Fraction(aij).limit_denominator(denom_limit) / Fraction(detA).limit_denominator(denom_limit)
            row_frac.append(fij)
            Ainv_float[i, j] = float(aij) / float(detA)

            # Cetak langkah untuk elemen (i+1,j+1)
            left = f"(A^-1)[{i+1},{j+1}]"
            right_frac = f"{fij.numerator}/{fij.denominator}" if fij.denominator != 1 else f"{fij.numerator}"
            print(
                "        "
                + f"{left} = adj[{i+1},{j+1}] / det(A) = "
                + f"{_fmt_scalar(aij)} / {_fmt_scalar(detA)}"
                + f" = {right_frac} ≈ {Ainv_float[i,j]:.{decimals}f}"
            )
        Ainv_frac.append(row_frac)

    # 5.4 Tampilkan matriks invers bentuk pecahan (rapi)
    print("\n   5.4) Matriks A^{-1} (bentuk pecahan):")
    print_fraction_matrix(Ainv_frac)

    # 5.5 Tampilkan matriks invers bentuk desimal
    print("\n   5.5) Matriks A^{-1} (bentuk desimal):")
    print(pretty_matrix(Ainv_float, decimals=decimals))

    # 5.6 Verifikasi A·A^{-1} dan A^{-1}·A (opsional)
    print("\n   5.6) Verifikasi (opsional): A · A^{-1} ≈ I, dan A^{-1} · A ≈ I")
    I_left = A @ Ainv_float
    I_right = Ainv_float @ A
    print(indent_block("A · A^{-1} ≈", prefix="        "))
    print(indent_block(pretty_matrix(I_left, decimals=6), prefix="        "))
    print(indent_block("A^{-1} · A ≈", prefix="        "))
    print(indent_block(pretty_matrix(I_right, decimals=6), prefix="        "))

# ========== Main ========== #
def main():
    # ===== INPUT =====
    print("Masukkan matriks 3x3 baris per baris (pisahkan angka dengan spasi/koma).")
    rows = []
    for r in range(3):
        while True:
            try:
                line = input(f"Baris {r+1}: ").replace(",", " ")
                nums = [float(x) for x in line.split()]
                if len(nums) != 3:
                    raise ValueError("Harus tepat 3 angka per baris.")
                rows.append(nums)
                break
            except Exception as e:
                print("Input tidak valid:", e, "- Coba lagi.")
    A = np.array(rows, dtype=float)

    # Tampilkan matriks A dengan format rapi
    print("\nMatriks A =")
    print(pretty_matrix(A, decimals=4))

    # ===== Minor =====
    M = minor_matrix(A, verbose=True)
    print("\nMatriks Minor (M) =")
    print(pretty_matrix(M, decimals=4))

    # ===== Cofactor =====
    C = cofactor_matrix(M, verbose=True)
    print("\nMatriks Kofaktor (C) =")
    print(pretty_matrix(C, decimals=4))

    # ===== Adjoint =====
    Adj = adjoint(C, verbose=True)
    print("\nAdjoin / Adjugate adj(A) =")
    print(pretty_matrix(Adj, decimals=4))

    # ===== Determinant =====
    detA = determinant_from_cofactor_row1(A, C, verbose=True)
    print(f"\nDeterminannya det(A) = {_fmt_scalar(detA)}")

    # ===== Inverse (lebih rinci) =====
    if abs(detA) > 1e-12:
        print_inverse_steps(A, Adj, detA, decimals=8, denom_limit=999999)
    else:
        print("\nMatriks singular (det(A) = 0), invers tidak ada.")

if __name__ == "__main__":
    main()
