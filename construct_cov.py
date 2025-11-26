# build_covariances.py
# -*- coding: utf-8 -*-
# Darwin Yao
"""
从 500×63 的月度日收益矩阵构造三类协方差估计：
1) Ledoit–Wolf 收缩到缩放单位阵（LW）
2) PCA 因子协方差（前 k 个主成分 + 残差对角）
3) James–Stein 对特征向量收缩（JS-eigvec）+ 因子协方差

输入：
- in_dir 目录下若干 yyyymm_full.csv（500 行 × 63 列，行=股票，列=交易日）
- result_500.txt：每行 "yyyymm_full x y"，使用 x 作为该月因子数 k

输出：
- out_root/LW|PCA|JS_eigvec/yyyymm_full_cov.csv
- out_root/meta/yyyymm_full.json（元数据）
- out_root/logs/summary.csv（汇总日志）

用法：
python build_covariances.py --in_dir ./500_ret_new --result_txt ./result_500.txt --out_root ./covariance_outputs
"""
import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from honeyShrinkage import covCor


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_factor_counts(result_txt: Path) -> Dict[str, int]:
    """
    解析 result_500.txt ："yyyymm_full x y" -> k=x
    允许行尾有额外字段，只取第一个 token 为 key，第二个为 k
    """
    mapping: Dict[str, int] = {}
    with open(result_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            key = parts[0]
            if len(parts) < 2:
                continue
            try:
                k = int(parts[1])
            except Exception:
                continue
            mapping[key] = k
    return mapping


def demean_over_time(X: np.ndarray) -> np.ndarray:
    """按行去均值（每只股票减 63 天均值）"""
    return X - X.mean(axis=1, keepdims=True)


def sample_cov(X: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    X: (p, n) 去均值后的收益矩阵（行=资产，列=时间）
    返回：S = (1/n) X X^T（而非 n-1），n
    """
    p, n = X.shape
    if n <= 1:
        raise ValueError("需要至少 2 个样本期。")
    S = (X @ X.T) / n
    S = (S + S.T) / 2.0
    return S, n


def ledoit_wolf(S: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Ledoit–Wolf（2004）收缩到 F = μ I_p：
        Σ_LW = (1-δ) S + δ F,  μ = tr(S)/p
    其中
        φ̂ = (1/n²) Σ_t || x_t x_t^T - S ||_F^2  [CORRECTED: divide by n², not n]
        γ̂ = || S - F ||_F^2
        δ = clip(φ̂/γ̂, 0, 1)
    
    Note: The original Ledoit-Wolf formula requires dividing by n² to properly 
    estimate the variance of the sample covariance matrix S (not the variance of 
    individual outer products x_t x_t^T).
    """
    p = S.shape[0]
    n = X.shape[1]
    mu = float(np.trace(S)) / p
    F = mu * np.eye(p)

    # 计算 φ̂ (CORRECTED: divide by n² instead of n)
    phi_acc = 0.0
    for t in range(n):
        xt = X[:, t:t + 1]            # (p,1)
        outer = xt @ xt.T             # (p,p)
        diff = outer - S
        phi_acc += float(np.sum(diff * diff))
    phi_hat = phi_acc / (n * n)  # FIXED: was phi_acc / n, now phi_acc / n²

    # 计算 γ̂
    gamma_hat = float(np.sum((S - F) ** 2))
    if gamma_hat <= 0:
        delta = 1.0
    else:
        delta = max(0.0, min(1.0, phi_hat / gamma_hat))

    Sigma = (1.0 - delta) * S + delta * F
    Sigma = (Sigma + Sigma.T) / 2.0
    return Sigma, float(delta)


def top_k_eigenpairs(S: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """对称矩阵特征分解，返回前 k 个（按特征值降序）"""
    vals, vecs = np.linalg.eigh(S)     # 升序
    idx = np.argsort(vals)[::-1]       # 转为降序
    vals = vals[idx]
    vecs = vecs[:, idx]
    k = min(k, vecs.shape[1])
    return vecs[:, :k], vals[:k]


def pca_factor_cov(S: np.ndarray, k: int, eps: float = 1e-12) -> np.ndarray:
    """
    PCA 因子协方差：
        S ≈ U_k Λ_k U_k^T + Ψ，
        其中 Ψ = diag(diag(S - U_k Λ_k U_k^T)) 截断到 eps 为正
    """
    p = S.shape[0]
    k_eff = max(0, min(k, p - 1))
    if k_eff == 0:
        diag = np.maximum(np.diag(S), eps)
        return np.diag(diag)

    U, lam = top_k_eigenpairs(S, k_eff)      # U: (p,k), lam: (k,)
    Sig_k = U @ (lam[:, None] * U.T)
    diag_resid = np.diag(S - Sig_k)
    psi = np.maximum(diag_resid, eps)
    Sigma = Sig_k + np.diag(psi)
    Sigma = (Sigma + Sigma.T) / 2.0
    return Sigma


def ortho(M: np.ndarray) -> np.ndarray:
    """
    Orthonormalize columns of M using eigendecomposition.
    Returns orthonormal basis with same column space as M.
    """
    vals, vecs = np.linalg.eigh(M.T @ M)
    idx = np.argsort(vals)[::-1]  # descending order
    vals = vals[idx]
    vecs = vecs[:, idx]
    Q = M @ vecs / np.sqrt(vals)
    return Q


def js_eigvec_factor_cov(S: np.ndarray, k: int, n: int, eps: float = 1e-12) -> np.ndarray:
    """
    JS-eigvec 因子协方差：
      1) Top k eigenpairs of S: (U_k, Λ_k)
      2) Apply James–Stein shrinkage to each column of U_k:
         h^JSE = m(h)·1 + c^JSE (h - m(h)·1),
         c^JSE = 1 - ν^2 / s^2(h),
         s^2(h) = (λ^2/p) * Σ (h_i - m(h))^2,
         ν^2 = (tr(S) - λ^2) / (p * (n - 1))
      3) Σ_JS = \tilde U_k Λ_k \tilde U_k^T + diag(diag(S - \tilde U_k Λ_k \tilde U_k^T))
    """
    p = S.shape[0]
    k_eff = max(0, min(k, p - 1))
    if k_eff == 0:
        diag = np.maximum(np.diag(S), eps)
        return np.diag(diag)

    U, lam = top_k_eigenpairs(S, k_eff)

    # Column-wise JSE shrinkage (following the formula strictly)
    U_js = U.copy()
    trS = float(np.trace(S))
    one = np.ones((p, 1))
    for j in range(k_eff):
        h = U[:, j:j+1]                 # j-th column
        lamj = float(lam[j])            # corresponding eigenvalue λ_j
        m = float(h.mean())             # m(h)
        h_c = h - m * one
        # s2 = (lamj ** 2) * float(np.sum(h_c ** 2)) / p
        # v2 = (trS - (lamj ** 2)) / (p * (n - 1))
        s2 = lamj * float(np.sum(h_c ** 2)) / p
        # v2 represents the average variance of remaining eigenvalues
        # After extracting factors 0 to j, the remaining variance is from eigenvalues j+1 onwards
        sum_lam_up_to_j = float(np.sum(lam[:j+1]))  # sum of λ_0 + λ_1 + ... + λ_j
        v2 = (trS - sum_lam_up_to_j) / (p * (n - (j + 1)))
        c = 1.0 - (v2 / (s2 + 1e-18))   # add small value for numerical stability
        U_js[:, j:j+1] = m * one + c * h_c

    # QR orthogonalization + align sign with original U column vectors
    Q, _ = np.linalg.qr(U_js)
    for j in range(k_eff):
        if float(np.dot(Q[:, j], U[:, j])) < 0:
            Q[:, j] *= -1.0

    Sig_k = Q @ (lam[:, None] * Q.T)
    diag_resid = np.diag(S - Sig_k)
    psi = np.maximum(diag_resid, eps)
    Sigma = Sig_k + np.diag(psi)
    Sigma = (Sigma + Sigma.T) / 2.0
    return Sigma


def jsm_factor_cov(S: np.ndarray, X: np.ndarray, k: int, n: int, eps: float = 1e-12) -> np.ndarray:
    """
    JSM (James-Stein Multiple factors) factor covariance for multiple factors.
    
    This method performs joint shrinkage of multiple eigenvectors toward a target space
    spanned by the constant vector and the mean return vector, following the approach
    in H_jsm.py.
    
    Steps:
      1) Compute top k eigenpairs of S: (U_k, Λ_k)
      2) Estimate residual variance from bulk eigenvalues
      3) Compute shrinkage factors Ψ² for each eigenvalue
      4) Create target matrix from [constant vector, mean vector]
      5) Apply weighted JSM shrinkage:
         - Weight data by inverse sqrt of residual variances
         - Compute eigenvectors of weighted covariance
         - Shrink toward orthonormalized target space
      6) Construct covariance: Σ = H_jsm Λ_k H_jsm^T + Ψ
    
    Args:
        S: Sample covariance matrix (p × p)
        X: Demeaned return matrix (p × n), rows=assets, columns=time
        k: Number of factors
        n: Number of observations
        eps: Minimum idiosyncratic variance
        
    Returns:
        Covariance matrix with JSM-shrunk factors
    """
    p = S.shape[0]
    k_eff = max(0, min(k, p - 1))
    if k_eff == 0:
        diag = np.maximum(np.diag(S), eps)
        return np.diag(diag)

    # Get top k eigenpairs
    U, lam = top_k_eigenpairs(S, k_eff)
    
    # Compute mean returns (time average for each asset)
    m = X.mean(axis=1)  # (p,)
    
    # Marchenko-Pastur correction parameter
    npc = n / p
    
    # Estimate residual variance from bulk eigenvalues
    trS = float(np.trace(S))
    sum_top_lam = float(np.sum(lam))
    
    # Adjusted formula for bulk variance (residual/idiosyncratic variance)
    gam2 = (trS - sum_top_lam) / (n - k_eff - npc * k_eff)
    gam2 = max(gam2, eps)  # ensure positive
    
    # Shrinkage factors for eigenvalues (Ψ²)
    Psi2 = (lam - gam2) / lam
    Psi2 = np.maximum(Psi2, 0.0)  # ensure non-negative
    
    # Compute residual variances (Delta)
    eigs = lam  # eigenvalues
    Delta = np.sum((X / np.sqrt(n) - (U * np.sqrt(eigs)) @ (U.T @ X / np.sqrt(n)))**2, axis=1)
    Delta = Delta * gam2 / np.mean(Delta)  # enforce relation gam2 = mean(Delta)
    Delta = np.maximum(Delta, eps)  # ensure positive
    
    # Apply James-Stein shrinkage to mean vector
    e = np.ones(p)
    gm = (m @ e) * e / p  # grand mean vector
    m2 = float(np.sum((m - gm)**2))
    c_m = 1.0 - (p * gam2 / n) / (m2 + 1e-18)
    mjs = c_m * m + (1.0 - c_m) * gm  # JS-shrunk mean
    
    # === Weighted JSM calculations ===
    # Weight by inverse sqrt of residual variances
    D = 1.0 / np.sqrt(Delta)  # (p,)
    
    # Weighted data matrix
    Y = X.copy()  # (p, n)
    YD = (Y.T * D).T  # weight each row by D
    
    # Compute covariance of weighted data
    LD = YD.T @ YD / p  # (n, n) dual covariance
    
    # Get top k eigenpairs of weighted covariance
    vals_D, vecs_D = np.linalg.eigh(LD)
    idx = np.argsort(vals_D)[::-1]  # descending order
    vals_D = vals_D[idx][:k_eff]
    vecs_D = vecs_D[:, idx][:, :k_eff]
    
    # Convert to asset-space eigenvectors
    sHD = YD @ vecs_D / np.sqrt(p * vals_D)  # (p, k)
    eigs_D = vals_D * p / n
    
    # Compute shrinkage factors for weighted eigenvalues
    sum_vals_D = float(np.sum(vals_D))
    trace_LD = float(np.trace(LD))
    gam2D = (trace_LD - sum_vals_D) / (n - k_eff - npc)
    gam2D = max(gam2D, eps)
    Psi2D = (vals_D - gam2D) / vals_D
    Psi2D = np.maximum(Psi2D, 0.0)
    
    # Create weighted target matrix: [e*D, mjs*D]
    AD = np.vstack((e * D, mjs * D)).T  # (p, 2)
    sMD = ortho(AD)  # orthonormalize
    
    # Compute weighted factor matrix
    HD = sHD * np.sqrt(eigs_D)  # (p, k)
    
    # Project HD onto target space
    MD = (AD @ np.linalg.inv(AD.T @ AD)) @ (AD.T @ HD)  # (p, k)
    
    # Compute shrinkage matrix
    ND = (HD - MD).T @ (HD - MD)  # (k, k)
    CD = np.eye(k_eff) - np.linalg.inv(ND + eps * np.eye(k_eff)) * (gam2D * p / n)
    
    # Apply shrinkage
    sHD_jsm = (HD @ CD + MD @ (np.eye(k_eff) - CD)) / np.sqrt(eigs_D)
    
    # Unweight back to original space
    sH_jsm = ortho((sHD_jsm.T / D).T)  # (p, k)
    
    # Estimate factor variances (use original eigenvalues with shrinkage)
    fvar_est = Psi2 * lam
    
    # Construct covariance matrix
    Sig_k = sH_jsm @ (fvar_est[:, None] * sH_jsm.T)
    
    # Residual diagonal
    diag_resid = np.diag(S - Sig_k)
    psi = np.maximum(diag_resid, eps)
    
    Sigma = Sig_k + np.diag(psi)
    Sigma = (Sigma + Sigma.T) / 2.0
    
    return Sigma

def process_folder(in_dir: Path, result_txt: Path, out_root: Path, eps: float = 1e-12, num_factors: int = 0) -> None:
    """主流程：读取每个 yyyymm_full.csv，输出三类协方差与元数据
    
    Args:
        in_dir: Input directory
        result_txt: Factor count file (optional, not used if num_factors > 0)
        out_root: Output root directory
        eps: Minimum idiosyncratic variance
        num_factors: Fixed number of factors (if > 0, use this value; otherwise read from result_txt)
    """
    ensure_dir(out_root)
    out_js = out_root / "JSE"
    out_lw = out_root / "LW"
    out_pca = out_root / "PCA"
    out_meta = out_root / "meta"
    out_logs = out_root / "logs"
    out_sample = out_root / "raw"
    for d in (out_js, out_lw, out_pca, out_meta, out_logs):
        ensure_dir(d)

    # If num_factors is specified (> 0), use it; otherwise read from file
    use_fixed_k = (num_factors > 0)
    if use_fixed_k:
        print(f"Using fixed number of factors: k = {num_factors}")
        factor_counts = None
    else:
        factor_counts = read_factor_counts(result_txt)
        print(f"Reading factor counts from {result_txt.name}")

    files = sorted([p for p in in_dir.glob("*.csv") if p.name.endswith("_full.csv") and not p.name.startswith("._")])
    if not files:
        print(f"[WARN] 未在 {in_dir} 找到 *_full.csv 文件。")
        return

    log_rows = []
    def save_with_permno(mat: np.ndarray, out_path: Path, permno_vec: np.ndarray) -> None:
        df_out = pd.DataFrame(mat)
        # 在第 0 列插入 permno；保持 header=False 以与原脚本行为一致
        df_out.insert(0, "permno", permno_vec)
        df_out.to_csv(out_path, header=False, index=False)

    for fpath in files:
        key = fpath.stem  # yyyymm_full
        print(f"[Processing] {fpath.name}...")
        
        # Determine k: use fixed num_factors or read from file
        if use_fixed_k:
            k = num_factors
        else:
            if key not in factor_counts:
                print(f"[SKIP] {key}: 在 {result_txt.name} 中未找到因子数，跳过。")
                continue
            k = int(factor_counts[key])

        # ✨改动：读取时区分 permno 与收益矩阵
        try:
            df_all = pd.read_csv(fpath, header=None)
        except UnicodeDecodeError:
            # Try with latin-1 encoding if UTF-8 fails
            df_all = pd.read_csv(fpath, header=None, encoding='latin-1')
        permno_col = df_all.iloc[:, 0].values  # (原始行数,)
        ret_df = df_all.iloc[:, 1:]            # 只把第2列开始当作收益

        # ✨改动：去缺失只看收益列；同步过滤 permno
        X_raw = ret_df.values.astype(float)    # 预期 (500, 63)
        mask = ~np.any(np.isnan(X_raw), axis=1)
        kept_idx = np.where(mask)[0]
        kept_permno = permno_col[mask]
        X = X_raw[mask, :]
        p, n = X.shape

        # 去均值
        X = demean_over_time(X)

        # 样本协方差
        S, nobs = sample_cov(X)

        # 有效因子数：不能超过 p-1 和 n-1
        # k_eff = max(0, min(k, p - 1, nobs - 1))

        # # Temporarily setting factor number as 1
        # k_eff = 1

        # 三种估计
        Sigma_LW, delta = ledoit_wolf(S, X)
        Sigma_PCA = pca_factor_cov(S, k_eff, eps=eps)
        Sigma_JS = js_eigvec_factor_cov(S, k_eff, nobs, eps=eps)

        # ✨改动：保存时在最前面加上一列 permno
        save_with_permno(Sigma_LW,    out_lw     / f"{key}_cov.csv", kept_permno)
        save_with_permno(Sigma_PCA,   out_pca    / f"{key}_cov.csv", kept_permno)
        save_with_permno(Sigma_JS,    out_js     / f"{key}_cov.csv", kept_permno)

        # 元数据
        meta = {
            "file": fpath.name,
            "p_after_drop": int(p),
            "n_obs": int(nobs),
            "k_requested": int(k),
            "k_used": int(k_eff),
            "num_dropped_rows": int(X_raw.shape[0] - p),
            "kept_row_indices": kept_idx.tolist(),
            "lw_delta": float(delta),
        }
        with open(out_meta / f"{key}.json", "w", encoding="utf-8") as jf:
            json.dump(meta, jf, ensure_ascii=False, indent=2)

        # 日志
        log_rows.append(
            dict(
                month=key,
                p_after_drop=p,
                n_obs=nobs,
                k_req=k,
                k_used=k_eff,
                dropped=int(X_raw.shape[0] - p),
                lw_delta=float(delta),
            )
        )
        print(f"[OK] {key}: p={p}, n={nobs}, k={k_eff} -> 已保存三类协方差 (LW, PCA, JSE)。")

    if log_rows:
        df_log = pd.DataFrame(log_rows)
        df_log.to_csv(out_logs / "summary.csv", index=False)
        print(f"完成：共处理 {len(log_rows)} 个月。输出根目录：{out_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构造 LW / PCA / JS-eigvec 协方差矩阵")
    parser.add_argument("--in_dir", type=str, default="500_ret_emp",
                        help="月度 CSV 输入目录（默认 ./500_ret_emp）")
    parser.add_argument("--result_txt", type=str, default="result_500.txt",
                        help="含因子数的文本文件（默认 ./result_500.txt，如果指定 --num_factors 则忽略）")
    parser.add_argument("--out_root", type=str, default="covariance_outputs",
                        help="输出根目录（默认 ./covariance_outputs）")
    parser.add_argument("--eps", type=float, default=1e-12,
                        help="特质方差的最小截断（默认 1e-12）")
    parser.add_argument("--num_factors", type=int, default=0,
                        help="固定因子数（默认 0，从 result_txt 读取；如果 > 0，则使用此值）")
    return parser.parse_args()

# c:/Users/remote/Desktop/temp/tmp1030/code by Darwin/

if __name__ == "__main__":
    args = parse_args()
    in_dir = Path(args.in_dir).expanduser().resolve()
    result_txt = Path(args.result_txt).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    print("输入目录:   ", in_dir)
    print("因子数文件: ", result_txt)
    print("输出根目录: ", out_root)
    print("固定因子数: ", args.num_factors if args.num_factors > 0 else "从文件读取")

    if not in_dir.exists():
        raise FileNotFoundError(f"未找到输入目录：{in_dir}")
    
    # Only require result_txt if num_factors is not specified
    if args.num_factors <= 0 and not result_txt.exists():
        raise FileNotFoundError(f"未找到因子数文件：{result_txt}")

    ensure_dir(out_root)
    process_folder(in_dir, result_txt, out_root, eps=float(args.eps), num_factors=args.num_factors)
