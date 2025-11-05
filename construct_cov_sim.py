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

def js_eigvec_factor_cov(S: np.ndarray, k: int, n: int, eps: float = 1e-12) -> np.ndarray:
    """
    JS-eigvec 因子协方差：
      1) S 的前 k 个特征对 (U_k, Λ_k)
      2) 对 U_k 前 k 列按截图公式做 James–Stein 收缩（逐列）
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

    # ——逐列 JSE 收缩（严格按照公式）——
    U_js = U.copy()
    trS = float(np.trace(S))
    one = np.ones((p, 1))
    for j in range(k_eff):
        h = U[:, j:j+1]                 # 第 j 列
        lamj = float(lam[j])            # 对应特征值 λ_j
        m = float(h.mean())             # m(h)
        h_c = h - m * one
        # s2 = (lamj ** 2) * float(np.sum(h_c ** 2)) / p
        # v2 = (trS - (lamj ** 2)) / (p * (n - 1))
        s2 = lamj * float(np.sum(h_c ** 2)) / p
        v2 = (trS - lamj) / (p * (n - 1))
        c = 1.0 - (v2 / (s2 + 1e-18))   # 仅加极小项稳数值
        U_js[:, j:j+1] = m * one + c * h_c

    # 保留：QR 正交化 + 与原 U 的列向量符号对齐
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

def process_folder(in_dir: Path, result_txt: Path, out_root: Path, eps: float = 1e-12, num_factors: int = 1) -> None:
    """主流程：读取每个 yyyymm_full.csv，输出三类协方差与元数据
    
    Args:
        in_dir: 输入目录
        result_txt: 因子数文件（可选，如果 num_factors > 0 则不使用）
        out_root: 输出根目录
        eps: 最小特质方差
        num_factors: 固定因子数（如果 > 0，则使用此值；否则从 result_txt 读取）
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
        print(f"使用固定因子数: k = {num_factors}")
        factor_counts = None
    else:
        factor_counts = read_factor_counts(result_txt)
        print(f"从 {result_txt.name} 读取因子数")

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
        k_eff = max(0, min(k, p - 1, nobs - 1))

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
        print(f"[OK] {key}: p={p}, n={nobs}, k={k_eff} -> 已保存三类协方差。")

    if log_rows:
        df_log = pd.DataFrame(log_rows)
        df_log.to_csv(out_logs / "summary.csv", index=False)
        print(f"完成：共处理 {len(log_rows)} 个月。输出根目录：{out_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构造 LW / PCA / JS-eigvec 协方差矩阵")
    parser.add_argument("--in_dir", type=str, default="500_ret_sim",
                        help="月度 CSV 输入目录（默认 ./500_ret_sim）")
    parser.add_argument("--result_txt", type=str, default="result_500_sim.txt",
                        help="含因子数的文本文件（默认 ./result_500_sim.txt，如果指定 --num_factors 则忽略）")
    parser.add_argument("--out_root", type=str, default="covariance_outputs_sim",
                        help="输出根目录（默认 ./covariance_outputs_sim）")
    parser.add_argument("--eps", type=float, default=1e-12,
                        help="特质方差的最小截断（默认 1e-12）")
    parser.add_argument("--num_factors", type=int, default=1,
                        help="固定因子数（默认 1，适用于 1-factor 模拟；设为 0 则从 result_txt 读取）")
    return parser.parse_args()


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
