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
        φ̂ = (1/n) Σ_t || x_t x_t^T - S ||_F^2
        γ̂ = || S - F ||_F^2
        δ = clip(φ̂/γ̂, 0, 1)
    """
    p = S.shape[0]
    n = X.shape[1]
    mu = float(np.trace(S)) / p
    F = mu * np.eye(p)

    # 计算 φ̂
    phi_acc = 0.0
    for t in range(n):
        xt = X[:, t:t + 1]            # (p,1)
        outer = xt @ xt.T             # (p,p)
        diff = outer - S
        phi_acc += float(np.sum(diff * diff))
    phi_hat = phi_acc / n

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


def js_shrink_eigenvectors(U: np.ndarray, n: int) -> np.ndarray:
    """
    对每个样本特征向量 u_j 做 James–Stein 收缩到“市场方向” m：
        m = 1/sqrt(p) * 1_p
        w_j = clip( 1 - ((p-2)*σ^2) / ||u_j - m||^2 , 0, 1 )
        u_j^JS = m + w_j (u_j - m)
    其中 σ^2 ~ 1/n 作为特征向量噪声的近似量级。最后对列做 QR 重新正交化，
    并用与原 u_j 的点积对齐符号，避免任意翻转。
    """
    p, k = U.shape
    m = np.ones((p, 1)) / math.sqrt(p)
    sigma2 = 1.0 / max(1, n)

    U_js = np.zeros_like(U)
    for j in range(k):
        u = U[:, j:j + 1]
        diff = u - m
        denom = float(np.sum(diff * diff)) + 1e-18
        w = 1.0 - ((p - 2.0) * sigma2) / denom
        w = max(0.0, min(1.0, w))
        U_js[:, j:j + 1] = m + w * diff

    # 重新正交化
    Q, _ = np.linalg.qr(U_js)
    # 与原 U 对齐符号
    for j in range(k):
        if float(np.dot(Q[:, j], U[:, j])) < 0:
            Q[:, j] *= -1.0
    return Q


def js_eigvec_factor_cov(S: np.ndarray, k: int, n: int, eps: float = 1e-12) -> np.ndarray:
    """
    JS-eigvec 因子协方差：
      1) S 的前 k 个特征对 (U_k, Λ_k)
      2) 对 U_k 列做 JS 收缩 -> \tilde U_k（并 QR 正交化）
      3) Σ_JS = \tilde U_k Λ_k \tilde U_k^T + diag(diag(S - \tilde U_k Λ_k \tilde U_k^T))
    """
    p = S.shape[0]
    k_eff = max(0, min(k, p - 1))
    if k_eff == 0:
        diag = np.maximum(np.diag(S), eps)
        return np.diag(diag)

    U, lam = top_k_eigenpairs(S, k_eff)
    U_js = js_shrink_eigenvectors(U, n)
    Sig_k = U_js @ (lam[:, None] * U_js.T)
    diag_resid = np.diag(S - Sig_k)
    psi = np.maximum(diag_resid, eps)
    Sigma = Sig_k + np.diag(psi)
    Sigma = (Sigma + Sigma.T) / 2.0
    return Sigma


def process_folder(in_dir: Path, result_txt: Path, out_root: Path, eps: float = 1e-12) -> None:
    """主流程：读取每个 yyyymm_full.csv，输出三类协方差与元数据"""
    ensure_dir(out_root)
    out_js = out_root / "JSE"
    out_lw = out_root / "LW"
    out_pca = out_root / "PCA"
    out_meta = out_root / "meta"
    out_logs = out_root / "logs"
    out_sample = out_root / "raw"
    for d in (out_js, out_lw, out_pca, out_meta, out_logs):
        ensure_dir(d)

    factor_counts = read_factor_counts(result_txt)

    files = sorted([p for p in in_dir.glob("*.csv") if p.name.endswith("_full.csv")])
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
        if key not in factor_counts:
            print(f"[SKIP] {key}: 在 {result_txt.name} 中未找到因子数，跳过。")
            continue

        k = int(factor_counts[key])

        # ✨改动：读取时区分 permno 与收益矩阵
        df_all = pd.read_csv(fpath, header=None)
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
        save_with_permno(S,           out_sample / f"{key}_cov.csv", kept_permno)
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
    parser.add_argument("--in_dir", type=str, default="./500_ret_new",
                        help="月度 CSV 输入目录（默认 ./500_ret_new）")
    parser.add_argument("--result_txt", type=str, default="./result_500.txt",
                        help="含因子数的文本文件（默认 ./result_500.txt）")
    parser.add_argument("--out_root", type=str, default="./covariance_outputs",
                        help="输出根目录（默认 ./covariance_outputs）")
    parser.add_argument("--eps", type=float, default=1e-12,
                        help="特质方差的最小截断（默认 1e-12）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    in_dir = Path(args.in_dir).expanduser().resolve()
    result_txt = Path(args.result_txt).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    print("输入目录:   ", in_dir)
    print("因子数文件: ", result_txt)
    print("输出根目录: ", out_root)

    if not in_dir.exists():
        raise FileNotFoundError(f"未找到输入目录：{in_dir}")
    if not result_txt.exists():
        raise FileNotFoundError(f"未找到因子数文件：{result_txt}")

    ensure_dir(out_root)
    process_folder(in_dir, result_txt, out_root, eps=float(args.eps))
