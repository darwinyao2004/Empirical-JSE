import os
import glob
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from itertools import combinations
from math import comb
from scipy.stats import norm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import multiprocessing as mp


# ---------------------------------------------------------------------
# 0. 通用工具函数
# ---------------------------------------------------------------------
def plot_eigenvalue_spectrum(cov_matrix, date, bar=0.1):        # ← 增加 date 形参
    """
    Plot the spectrum of eigenvalues of a covariance matrix.
    """'''
    eigenvalues, _ = np.linalg.eig(cov_matrix)
    eigenvalues = eigenvalues[eigenvalues > bar]
    plt.hist(eigenvalues, bins=100)
    plt.title('Eigenvalue Spectrum with Zeros Filtered')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.savefig(f"500/e_{date}.png", dpi=300)
    plt.close()'''


def z_score_normalize(data):
    """
    Normalize each column of the data using z-score normalization.
    """
    if data is None or data.size == 0:
        return None
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    return (data - mean) / std_dev


# ---------------------------------------------------------------------
# 1. Triples‑test 及因子数估计（保持原样）
# ---------------------------------------------------------------------
def _triple_kernel(a, b, c):
    return (
        np.sign(a + b - 2 * c)
        + np.sign(a + c - 2 * b)
        + np.sign(b + c - 2 * a)
    ) / 3.0


def triples_test(sample, *, alternative="greater", max_triples=150_000,
                 random_state=None):
    x = np.asarray(sample, dtype=float)
    n = x.size
    if n < 20:
        raise ValueError("Triples test needs n ≥ 20 for the asymptotics.")
    x -= np.median(x)

    rng = np.random.default_rng(random_state)
    all_idx = np.arange(n)
    N_triples = comb(n, 3)

    if N_triples <= max_triples:
        triple_index_iter = combinations(all_idx, 3)
    else:
        triple_index_iter = (
            rng.choice(all_idx, size=3, replace=False)
            for _ in range(max_triples)
        )
        N_triples = max_triples

    sum_f = 0.0
    f1 = np.zeros(n)
    for i, j, k in triple_index_iter:
        val = _triple_kernel(x[i], x[j], x[k])
        sum_f += val
        f1[i] += val
        f1[j] += val
        f1[k] += val

    U_hat = sum_f / N_triples
    denom_f1 = comb(n - 1, 2) if N_triples == comb(n, 3) else max_triples * 3 / n
    f1 /= denom_f1
    t1_hat = np.mean((f1 - U_hat) ** 2)
    sigma_hat = np.sqrt(t1_hat)

    z_stat = np.sqrt(n) * U_hat / sigma_hat
    if alternative == "greater":
        p_value = 1.0 - norm.cdf(z_stat)
    else:
        p_value = norm.cdf(z_stat)
    return z_stat, p_value


def estimate_n_factors(X, *, alpha=0.1, max_components=None,
                       pca_kwargs=None, triples_kwargs=None):
    if pca_kwargs is None:
        pca_kwargs = {}
    if triples_kwargs is None:
        triples_kwargs = {}

    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if max_components is None:
        max_components = min(n - 1, d - 1)

    Xc = X - X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=max_components, svd_solver="full", **pca_kwargs)
    scores = pca.fit_transform(Xc)
    comps = pca.components_.T

    p_vals = []
    n = 0
    for k in range(max_components + 1):
        resid = Xc if k == 0 else Xc - scores[:, :k] @ comps[:, :k].T
        R = np.sum(resid ** 2, axis=1) / d
        _, p = triples_test(R, alternative="greater", **triples_kwargs)
        p_vals.append(p)
        if p > alpha and n == 0:
            n = k
    return n, p_vals


# ---------------------------------------------------------------------
# 2. 核心计算过程，封装成单文件函数
# ---------------------------------------------------------------------
def _process_file(path):                                     # ← 新增
    """
    对单个 CSV 文件执行全部流程并保存 3 张图（λ谱图 + 双子图比较）。
    """
    date = os.path.splitext(os.path.basename(path))[0]       # ← 用文件名作 date
    data = np.loadtxt(path, delimiter=',').T                 # ← 修正变量名
    # ---- 2.1 Eigenvalue 谱图 ------------------------------------------------
    cov_matrix = np.cov(data)
    plot_eigenvalue_spectrum(cov_matrix, date, bar=0.0001)

    # ---- 2.2 因子数序贯估计（原数据 & Z‑score）------------------------------
    data_z = z_score_normalize(data)

    k_raw, p_seq_raw = estimate_n_factors(
        data,
        alpha=0.05,
        triples_kwargs=dict(max_triples=50_000)
    )
    k_z, p_seq_z = estimate_n_factors(
        data_z,
        alpha=0.05,
        triples_kwargs=dict(max_triples=50_000)
    )

    with open("result_500.txt", "a", encoding="utf-8") as f:
        f.write(f'{date} {k_raw} {k_z}\n')

    # ---- 2.3 合并成一张含两个子图的比较图 ----------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(range(len(p_seq_raw)), p_seq_raw, marker='o', linestyle='-')
    axes[0].axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
    axes[0].set_title('P-value Sequence (Raw Data)')
    axes[0].set_ylabel('P-value')
    axes[0].legend()

    axes[1].plot(range(len(p_seq_z)), p_seq_z, marker='o', linestyle='-')
    axes[1].axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
    axes[1].set_title('P-value Sequence (Z‑score Normalized)')
    axes[1].set_xlabel('Number of Components (k)')
    axes[1].set_ylabel('P-value')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"500/u_{date}_compare.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# 3. 并行遍历文件夹中所有 CSV（多核）
# ---------------------------------------------------------------------
if __name__ == "__main__":                                   # ← 新增
    folder = "500_ret_new"      # 改成你的文件夹路径或使用 sys.argv 䅥
    #mp.set_start_method("spawn", force=True)
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    #print(f"Found {len(csv_files)} CSV files")
    #list(ProcessPoolExecutor().map(_process_file, csv_files))
    with ProcessPoolExecutor() as pool:                      # ← 多核并行
        pool.map(_process_file, csv_files)

