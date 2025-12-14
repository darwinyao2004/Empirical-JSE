import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === 需要修改的参数 ===
input_dir = r"./"   # 放 CSV 文件的文件夹路径
output_dir = r"./hist_output" # 输出直方图图片的文件夹路径
bins = 30                     # 直方图的柱子数量

os.makedirs(output_dir, exist_ok=True)

# 遍历文件夹下的所有 csv 文件
for filename in os.listdir(input_dir):
    if not filename.lower().endswith(".csv"):
        continue  # 跳过非 csv 文件

    csv_path = os.path.join(input_dir, filename)
    print(f"Processing: {csv_path}")

    # 读取 CSV：原文件没有标题行，所以 header=None
    # 第一列是变量名，剩下的是协方差矩阵
    df = pd.read_csv(csv_path, header=None)

    # 变量名列（如需要可以留着用）
    var_names = df.iloc[:, 0]

    # 协方差矩阵
    cov_df = df.iloc[:, 1:]

    # 转成 numpy 数组
    cov = cov_df.to_numpy(dtype=float)

    # 检查是否为方阵，不是的话取最小的正方子矩阵
    n_rows, n_cols = cov.shape
    n = min(n_rows, n_cols)
    if n_rows != n_cols:
        print(f"  Warning: covariance matrix in {filename} is not square "
              f"({n_rows}x{n_cols}), using top-left {n}x{n} submatrix.")
        cov = cov[:n, :n]

    # 提取对角线上的所有值
    diag_vals = np.diag(cov)

    # 画直方图
    plt.figure()
    plt.hist(diag_vals, bins=bins)
    plt.xlabel("Diagonal values")
    plt.ylabel("Frequency")
    plt.title(f"Diagonal Histogram - {filename}")

    # 保存图片
    base_name = os.path.splitext(filename)[0]
    out_path = os.path.join(output_dir, f"{base_name}_diag_hist.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Saved histogram to: {out_path}")

print("Done.")
