import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def process_cov_csv_folder(folder_path, output_folder=None, bins=30):
    """
    对 folder_path 下所有 csv 文件：
      1. 读取协方差矩阵（第一列为变量名，后面是协方差矩阵）
      2. 求最大特征值对应的特征向量
      3. 绘制该特征向量分量的直方图，并保存为 png

    :param folder_path: 包含若干 csv 文件的文件夹路径
    :param output_folder: 输出图片的文件夹路径；如果为 None，则输出到原 csv 所在文件夹
    :param bins: 直方图的箱数
    """
    folder_path = Path(folder_path)

    if output_folder is None:
        output_folder = folder_path
    else:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

    # 遍历该目录下所有 csv 文件
    for csv_path in folder_path.glob("*.csv"):
        print(f"Processing: {csv_path}")

        # 读取 csv。根据你给的样例，没有表头，所以 header=None
        df = pd.read_csv(csv_path, header=None)

        # 第一列是变量名，剩下的是协方差矩阵
        # df 的形状应该是 (n, n+1)，n 为变量个数
        cov_matrix = df.iloc[:, 1:].to_numpy(dtype=float)

        # 检查是否为方阵
        if cov_matrix.shape[0] != cov_matrix.shape[1]:
            print(f"  [Warning] {csv_path.name} 协方差矩阵不是方阵，跳过。"
                  f" shape={cov_matrix.shape}")
            continue

        # 使用 eigh（适合对称矩阵）来分解
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)

        # 找到最大特征值对应的下标
        max_idx = np.argmax(eigvals)
        largest_eigvec = eigvecs[:, max_idx]

        # 画直方图
        plt.figure()
        plt.hist(largest_eigvec, bins=bins)
        plt.title(f"Largest eigenvector histogram\n{csv_path.name}")
        plt.xlabel("Component value")
        plt.ylabel("Frequency")
        plt.tight_layout()

        # 输出文件名：原文件名 + _largest_eigvec_hist.png
        out_name = csv_path.stem + "_largest_eigvec_hist.png"
        out_path = output_folder / out_name
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"  Saved histogram to: {out_path}")


if __name__ == "__main__":
    # 把这里改成你实际的文件夹路径
    folder = r"."
    # 可选：单独指定输出目录；不指定就和原 csv 在同一目录
    out_folder = None  # 或者 r"/path/to/save/histograms"

    process_cov_csv_folder(folder, out_folder, bins=30)
