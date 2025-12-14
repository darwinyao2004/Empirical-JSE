'''import os
import pandas as pd

input_folder = r"ret_full_new"
output_folder = r"500_ret_new"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        
        df = pd.read_csv(file_path, header=0)  # 第一行作为表头
        keep_cols = ['permno'] + [c for c in df.columns if c not in ('permno', 'mkt_cap')]
        df = df[keep_cols]
        df = df.head(500)
        #num = df.apply(pd.to_numeric, errors='coerce')
        #df = df[~((num.gt(0.5) | num.lt(-0.5)).any(axis=1))]
        
        output_path = os.path.join(output_folder, filename)
        df.to_csv(output_path, index=False, header=False)'''
import os
import pandas as pd
import numpy as np

input_folder = r"ret_full_new"
output_folder = r"500_ret_exlude_zero_var"

# ===在此处调整过滤阈值===
# 1. 波动率阈值：标准差低于此值（例如停牌股）将被剔除
MIN_VOL_THRESHOLD = 3e-3  
# 2. 绝对相关性阈值：与市场平均表现的相关性绝对值低于此值将被剔除
MIN_ABS_CORR_THRESHOLD = 0.05 
# ======================

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        
        # 1. 读取数据
        try:
            df = pd.read_csv(file_path, header=0)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        # 2. 识别收益率数据列
        # 假设除了 permno 和 mkt_cap 之外的列都是时间序列的收益率数据
        ret_cols = [c for c in df.columns if c not in ('permno', 'mkt_cap')]
        
        # 确保收益率列是数值型（处理可能存在的非数值字符）
        df[ret_cols] = df[ret_cols].apply(pd.to_numeric, errors='coerce')

        # 3. 计算统计指标
        # A. 计算波动率 (Standard Deviation)
        volatility = df[ret_cols].std(axis=1)
        
        # B. 计算相关性 (Correlation with Market Mean)
        # 先计算这一段时间的市场平均收益率序列
        market_return_series = df[ret_cols].mean(axis=0) 
        # 计算每只股票与市场均值的相关性
        correlation = df[ret_cols].corrwith(market_return_series, axis=1)

        # 4. 构建过滤 Mask (保留符合条件的股票)
        # 条件1: 波动率不能太低 (排除长期停牌或死股)
        # 条件2: 绝对相关性不能太低 (排除与市场完全脱钩的噪音股)
        # 条件3: 市值不能为空 (如果市值缺失，无法进行 Top 500 排序)
        keep_mask = (
            (volatility > MIN_VOL_THRESHOLD) & 
            (correlation.abs() > MIN_ABS_CORR_THRESHOLD) &
            (df['mkt_cap'].notna())
        )
        
        df_filtered = df[keep_mask].copy()

        # 5. 选取 Top 500
        # 逻辑：在清洗后的股票中，按市值从大到小排序，取前 500 个
        # 这样能保证取到的一定是流动性最好、最主要的那 500 只
        if len(df_filtered) >= 500:
            df_final = df_filtered.sort_values(by='mkt_cap', ascending=False).head(500)
        else:
            # 如果清洗完不够 500 只，则发出警告，并保留所有剩余的
            print(f"Warning: File {filename} has only {len(df_filtered)} stocks after filtering (Target: 500).")
            df_final = df_filtered

        # 6. 整理输出格式
        # 按照原来的要求，只保留 permno 和收益率列，去掉 mkt_cap
        final_cols = ['permno'] + ret_cols
        df_final = df_final[final_cols]
        
        # 7. 保存
        output_path = os.path.join(output_folder, filename)
        # 注意：这里保留了 header=True 方便查看，如果下游程序严格不需要表头，改为 False
        df_final.to_csv(output_path, index=False, header=False)
        
        print(f"Processed {filename}: kept {len(df_final)} stocks.")
