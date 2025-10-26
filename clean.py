import os
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
        df.to_csv(output_path, index=False, header=False)
