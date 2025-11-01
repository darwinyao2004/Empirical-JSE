import matplotlib.pyplot as plt
import pandas as pd

# 读取并处理数据
data = []
with open('result_500.txt', 'r') as file:
    for line in file:
        parts = line.strip().split()
        date_str = parts[0].replace('_full', '')
        # 按年月格式解析
        date = pd.to_datetime(date_str, format='%Y%m')
        data.append((date, int(parts[1]), int(parts[2])))

# 转为 DataFrame 并排序
df = pd.DataFrame(data, columns=['date', 'n1', 'n2']).sort_values('date')

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['n1'], label='Covariance')
plt.plot(df['date'], df['n2'], label='Correlation')
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Number of Factors Over Time (p=500)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_500.png')
plt.close()
'''
data = []
with open('result_1000.txt', 'r') as file:
    for line in file:
        parts = line.strip().split()
        date_str = parts[0].replace('_full', '')
        # 按年月格式解析
        date = pd.to_datetime(date_str, format='%Y%m')
        data.append((date, int(parts[1]), int(parts[2])))

# 转为 DataFrame 并排序
df = pd.DataFrame(data, columns=['date', 'n1', 'n2']).sort_values('date')

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['n1'], label='Covariance')
plt.plot(df['date'], df['n2'], label='Correlation')
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Number of Factors Over Time (p=1000)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_1000.png')
plt.close()

data = []
with open('result_1500.txt', 'r') as file:
    for line in file:
        parts = line.strip().split()
        date_str = parts[0].replace('_full', '')
        # 按年月格式解析
        date = pd.to_datetime(date_str, format='%Y%m')
        data.append((date, int(parts[1]), int(parts[2])))

# 转为 DataFrame 并排序
df = pd.DataFrame(data, columns=['date', 'n1', 'n2']).sort_values('date')

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['n1'], label='Covariance')
plt.plot(df['date'], df['n2'], label='Correlation')
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Number of Factors Over Time (p=1500)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_1500.png')
plt.close()


data = []
with open('result_2000.txt', 'r') as file:
    for line in file:
        parts = line.strip().split()
        date_str = parts[0].replace('_full', '')
        date = pd.to_datetime(date_str, format='%Y%m')
        data.append((date, int(parts[1]), int(parts[2])))

# 转为 DataFrame 并排序
df = pd.DataFrame(data, columns=['date', 'n1', 'n2']).sort_values('date')

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['n1'], label='Covariance')
plt.plot(df['date'], df['n2'], label='Correlation')
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Number of Factors Over Time (p=2000)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_2000.png')
plt.close()


data = []
with open('result_2500.txt', 'r') as file:
    for line in file:
        parts = line.strip().split()
        date_str = parts[0].replace('_full', '')
        date = pd.to_datetime(date_str, format='%Y%m')
        data.append((date, int(parts[1]), int(parts[2])))

# 转为 DataFrame 并排序
df = pd.DataFrame(data, columns=['date', 'n1', 'n2']).sort_values('date')

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['n1'], label='Covariance')
plt.plot(df['date'], df['n2'], label='Correlation')
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Number of Factors Over Time (p=2500)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_2500.png')
plt.close()
'''
