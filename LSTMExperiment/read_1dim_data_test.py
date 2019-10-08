import pandas as pd
import numpy as np


f = open('temp.csv')
df = pd.read_csv(f)  # 读入数据
data = df.iloc[:, :7].values  # 取第1-7列

print(data)