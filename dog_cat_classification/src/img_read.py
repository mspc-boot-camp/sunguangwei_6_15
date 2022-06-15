#因从img_read.py往data_preprocess.py传递数据时一直报错，且没修改好，故此处的函数再data_preprocess.py中
df = []
for i in range(10):
    df.append([i,i])
print(df)
