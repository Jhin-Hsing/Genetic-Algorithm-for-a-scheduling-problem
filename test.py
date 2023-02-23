import pandas as pd

order = pd.read_excel('製令單_1121-1125 - 少.xlsx')

new_index = [10, 4, 9, 2, 16]

df = order.reindex(new_index)


ts1 = df['預計出貨'][10]
ts2 = pd.to_datetime('2022/11/30', format='%Y/%m/%d')

print(ts1)
print(ts2)

print(ts1<ts2)