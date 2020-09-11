import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from scipy import stats  # 正态分布检验时使用到
data = load_iris()  # 载入数据集



"""  data数据集的格式解析
data = {data：矩阵， 
        target：[0...1...2..]150个， 
        ...: ...  ，
        filename：['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']}
data： 特征矩阵 150×4
target： 三个亚种，分别用0,1,2表示  size=150
target_names：  三个亚种的名称
DESCR：  无解释
feature_names：  四个特征的相关描述，以不同器官的长、宽作为分类依据
filename：  四个特征的名称
"""

'''1、查看数据'''
# for k in data:
#     print(data[k])
#     print('===============')
'''2、使用pandas 分析数据   特点：对矩阵格式的数据  结构化处理'''
df = pd.DataFrame(data.data)  # 将矩阵数据转换成表的形式  150*4的形状大小 这个方法还有两个参数: 行索引名称index=[] 列索引名称columns=[] ,默认都从0开始打印
df.columns = data.feature_names # 对df数据表的每一列进行命名 一共有4列，feature_names有4个元素
df['species'] = [ data['target_names'][x] for x in data.target ]  # 添加新一列，使表的尺寸变成150*5，新添加的一列的列名是'species'
# print(data.target.size)
# print(df.head(3))  # 查看前三行
df_cnt= df['species'].value_counts().reset_index()  # 对'species'列先计数， 再转换成pd表格格式
# print(df_cnt)  # 统计样本中三个分类的数量，判断样本分类是否均衡。三个样本数量都是50
sns.barplot(data=df_cnt, x='index', y='species')
# plt.show()  # 显示图像
describ = df.describe()  # 结果每一行分别表示计数，平均值，标准差，最小值，较低的百分位数和50、最大值
# print(describ) 得到的是一个表
for i in range(4):
    name = data.feature_names[i]
    ax = plt.subplot(2,2,i+1) # ax将整个图像分为2行2列，当前位置为1
    stats.probplot(df[name], plot=ax) # probplot可以选择计算数据的最佳拟合线
    ax.set_title(name)
# plt.show()
# 计算每一个花分类的平均值、方差。这里使用的是一种基于透视表的方法。将150×4的二维矩阵变成600×1的一维矩阵。150条÷3类×4特征=200 200×3类=600条
df1 = pd.melt(df, id_vars=['species'])  # df1 = 600×1， df['species'] 这一列在上面已经添加表的第5列了
df2 = df1.pivot_table(index=['species'], columns=['variable'],aggfunc=[np.mean,np.var])  # 以index分类为行名称，针对每个特征计算平均值及方差
fig= plt.figure(figsize=(12,4))
for i in range(3):  # 对每个亚种分类的分组进行正态分布检验
    name = data.target_names[i]
    ax = plt.subplot(1,3,i+1)
    # stats.probplot(df[df['species']==name][data.feature_names[2]],plot=ax)
    stats.probplot(df[df['species'] == name][data.feature_names[3]], plot=ax)
    ax.set_title(name)
# plt.show()




