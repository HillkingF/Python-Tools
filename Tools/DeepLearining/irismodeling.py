from sklearn.datasets import load_iris  # 使用鸢尾花数据集进行数据预处理和建模
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# 获取数据，载入数据集
data = load_iris()  # 载入数据集
df = pd.DataFrame(data.data)  # 将矩阵数据转换成表的形式  150*4的形状大小 这个方法还有两个参数: 行索引名称index=[] 列索引名称columns=[] ,默认都从0开始打印
df.columns = data.feature_names # 对df数据表的每一列进行命名 一共有4列，feature_names有4个元素
df['species'] = [ data['target_names'][x] for x in data.target ]  # 添加新一列，使表的尺寸变成150*5，新添加的一列的列名是'species'

# 数据准备---划分数据集
df_train, df_val = train_test_split(df, train_size=0.8, random_state=0)
X_train = df_train.drop(['species'], axis=1)  # X_是提取分类特征
X_val = df_val.drop(['species'], axis=1)
Y_train = df_train['species']   # Y_ 是提取分类结果
Y_val = df_val['species']

# 设定分布X_scarler ，用训练集估计(fit)分布，然后对验证集进行转换 , 一般都这样使用
# X_scaler.fit_transform(trainData)对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），
# 然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等。
# 根据对之前部分X_scaler.fit_transform(trainData)进行fit的整体指标，
# 对剩余的数据（testData，valData）使用同样的均值、方差、最大最小值等指标进行转换transform(testData)，
# 从而保证train、val、test处理方式相同。
# axis 指示维度，X_trainT.mean(axis=0)表示输出每一列的维度，X_trainT.mean(axis=1)表示输出每一行的维度
# mean()计算的是某个维度上的平均值   var()计算的是某个维度上的方差  std()是标准差
# 数据准备---数据标准化
X_scaler = StandardScaler()  # 数据预处理标准化StandardScaler模型
X_trainT = X_scaler.fit_transform(X_train)  # fit_transform是fit和transform的组合，既包括训练又包含转换。
X_valT = X_scaler.transform(X_val)
print(X_trainT.mean(axis=0)) # axis=1
print(X_trainT.var(axis=0))
print(X_valT.mean(axis=0))
print(X_valT.var(axis=0))

