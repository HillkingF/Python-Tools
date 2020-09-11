import pandas as pd
from sklearn import svm
from numpy import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
'''
data = pd.read_csv('../jiqixuexi/data/bank-additional-full.csv', sep=';')
# print(data.info)
# print(data.describe())
# # #统计缺失值
# # for col in data.columns: # 每一列
# #     if type(data[col][0]) is str:
# #         count = data[data[col]=='unknown']['y'].count()
# #         print('"unknown" data of ' + col + ': ' + str(count))
pridata = data.copy()

# 去掉所有的缺失值，分类数值化
for col in data.columns:
    data = data[~data[col].isin(['unknown'])]
coldict = ['job', 'marital', 'education', 'default', 'housing', 'loan',
           'contact', 'month', 'day_of_week', 'poutcome', 'y']
spdict = dict()
for col in data.columns:  # col是每一列
    if col in coldict:  # 判断出这一列是需要数值化的列，将这一列中不同的特征加入字典
        count = 0
        eachcoldict = {}
        for x in data[col]:
            if x not in eachcoldict:
                eachcoldict[x] = count
                count += 1
        spdict[col] = eachcoldict
for key in spdict.keys():
    for childtype in spdict[key].keys():
        data['temp'] = data[key]
        data.loc[data[key]==childtype, 'temp'] = spdict[key][childtype]
        data[key] = data.temp
data = data.drop(['temp'], axis=1)
print(data.info)
data.to_csv('../jiqixuexi/data/bank_tonumber_1.csv', header=False, index=False)'''

# 统计删去缺失值的类别数量  yes  和 no  类
datav1 = pd.read_csv('../DeepLearining/data/bank_tonumber.csv', sep=',')
datav1count = datav1.y.value_counts().reset_index()
# sns.barplot(data=datav1count, x='index', y='y')
# plt.show()
# print(datav1count)
# 划分数据集
datav1 = pd.DataFrame(datav1)
def split_data(data):
    data_len = data['y'].count()
    split1 = int(data_len * 0.6)
    split2 = int(data_len * 0.8)
    train_data = data[:split1]
    cv_data = data[split1:split2]
    test_data = data[split2:]
    return train_data, cv_data, test_data
trainset, valset, testset = split_data(datav1)
# SMOTE处理数据不均衡问题
X_resampled_smote, y_resampled_smote = SMOTE().fit_sample(trainset.iloc[:, :-1], trainset.iloc[:, -1:])



# 第三部分 训练模型
# SVM 决策树  朴素贝叶斯  KNN   随机森林  逻辑回归   集成学习  GBDT
methods = [svm.SVC(gamma='auto'),
          DecisionTreeClassifier(criterion='entropy'),
          GaussianNB(),
          KNeighborsClassifier(n_neighbors=3),
          RandomForestClassifier(n_estimators=50),
          LogisticRegression(),
          VotingClassifier(estimators=[
              ("log_cla",LogisticRegression()),
              ("svm_cla",svm.SVC(probability=True)),
              ("knn", KNeighborsClassifier()),
              ("tree",DecisionTreeClassifier(random_state=666))
              ], voting="soft"),
          GradientBoostingClassifier(n_estimators=200)
          ]
methodname = ['SVM', '决策树', '朴素贝叶斯', 'KNN', '随机森林', '逻辑回归', '集成学习', 'GBDT']
for i, method in enumerate(methods):
    # method.fit(trainset.iloc[:, :-1], trainset.iloc[:, -1:])
    # score = method.score(testset.iloc[:, :-1], testset.iloc[:, -1:])
    X_resampled_smote, y_resampled_smote = \
        SMOTE().fit_sample(trainset.iloc[:, :-1], trainset.iloc[:, -1:])
    method.fit(X_resampled_smote, y_resampled_smote)
    score = method.score(X_resampled_smote, y_resampled_smote)
    print('算法准确度 ===== ' + methodname[i] + ' : ' + str(score))










