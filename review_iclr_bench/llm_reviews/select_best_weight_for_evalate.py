"""
你有以下特征，你的最终score为这些特征的加权和， 'Soundness', 'Presentation', 'Contribution', 'Overall', 'Confidence', 'Originality', 'Quality', 'Clarity', 'Significance' ，这些指标都是整数，有些数值是缺失的，其中labels为0,1两种选择，通过权重搜索， 使最终的score得到的f1指标最大，出了拿到权重系数，还可以对最终的score的阈值进行筛选，筛选出f1最高的阈值 给出python代码，输出并保留最终的权重，以后哪新数据预测时候，加载权重来计算
"""


import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score
import pickle
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import joblib

# f1 = '/Users/huxiang/Documents/GitHub/Research/AI-Scientist/review_iclr_bench/llm_reviews/gpt-4o-2024-05-13_temp_0_1_fewshot_1_reflect_5_ensemble_5_pages_all.csv'
f1 = "/Users/huxiang/Documents/GitHub/Research/AI-Scientist/review_iclr_bench/llm_reviews/our0908_gpt-4o_temp_0_1_fewshot_3_reflect_5_num_reviews_500_ensemble_5_only_problem_and_method_1_pages_all.csv"
# f1 = "/Users/huxiang/Documents/GitHub/Research/AI-Scientist/review_iclr_bench/llm_reviews/our0908_gpt-4o_temp_0_1_reflect_5_num_reviews_500_ensemble_5_only_problem_and_method_1_pages_all.csv"
df = pd.read_csv(f1)

f2 = "/Users/huxiang/Documents/GitHub/Research/AI-Scientist/review_iclr_bench/ratings_subset.tsv"

df2 = pd.read_csv(f2, sep='\t')

df_merged = df.merge(df2, on='paper_id', how='left')

# 你有以下特征：'Soundness', 'Presentation', 'Contribution', 'Overall', 'Confidence', 'Originality', 'Quality', 'Clarity', 'Significance'
X = df_merged[['Soundness', 'Presentation', 'Contribution', 'Overall', 'Confidence', 'Originality', 'Quality', 'Clarity', 'Significance']]


# 假设我们有一个DataFrame df，其中包含特征和标签
# df = pd.read_csv('your_data.csv')
#
# # 这里是一个示例数据框架
# data = {
#     'Soundness': [1, 2, 3, 4, 5],
#     'Presentation': [2, 3, 4, 5, 1],
#     'Contribution': [3, 4, 5, 1, 2],
#     'Overall': [4, 5, 1, 2, 3],
#     'Confidence': [5, 1, 2, 3, 4],
#     'Originality': [1, 2, 3, 4, 5],
#     'Quality': [2, 3, 4, 5, 1],
#     'Clarity': [3, 4, 5, 1, 2],
#     'Significance': [4, 5, 1, 2, 3],
#     'label': [0, 1, 0, 1, 0]
# }
#
# df = pd.DataFrame(data)

# 分离特征和标签
# X = df.drop(columns=['label'])
# y = df['label']


df_merged['label'] = (df_merged['decision'] !=  'Reject').astype(int)
df_merged['label'].value_counts()
# y = (df_merged['decision'] !=  'Reject').astype(int)
# print(df_merged['decision'].value_counts())
y = (df_merged['decision'] !=  'Reject').astype(int)

# 填充缺失值
X = X.fillna(X.mean())

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义权重搜索范围
param_grid = {
    'weights': [np.random.dirichlet(np.ones(9), size=1)[0] for _ in range(1000)],
    'threshold': np.linspace(0, 1, 100)
}

# 自定义评分函数
# def custom_f1_score(weights, threshold, X, y):
#     scores = np.dot(X, weights)
#     y_pred = (scores >= threshold).astype(int)
#     return f1_score(y, y_pred)

from analyze_f1_from_score import compute_metrics_from_scrach
def custom_f1_score(weights, threshold, X, y, return_f1_only=True):
    scores = np.dot(X, weights)
    y_pred = (scores >= threshold).astype(int)
    if return_f1_only:
        return compute_metrics_from_scrach(y, y_pred)[1]
    else:
        return compute_metrics_from_scrach(y, y_pred)

# 网格搜索
best_f1 = 0
best_weights = None
best_threshold = None

for weights in param_grid['weights']:
    for threshold in param_grid['threshold']:
        f1 = custom_f1_score(weights, threshold, X_train, y_train)
        if f1 > best_f1:
            best_f1 = f1
            best_weights = weights
            best_threshold = threshold

# 在测试集上评估
accuracy, test_f1, roc, fpr, fnr = custom_f1_score(best_weights, best_threshold, X_test, y_test)
print(f"Best F1 Score: {best_f1}")
print(f"Best Weights: {best_weights}")
print(f"Best Threshold: {best_threshold}")
print(f"Test F1 Score: {test_f1},  Accuracy: {accuracy}, ROC: {roc}, FPR: {fpr}, FNR: {fnr}")

# 保存权重和阈值
joblib.dump((best_weights, best_threshold), 'best_weights_threshold.pkl')

# 加载权重和阈值
loaded_weights, loaded_threshold = joblib.load('best_weights_threshold.pkl')

# 使用加载的权重和阈值进行预测
def predict(X, weights, threshold):
    scores = np.dot(X, weights)
    return (scores >= threshold).astype(int)

# 示例预测
new_data = pd.DataFrame({
    'Soundness': [3],
    'Presentation': [4],
    'Contribution': [5],
    'Overall': [1],
    'Confidence': [2],
    'Originality': [3],
    'Quality': [4],
    'Clarity': [5],
    'Significance': [1]
})

new_data = new_data.fillna(new_data.mean())
prediction = predict(new_data, loaded_weights, loaded_threshold)
print(f"Prediction: {prediction}")