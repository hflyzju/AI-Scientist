import pandas as pd
import numpy as np


df = pd.read_csv("paper_forum_to_score_info_dict.csv")
"""
['paper_id', 'mean_correctness',
       'mean_technical_novelty_and_significance',
       'mean_empirical_novelty_and_significance', 'mean_recommendation',
       'mean_confidence'],
"""

print(df.columns)

f2 = "../ratings_subset.tsv"
df2 = pd.read_csv(f2, sep='\t')
print(df2.columns)
# ['paper_id', '0', '1', '2', '3', '4', '5', '6', 'decision']


# f3 = "../llm_reviews/gpt-4o-2024-05-13_temp_0_1_fewshot_1_reflect_5_ensemble_5_pages_all.csv"
# f3 = "../llm_reviews/gpt-4o-mini-2024-07-18_temp_0_1_fewshot_1_reflect_5_ensemble_5_pages_all.csv"
# f3 = "../llm_reviews/gpt-4o-2024-05-13_temp_0_1_reflect_5_ensemble_5_pages_all.csv"
# f3 = "../llm_reviews/0909_gpt-4o_temp_0_1_reflect_5_ensemble_5_pages_all.csv"
# f3 = "../llm_reviews/0909_title_abstract_gpt-4o_temp_0_1_num_reviews_20_reflect_5_ensemble_5_pages_all.csv"
f3 = '../llm_reviews/0909_problem_method_gpt-4o_temp_0_1_num_reviews_20_reflect_5_ensemble_5_pages_all.csv'
# f3 = "../llm_reviews/our0908_gpt-4o_temp_0_1_fewshot_3_reflect_5_num_reviews_500_ensemble_5_only_problem_and_method_1_pages_all.csv"
# f3 = "../llm_reviews/our0908_new_prompt_gpt-4o_temp_0_1_fewshot_3_reflect_5_num_reviews_500_ensemble_5_only_problem_and_method_1_pages_all.csv"
df3 = pd.read_csv(f3, sep=',')
print(df3.columns)
# ['paper_id,Summary,Questions,Limitations,Ethical Concerns,Soundness,Presentation,Contribution,Overall,Confidence,Strengths,Weaknesses,Originality,Quality,Clarity,Significance,Decision']


# 杰卡德相似系数 (Jaccard Similarity Coefficient)： 适用于集合或二值型特征。
def manhattan_distance(feature1, feature2):
    return np.sum(np.abs(np.array(feature1) - np.array(feature2)))


# 皮尔逊相关系数 (Pearson Correlation Coefficient)： 适用于数值型特征，尤其是线性相关的特征。
def pearson_correlation(feature1, feature2):
    return np.corrcoef(feature1, feature2)[0, 1]


df1_merged = df.merge(df3, on=["paper_id"])

print("len(df1_merged):", len(df1_merged))
print(df1_merged.columns)


for human_mean_feature in ['mean_technical_novelty_and_significance', 'mean_empirical_novelty_and_significance']:
    df1_merged[human_mean_feature].fillna(df1_merged[human_mean_feature].mean(), inplace=True)


for human_mean_feature in ['mean_technical_novelty_and_significance', 'mean_empirical_novelty_and_significance']:
    for machine_feature in ["Soundness", "Originality","Quality","Significance", "Overall"]:
    # for machine_feature in ["Originality","Quality","Clarity","Significance"]:
        md = manhattan_distance(df1_merged[human_mean_feature], df1_merged[machine_feature])
        pc = pearson_correlation(df1_merged[human_mean_feature], df1_merged[machine_feature])
        print(f"human_mean_feature:{human_mean_feature}, machine_feature:{machine_feature}, md:{md}, pc:{pc}")


# for human_mean_feature in ['mean_technical_novelty_and_significance', 'mean_empirical_novelty_and_significance']:
#     if human_mean_feature == "mean_empirical_novelty_and_significance":
#         df1_merged[human_mean_feature].fillna(df1_merged[human_mean_feature].mean(), inplace=True)
#         print(1)
#     for i in range(10):
#         df1_merged["Ori_Sig"] = df1_merged["Originality"] * (i / 10.0) + df1_merged["Significance"] * ((10 - i) / 10.0)
#         ML_Feature_Key = "Ori_Sig"
#         md = manhattan_distance(df1_merged[human_mean_feature], df1_merged[ML_Feature_Key])
#         pc = pearson_correlation(df1_merged[human_mean_feature], df1_merged[ML_Feature_Key])
#         print(
#             f"weight:{i / 10.0}, human_mean_feature:{human_mean_feature}, machine_feature:{ML_Feature_Key}, md:{md}, pc:{pc}")
#
#


