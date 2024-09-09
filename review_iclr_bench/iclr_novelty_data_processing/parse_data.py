import json
import os
import glob
import pandas as pd
import numpy as np


f1 = "review_dataset_ICLR2022_2023.json"

f2 = "../ratings_subset.tsv"
df = pd.read_csv(f2, sep='\t')

paper_id_set = df['paper_id'].unique()

print("len(paper_id_set):", len(paper_id_set))

print(df.head())

def save_json_data_to_file(data, path):
    with open(path, 'w') as fw:
        json.dump(data, fw, ensure_ascii=False, indent=4)

paper_forum_to_score_info_dict = dict()
with open(f1, 'r') as fr:
    for i, line in enumerate(fr):
        data = json.loads(line.strip())
        b_forum = data['b_forum']
        if b_forum not in paper_id_set:
            continue
        title = data['b_title']
        if b_forum not in paper_forum_to_score_info_dict:
            paper_forum_to_score_info_dict[b_forum] = {
                "title": title,
                "c_correctness":[],
                "c_technical_novelty_and_significance":[],
                "c_empirical_novelty_and_significance":[],
                "c_recommendation":[],
                "c_confidence": []
            }
        """
        Correctness: 4: All of the claims and statements are well-supported and correct.
        Technical Novelty And Significance: 3: The contributions are significant and somewhat new. Aspects of the contributions exist in prior work.
        Empirical Novelty And Significance: 2: The contributions are only marginally significant or novel.
        Flag For Ethics Review: NO.
        Recommendation: 6: marginally above the acceptance threshold
        Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
        """
        c_correctness = data['c_correctness']
        c_technical_novelty_and_significance = data['c_technical_novelty_and_significance']
        c_empirical_novelty_and_significance = data['c_empirical_novelty_and_significance']
        c_recommendation = data['c_recommendation']
        c_confidence = data['c_confidence']
        # 处理以上数据为None的情况
        if c_correctness is not None:
            c_correctness = c_correctness.split("Correctness:")[-1].split(":")[0]
            if str(c_correctness).isdigit():
                paper_forum_to_score_info_dict[b_forum]["c_correctness"].append(int(c_correctness))
        if c_technical_novelty_and_significance is not None:
            c_technical_novelty_and_significance = c_technical_novelty_and_significance.split("Technical Novelty And Significance:")[-1].split(":")[0]
            if str(c_technical_novelty_and_significance).isdigit():
                paper_forum_to_score_info_dict[b_forum]["c_technical_novelty_and_significance"].append(int(c_technical_novelty_and_significance))
        if c_empirical_novelty_and_significance is not None:
            c_empirical_novelty_and_significance = c_empirical_novelty_and_significance.split("Empirical Novelty And Significance:")[-1].split(":")[0]
            if str(c_empirical_novelty_and_significance).isdigit():
                paper_forum_to_score_info_dict[b_forum]["c_empirical_novelty_and_significance"].append(int(c_empirical_novelty_and_significance))
        if c_recommendation is not None:
            c_recommendation = c_recommendation.split("Recommendation:")[-1].split(":")[0]
            if str(c_recommendation).isdigit():
                paper_forum_to_score_info_dict[b_forum]["c_recommendation"].append(int(c_recommendation))
        if c_confidence is not None:
            c_confidence = c_confidence.split("Confidence:")[-1].split(":")[0]
            if str(c_confidence).isdigit():
                paper_forum_to_score_info_dict[b_forum]["c_confidence"].append(int(c_confidence))

    print("len(paper_forum_to_score_info_dict):", len(paper_forum_to_score_info_dict))
    save_json_data_to_file(paper_forum_to_score_info_dict, "paper_forum_to_score_info_dict.json")


df_list = []
for key, value in paper_forum_to_score_info_dict.items():
    c_correctness = np.mean(value["c_correctness"])
    c_technical_novelty_and_significance = np.mean(value["c_technical_novelty_and_significance"])
    c_empirical_novelty_and_significance = np.mean(value["c_empirical_novelty_and_significance"])
    c_recommendation = np.mean(value["c_recommendation"])
    c_confidence = np.mean(value["c_confidence"])
    df_list.append([key, c_correctness, c_technical_novelty_and_significance, c_empirical_novelty_and_significance, c_recommendation, c_confidence])


df = pd.DataFrame(df_list, columns=['paper_id', 'mean_correctness', 'mean_technical_novelty_and_significance',
                                    "mean_empirical_novelty_and_significance", "mean_recommendation","mean_confidence"])

df.to_csv("paper_forum_to_score_info_dict.csv", index=False)





