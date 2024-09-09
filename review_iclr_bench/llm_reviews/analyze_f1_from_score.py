import pandas as pd
import numpy as np



from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix,accuracy_score
def compute_metrics(labels, preds):
    f1 = round(f1_score(labels, preds), 2)
    roc = round(roc_auc_score(labels, preds), 2)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    correct = sum([1 for i in range(len(labels)) if labels[i] == preds[i]])
    accuracy = correct / len(labels)
    return accuracy, f1, roc, fpr, fnr

def compute_metrics_from_scrach(labels, preds):
    """
    计算f1,precision,recall,acc
    Args:
          labels(list): 0, 1
          preds(list): 0, 1
    Returns:
          f1, precision, recall, acc
    """
    labels = list(labels)
    TP = sum([1 for i in range(len(labels)) if labels[i] == 1 and preds[i] == 1])
    TN = sum([1 for i in range(len(labels)) if labels[i] == 0 and preds[i] == 0])
    FP = sum([1 for i in range(len(labels)) if labels[i] == 0 and preds[i] == 1])
    FN = sum([1 for i in range(len(labels)) if labels[i] == 1 and preds[i] == 0])
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    roc = round(roc_auc_score(labels, preds), 2)
    fpr = FP / (FP + TN + 1e-8)
    fnr = FN / (FN + TP + 1e-8)
    return accuracy, f1, roc, fpr, fnr

if __name__ == '__main__':

    # f1 = '/Users/huxiang/Documents/GitHub/Research/AI-Scientist/review_iclr_bench/llm_reviews/gpt-4o-2024-05-13_temp_0_1_fewshot_1_reflect_5_ensemble_5_pages_all.csv'
    # f1 = "/Users/huxiang/Documents/GitHub/Research/AI-Scientist/review_iclr_bench/llm_reviews/our0908_gpt-4o_temp_0_1_fewshot_3_reflect_5_num_reviews_500_ensemble_5_only_problem_and_method_1_pages_all.csv"
    f1 = "/Users/huxiang/Documents/GitHub/Research/AI-Scientist/review_iclr_bench/llm_reviews/our0908_gpt-4o_temp_0_1_reflect_5_num_reviews_500_ensemble_5_only_problem_and_method_1_pages_all.csv"
    # f1 = '/Users/huxiang/Documents/GitHub/Research/AI-Scientist/review_iclr_bench/llm_reviews/our0908_new_prompt_gpt-4o_temp_0_1_fewshot_3_reflect_5_num_reviews_500_ensemble_5_only_problem_and_method_1_pages_all.csv'

    df = pd.read_csv(f1)

    print(df.columns)

    columns = ['paper_id', 'Summary', 'Questions', 'Limitations', 'Ethical Concerns',
               'Soundness', 'Presentation', 'Contribution', 'Overall', 'Confidence',
               'Strengths', 'Weaknesses', 'Originality', 'Quality', 'Clarity',
               'Significance', 'Decision'],
    print(df.head())

    f2 = "/Users/huxiang/Documents/GitHub/Research/AI-Scientist/review_iclr_bench/ratings_subset.tsv"

    df2 = pd.read_csv(f2, sep='\t')

    print(df2.columns)

    print(df2.head())

    print(len(df))
    print(len(df2))

    df_merged = df.merge(df2, on='paper_id', how='left')

    print(df_merged.columns)

    print(df_merged.head())

    print(df_merged.columns)

    print(len(df_merged))

    print(df_merged['Decision'].value_counts())
    print(df_merged['decision'].value_counts())

    df_merged['decision'] = df_merged['decision'].apply(lambda x: "Accept" if x != "Reject" else "Reject")
    print(df_merged['decision'].value_counts())

    """
    labels,preds分别为['Accept', 'Reject']两个选项的列表，给计算f1,precision,recall,acc的函数

    """

    labels = (df_merged['decision'] != 'Reject').astype(int)
    preds = (df_merged['Decision'] != 'Reject').astype(int)

    accuracy, f1, roc, fpr, fnr = compute_metrics(labels, preds)

    print(f"Accuracy: {accuracy}, F1: {f1}, ROC: {roc}, FPR: {fpr}, FNR: {fnr}")

    print(df_merged['Overall'].value_counts())

    """
    填充nan值为平均值
    'Originality', 'Quality', 'Clarity',
           'Significance', Overall
    """
    df_merged['Originality'].fillna(df_merged['Originality'].mean(), inplace=True)
    df_merged['Quality'].fillna(df_merged['Quality'].mean(), inplace=True)
    # df_merged['Clarity'].fillna(df_merged['Clarity'].mean(), inplace=True)
    df_merged['Significance'].fillna(df_merged['Significance'].mean(), inplace=True)

    key = "Overall"

    # df_merged["score"] = (df_merged['Originality'] + df_merged['Quality'] + df_merged['Clarity'] + df_merged['Significance']) / 4.0 * 0.5 + df['Overall'] * 0.5
    df_merged["score"] = (df_merged['Originality'] + df_merged['Quality'] + df_merged['Significance']) / 3.0 * 0.5 + df['Overall'] * 0.5

    # 你有以下特征：'Soundness', 'Presentation', 'Contribution', 'Overall', 'Confidence', 'Originality', 'Quality', 'Clarity', 'Significance'
    # 记得先填充nan的为平均值
    # 然后计算score
    # df_merged['Soundness'].fillna(df_merged['Soundness'].mean(), inplace=True)
    # df_merged['Presentation'].fillna(df_merged['Presentation'].mean(), inplace=True)
    # df_merged['Contribution'].fillna(df_merged['Contribution'].mean(), inplace=True)
    # df_merged['Confidence'].fillna(df_merged['Confidence'].mean(), inplace=True)
    # df_merged['Originality'].fillna(df_merged['Originality'].mean(), inplace=True)
    # df_merged['Quality'].fillna(df_merged['Quality'].mean(), inplace=True)
    # df_merged['Clarity'].fillna(df_merged['Clarity'].mean(), inplace=True)
    # df_merged['Significance'].fillna(df_merged['Significance'].mean(), inplace=True)
    # df_merged['Overall'].fillna(df_merged['Overall'].mean(), inplace=True)
    # df_merged['score'] = (df_merged['Soundness'] + df_merged['Presentation'] + df_merged['Contribution'] + df_merged['Overall'] + df_merged['Confidence'] + df_merged['Originality'] + df_merged['Quality'] + df_merged['Clarity'] + df_merged['Significance']) / 9.0
    print(df_merged[key].value_counts())

    # 遍历2,5,每次加0.1
    # for score in np.arange(2, 10, 0.1):
    for score in range(int(min(df_merged[key])),  int(max(df_merged[key])) + 1):
    # for score in range(int(min(df_merged[key])), int(max(df_merged[key])) + 1):
        # preds = (df_merged['Overall'].astype(int) >= score).astype(int)
        preds = (df_merged[key].astype(int) >= score).astype(int)
        labels = (df_merged['decision'] == 'Accept').astype(int)
        # f1, precision, recall, acc = compute_metrics(labels, preds)
        # print(f"Score: {score}, F1: {f1}, Precision: {precision}, Recall: {recall}, Acc: {acc}")

        accuracy, f1, roc, fpr, fnr = compute_metrics(labels, preds)
        print(f"Key:{key}, Score: {score}, F1: {f1}, Acc: {accuracy}, ROC: {roc}, FPR: {fpr}, FNR: {fnr}")