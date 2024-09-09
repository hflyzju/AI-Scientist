import json

with open("2_carpe_diem.json", "r") as fr:
    data = json.load(fr)
    print(data.keys())

    d = json.loads(data['review'])

    print(json.dumps(d, indent=4, ensure_ascii=False))


if 1:

    s = """
{
    "Summary": "This paper proposes Recency Bias, an adaptive mini batch selection method for training deep neural networks. To select informative minibatches for training, the proposed method maintains a fixed size sliding window of past model predictions for each data sample. At a given iteration, samples which have highly inconsistent predictions within the sliding window are added to the minibatch. The main contribution of this paper is the introduction of a sliding window to remember past model predictions",
    "Strengths": [
        "The idea of using a sliding window over a growing window in active batch selection is interesting.",
    ],
    "Originality": 3,
    "Quality": 2,
    "Significance": 2,
    "Soundness": 2,
    "Contribution": 2,
    "Overall": 4,
    "Confidence": 3,
    "Decision": "Reject"
}
    
"""
    f1 = '/Users/huxiang/Documents/GitHub/Research/AI-Scientist/ai_scientist/fewshot_examples/2_carpe_diem_problem_method_review.json'

    with open(f1, 'w') as fw:
        data = {
            "review": json.dumps(s)
        }

        json.dump(data, fw)