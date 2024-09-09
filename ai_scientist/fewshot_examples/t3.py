import json

with open("attention.json", "r") as fr:
    data = json.load(fr)
    print(data.keys())

    d = json.loads(data['review'])

    print(json.dumps(d, indent=4, ensure_ascii=False))


if 1:

    s = """
{
    "Summary": "The paper proposes the Transformer, a novel neural network architecture that relies entirely on self-attention mechanisms, eschewing traditional recurrent and convolutional layers. This innovation allows the model to achieve state-of-the-art results in machine translation tasks with significant improvements in both training efficiency and translation quality.",
    "Soundness": 4,
    "Contribution": 4,
    "Overall": 8,
    "Confidence": 5,
    "Strengths": [
        "The Transformer model introduces a highly innovative use of self-attention mechanisms, replacing traditional recurrent and convolutional layers.",
    ],
    "Originality": 4,
    "Quality": 4,
    "Significance": 4,
    "Decision": "Accept"
}
    
"""
    f1 = '/Users/huxiang/Documents/GitHub/Research/AI-Scientist/ai_scientist/fewshot_examples/attention_problem_method_review.json'

    with open(f1, 'w') as fw:
        data = {
            "review": json.dumps(s)
        }

        json.dump(data, fw)