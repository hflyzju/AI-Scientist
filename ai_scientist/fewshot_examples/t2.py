import json

with open("132_automated_relational.json", "r") as fr:
    data = json.load(fr)
    print(data.keys())

    d = json.loads(data['review'])

    print(json.dumps(d, indent=4, ensure_ascii=False))


if 1:

    s = """
{
    "Summary": "The paper provides an interesting direction in the meta-learning field. In particular, it proposes to enhance meta learning performance by fully exploring relations across multiple tasks. To capture such information, the authors develop a heterogeneity-aware meta-learning framework by introducing a novel architecture--meta-knowledge graph, which can dynamically find the most relevant structure for new tasks.",
    "Strengths": [
        "The paper takes one of the most important issues of meta-learning: task heterogeneity. For me, the problem itself is real and practical.",
        "The proposed meta-knowledge graph is novel for capturing the relation between tasks and addressing the problem of task heterogeneity. Graph structure provides a more flexible way of modeling relations. The design for using the prototype-based relational graph to query the meta-knowledge graph is reasonable and interesting.",
    ],
    "Originality": 3,
    "Quality": 3,
    "Significance": 4,
    "Soundness": 3,
    "Contribution": 3,
    "Overall": 7,
    "Confidence": 5,
    "Decision": "Accept"
}
    
"""
    f1 = '/Users/huxiang/Documents/GitHub/Research/AI-Scientist/ai_scientist/fewshot_examples/132_automated_relational_problem_method_review.json'

    with open(f1, 'w') as fw:
        data = {
            "review": json.dumps(s)
        }

        json.dump(data, fw)