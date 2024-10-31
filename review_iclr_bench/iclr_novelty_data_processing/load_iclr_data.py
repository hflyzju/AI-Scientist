import json

with open("review_dataset_ICLR2022_2023.json", 'r') as fr:
    for l in fr:
        data = json.loads(l.strip())
        print(l)
        break