import os
import pandas as pd
import sys
from openai import AzureOpenAI
import json


def load_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

sys.path.append("..")
from ai_scientist.perform_review_with_problem_and_method import (
    load_paper,
    perform_review,
    reviewer_system_prompt_neg,
    neurips_form,
)
import os
import openai



# client = openai.Client()
# model = "gpt-4o-2024-05-13"

client = AzureOpenAI(
    azure_endpoint="https://westlakeaustraliaeast.openai.azure.com/",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-15-preview"
)

model = "gpt-4o"
# model = "gpt-35-turbo-16k"

num_reflections = 5
num_fs_examples = 1
num_reviews_ensemble = 5
temperature = 0.1
reviewer_system_prompt = reviewer_system_prompt_neg
review_instruction_form = neurips_form


# pdf_path = "/Users/huxiang/Documents/Paper/ResearchAgent/ResearchAgent_Formatting_Instructions_for_ICLR_2025_Conference_Submissions.pdf"
# paper_name = os.path.basename(pdf_path).split('.')[0]
# txt_path = f"user_data/ai_scientist_reviews/{paper_name}.txt"


paper_name = "N0uJGWDw21d"

idea_dir = '/Users/huxiang/Documents/GitHub/Research/AI-Scientist/review_iclr_bench/iclr_idea_extract/'
problem_method_file = os.path.join(idea_dir, f'{paper_name}_problem_method.json')
problem_method = load_json_file(problem_method_file)

text = f"Problem:{problem_method['problem']}\nMethod:{problem_method['method']}"

print(text)

rating_fname = f"user_data/ai_scientist_reviews/{paper_name}_rating.csv"

# os.makedirs(os.path.dirname(txt_path), exist_ok=True)
# if not os.path.exists(txt_path):
#     text = load_paper(pdf_path, num_pages=None)
#     with open(txt_path, "w") as f:
#         f.write(text)
#     print(f"Generated txt file for {pdf_path}")
# else:
#     with open(txt_path, "r") as f:
#         text = f.read()

# os.makedirs(rating_fname, exist_ok=True)

llm_cols = [
    "paper_id",
    "Summary",
    "Soundness",
    "Contribution",
    "Overall",
    "Confidence",
    "Strengths",
    "Originality",
    "Quality",
    "Significance",
    "Decision",
]


# print(f"text:{text}")
print(f"reviewer_system_prompt:{reviewer_system_prompt}")
print(f"review_instruction_form:{review_instruction_form}")

llm_ratings = pd.DataFrame(columns=llm_cols)
llm_ratings.set_index("paper_id", inplace=True)


class Args:
    use_only_problem_and_method_for_review = 1

args = Args()
# import pdb;pdb.set_trace()
review = perform_review(
    text,
    model,
    client,
    num_reflections,
    num_fs_examples,
    num_reviews_ensemble,
    temperature,
    reviewer_system_prompt=reviewer_system_prompt,
    review_instruction_form=review_instruction_form,
    args=args
)
correct_review = sum([k in review for k in llm_cols[1:]]) == len(
    llm_cols[1:]
)
if correct_review:
    # Add the reviews to the rankings dataframe as a new row
    llm_ratings.loc[paper_name] = review
    llm_ratings.to_csv(rating_fname)
    print(f"Generated review file for {problem_method_file}")
    print(f"Decision: {review['Decision']}, Score: {review['Overall']}")
else:
    print(f"Review for {problem_method_file} was incorrect")