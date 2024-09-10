import sys

sys.path.append("../")
sys.path.append('/Users/huxiang/Documents/GitHub/Research/ResearchAgent/')
from ai_scientist.perform_review import (
    load_paper,
    perform_review,
    reviewer_system_prompt_neg,
    neurips_form,
)
import pathlib
import pandas as pd
import numpy as np
import requests
import argparse
import os
import time
import multiprocessing as mp
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from iclr_analysis import prep_open_review_data, parse_arguments

from utils_tool import save_json_data_to_file, load_json_from_file, init_logging

from agent.extract_problem_method_agent import ExtractProblemMethodFromTitleAbstractAgent


class Args2:
    model = "openai"


init_logging(log_dir="log", log_filename="extract_problem_method_from_txt.log")

args2 = Args2()
agent = ExtractProblemMethodFromTitleAbstractAgent(args2)



def open_review_validate(
    num_reviews,
    model,
    rating_fname,
    batch_size,
    num_reflections,
    num_fs_examples,
    num_reviews_ensemble,
    temperature,
    reviewer_system_prompt,
    review_instruction_form,
    num_paper_pages=None,
    data_seed=1,
    balanced_val=False,
):
    print("num_reviews:", num_reviews)
    ore_ratings = prep_open_review_data(
        data_seed=data_seed,
        balanced_val=balanced_val,
        num_reviews=num_reviews,
    )

    print(ore_ratings)
    print("len(ore_ratings):", len(ore_ratings))

    # 遍历pd:DataFrame:ore_ratings中的每个元素
    iclr_parsed_dir = '/Users/huxiang/Documents/GitHub/Research/AI-Scientist/review_iclr_bench/iclr_parsed/'
    iclr_idea_extract_dir = "/Users/huxiang/Documents/GitHub/Research/AI-Scientist/review_iclr_bench/iclr_idea_extract/"
    i = 0
    for index, row in ore_ratings.iterrows():
        print("index:", index, 'i:', i)
        print(row['paper_id'])
        paper_txt_file = os.path.join(iclr_parsed_dir, f"{row['paper_id']}.txt")
        idea_extract_file = os.path.join(iclr_idea_extract_dir, f"{row['paper_id']}.json")

        with open(paper_txt_file, 'r') as fr:
            paper_txt = fr.readlines()
            j = 0
            for line in paper_txt:
                line = line.strip().replace(" ", "").lower()
                if "introduction" in line:
                    break
                j += 1
            paper_txt = paper_txt[:j]
            title = paper_txt[0].strip() + paper_txt[1].strip()
            abstract = ''.join(paper_txt[2:]).split("1 INTRODUCTION")[0].strip().split("INTRODUCTION")[0].split("ABSTRACT")[-1]
            title = title.replace('\n', '').replace("#", "")
            abstract = abstract.replace('\n', '')[:2500]
            print("title:", title)
            print("abstract:", abstract)
            print('='*100)
            print('='*100)
            print('='*100)
            data = {
                "title": title,
                "abstract": abstract,
                "paper_id": row['paper_id']
            }
            save_json_data_to_file(data, idea_extract_file)
            problem_method_extract_file = os.path.join(iclr_idea_extract_dir,
                                                       f"{row['paper_id']}_task_method_experiment_conclusion.json")
            if not os.path.exists(problem_method_extract_file):
                result = agent.run_v2(
                    title, abstract
                )
                result.update(data)
                # result_data = result.to_dict()
                save_json_data_to_file(result, problem_method_extract_file)

        i += 1
        if i > 500:
            break




if __name__ == '__main__':
    args = parse_arguments()
    # Create client - float temp as string
    temperature = str(args.temperature).replace(".", "_")
    rating_fname = f"llm_reviews/{args.model}_temp_{temperature}"
    pathlib.Path("llm_reviews/").mkdir(parents=True, exist_ok=True)

    if args.num_fs_examples > 0:
        rating_fname += f"_fewshot_{args.num_fs_examples}"

    if args.num_reflections > 1:
        rating_fname += f"_reflect_{args.num_reflections}"

    if args.num_reviews_ensemble > 1:
        rating_fname += f"_ensemble_{args.num_reviews_ensemble}"

    num_paper_pages = None if args.num_paper_pages == 0 else args.num_paper_pages
    if num_paper_pages is not None:
        rating_fname += f"_pages_{num_paper_pages}"
    else:
        rating_fname += "_pages_all"

    # Settings for reviewer prompt
    reviewer_system_prompt = reviewer_system_prompt_neg
    reviewer_form_prompt = neurips_form
    rating_fname += ".csv"
    open_review_validate(
        args.num_reviews,
        args.model,
        rating_fname,
        args.batch_size,
        args.num_reflections,
        args.num_fs_examples,
        args.num_reviews_ensemble,
        args.temperature,
        reviewer_system_prompt,
        reviewer_form_prompt,
        num_paper_pages,
        balanced_val=False,
    )
