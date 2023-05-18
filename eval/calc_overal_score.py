import argparse
import json
import os
import time

import openai
import tqdm
#import ray

import shortuuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("-q", "--question-file")
    parser.add_argument("-r", "--review-file")
    parser.add_argument("-o", "--output-file")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()

    #ray.init()

    review_jsons = get_json_list(args.review_file)
    question_jsons = get_json_list(args.question_file)
    categories = {}
    for i, review in enumerate(review_jsons):
        #print(f"{i},{question_jsons[i]['category']},{review['score']}")
        if 'category' in question_jsons[i]:
            cat = question_jsons[i]['category']
        else:
            cat = 'generic'
        score = review['score'][1]/review['score'][0]*10
        if cat in categories:
            num, cur_score = categories[cat]
            categories[cat] = (num+1, cur_score + score)
        else:
            categories[cat] = (1, score )
    for key, value in categories.items():
        print(f"{key}: {round(value[1]*10/value[0],2)}%")


    with open(f"{args.output_file}", "w") as outfile:

        for key, value in categories.items():
            #print(f"{key}: {round(value[1]*10/value[0],2)}%")
            score_obj = {}
            score_obj['category'] = key
            score_obj['score'] = round(value[1]*10/value[0],2)
            json.dump(score_obj, outfile, ensure_ascii=False)
            outfile.write('\n')