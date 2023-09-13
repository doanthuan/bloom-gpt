import argparse
import json
import os
import logging
import random

from googletrans import Translator
translator = Translator()

def translate_google(value):
    response = None
    while response is None:
        try:
            response = translator.translate(value, dest='vi', src='en')
        except Exception as e:
            print(e)
    print(response.text)
    return response.text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file")
    parser.add_argument("-o", "--output-file")
    args = parser.parse_args()
 
    # Opening JSON file
    with open(args.input_file) as json_file:
        data = json.load(json_file)
    n = len(data)
    random_ls = []
    examples = []
    while len(random_ls) < 100:
        idx = random.randint(0, n)
        if idx not in random_ls:
            random_ls.append(idx)
            examples.append(data[idx]['input'])
    

    with open(args.output_file, 'w', encoding="utf-8") as outfile:
        for i, ex in enumerate(examples):
            item = {}
            item['question_id'] = i
            item['text'] = translate_google(ex)
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')
