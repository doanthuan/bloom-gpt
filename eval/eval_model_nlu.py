import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import ray
from peft import PeftModel
from utils import evaluate, disable_torch_init, make_prompt_instruct, make_prompt_medical


def run_eval(base_model, lora_path, question_file, answer_file, prompt_type, num_gpus):
    # split question file into num_gpus files
    ques_jsons = []
    with open(os.path.expanduser(question_file), "r", encoding="utf-8") as ques_file:
        for line in ques_file:
            ques_jsons.append(line)

    chunk_size = len(ques_jsons) // num_gpus
    ans_handles = []
    for i in range(0, len(ques_jsons), chunk_size):
        ans_handles.append(
            # get_model_answers.remote(
            #     base_model, lora_path, ques_jsons[i : i + chunk_size], prompt_type
            # )
            get_model_answers(base_model, lora_path, ques_jsons[i : i + chunk_size], prompt_type)
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        #ans_jsons.extend(ray.get(ans_handle))
        ans_jsons.extend(ans_handle)

    with open(os.path.expanduser(answer_file), "w", encoding="utf-8") as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line, ensure_ascii=False) + "\n")


#@ray.remote(num_gpus=1)
#@torch.inference_mode()
def get_model_answers(base_model, lora_path, question_jsons, prompt_type):
    #disable_torch_init()

    if lora_path:
        lora_path = os.path.expanduser(lora_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map = {'': 0},
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path, device_map={'': 0})

    ans_jsons = []
    for i, line in enumerate(tqdm(question_jsons)):
        ques_json = json.loads(line)
        idx = ques_json["question_id"]
        qs = ques_json["text"]
        outputs = evaluate(qs, tokenizer, model, prompt_type)
        ans_id = shortuuid.uuid()
        ans_jsons.append(
            {
                "question_id": idx,
                "text": outputs,
                "answer_id": ans_id,
                "model_id": base_model,
                "metadata": {},
            }
        )
    return ans_jsons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    parser.add_argument("--prompt-type", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()

    #ray.init()
    run_eval(
        args.base_model,
        args.lora_path,
        args.question_file,
        args.answer_file,
        args.prompt_type,
        args.num_gpus,
    )
