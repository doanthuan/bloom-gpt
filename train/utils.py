
import json
from typing import Dict
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def tokenize(prompt, tokenizer, cutoff_len=256, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point, tokenizer):
    full_prompt = generate_prompt(data_point)
    return tokenize(full_prompt, tokenizer)

def make_prompt(instruction):
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Response:"""

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501


# class SupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(self, data_path: str,
#                  tokenizer: transformers.PreTrainedTokenizer):
#         super(SupervisedDataset, self).__init__()
#         print("Loading data...")
#         list_data_dict = json.load(open(data_path, "r"))

#         print("Formatting inputs...")
#         sources = [example["conversations"] for example in list_data_dict]
#         data_dict = preprocess(sources, tokenizer)

#         self.input_ids = data_dict["input_ids"]
#         self.labels = data_dict["labels"]
#         self.attention_mask = data_dict["attention_mask"]

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         return dict(input_ids=self.input_ids[i],
#                     labels=self.labels[i],
#                     attention_mask=self.attention_mask[i])
    
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        print("Loading data...")
        with open(data_path, 'r') as in_file:
            list_data_dict = [json.loads(line) for line in in_file.readlines()]

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.list_data_dict[i]
        return generate_and_tokenize_prompt(item, self.tokenizer)
    

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_path) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # dataset_cls = (LazySupervisedDataset
    #                if data_args.lazy_preprocess else SupervisedDataset)
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_path)
    return dict(train_dataset=train_dataset,
                eval_dataset=None)