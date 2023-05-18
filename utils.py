import torch

def make_prompt_instruct(instruction):
        return f"""Hãy viết một phản hồi thích hợp cho chỉ dẫn dưới đây.

### Instruction:
{instruction}

### Response:"""

def make_prompt_medical(instruction):
        return f"""Nếu bạn là bác sĩ, vui lòng trả lời các câu hỏi y tế dựa trên mô tả của bệnh nhân.

### Instruction:
{instruction}

### Response:"""

def evaluate(q, tokenizer, model, prompt_type = 1):
    prompt = make_prompt_instruct(q) if prompt_type == 1 else make_prompt_medical(q)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to('cuda')
    with torch.no_grad():
        gen_tokens = model.generate(
            input_ids=input_ids,
            max_new_tokens = 512,
            do_sample=True,
            temperature=0.5,
            top_k=20,
            repetition_penalty=1.2,
            #eos_token_id=0, # for open-end generation.
            #pad_token_id=tokenizer.eos_token_id,
        )
    origin_output = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
    output = origin_output.split("### Response:")[1].strip()
    return output


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)