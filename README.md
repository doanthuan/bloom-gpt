# Bloom-GPT
Instruction Tuning Large Language Model for Vietnamese

### Setup Enviroment

1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

1. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).

### Download Dataset

- `instruct_merged.jsonl`: instruction dataset. It contains 52k samples from Alpaca + 170k samples from GPT4All. Then translated to Vietnamese.

   ```bash
   wget https://storage.googleapis.com/doanthuan/data/instruct_merged.jsonl
   ```

- `translated_health_200k.jsonl`: Medical instruction dataset. It was collected from [ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor)

   ```bash
   wget https://storage.googleapis.com/doanthuan/data/translated_health_200k.jsonl
   ```


### Training

Finetune Bloomz-Chat:

```bash
python finetune_bloomz_instruct.py \
    --base_model 'bigscience/bloomz-7b1-mt' \
    --data_path 'instruct_merged.jsonl' \
    --output_dir './bloomz-instruct'
```

Finetune Bloomz-Doctor:

```bash
python finetune_bloomz_doctor.py \
    --base_model 'bigscience/bloomz-7b1-mt' \
    --data_path 'translated_health_200k.jsonl' \
    --output_dir './bloomz-doctor'
```

We can also tweak our hyperparameters:

```bash
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-alpaca' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length
```

### Inference (`demo`)

In demo directory contains notebooks for a demo.

### Model weights
