# Bloom-GPT
Instruction Tuning Large Language Model for Vietnamese


The general architecture and experimental results of our method can be found in our [paper](https://arxiv.org/abs/2309.04646):

    @InProceedings{,
    title     = "{Efficient Finetuning Large Language Models For Vietnamese Chatbot}",
    author    = {Vu-Thuan Doan, Quoc-Truong Truong, Duc-Vu Nguyen, Vinh-Tiep Nguyen and Thuy-Ngan Nguyen Luu},
    booktitle = {In Proceedings of International Conference on Multimedia Analysis and Pattern Recognition (MAPR)},
    year      = {2023},
    pages     = "{to appear}"
    }

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
python finetune_bloomz_instruct.py \
    --base_model 'bigscience/bloomz-7b1-mt' \
    --data_path 'instruct_merged.jsonl' \
    --output_dir './bloomz-instruct' \
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

[Bloom-Chat demo](https://colab.research.google.com/drive/1MWQsvbanEwt6z4BLFq_BkYdM1WC6pEwA?usp=sharing)

[Bloom-Doctor demo](https://colab.research.google.com/drive/1kiqlFQToWO40L4lGqM8i4UQsHZIJzaPt?usp=sharing)


In demo directory also contains notebooks for a demo.

### Model weights

LoRA weights are in models directory