#download instruction data
wget https://storage.googleapis.com/doanthuan/data/instruct_merged.jsonl

#download medical data
wget https://storage.googleapis.com/doanthuan/data/translated_health_200k.jsonl

# train bloomz instruct
WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1234 finetune_bloomz_instruct.py

# train bloomz doctor
WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1234 finetune_bloomz_doctor.py