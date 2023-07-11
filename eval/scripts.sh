# get chatgpt base score
python -m eval.qa_baseline_gpt35 -q eval/data/question_instruct.jsonl -o eval/data/answer_gpt35_instruct.jsonl
python -m eval.qa_baseline_gpt35 -q eval/data/question_doctor.jsonl -o eval/data/answer_gpt35_doctor.jsonl

#get model answer
python -m eval.get_model_answer --base-model bigscience/bloomz-7b1-mt --lora-path models/bloomz-instruct --question-file eval/data/question_instruct.jsonl --answer-file eval/data/answer_bloomz_instruct.jsonl
python -m eval.get_model_answer --base-model bigscience/bloomz-7b1-mt --lora-path models/bloomz-doctor --question-file eval/data/question_doctor.jsonl --answer-file eval/data/answer_bloomz_doctor.jsonl --prompt-type 2
python -m eval.get_model_answer --base-model bigscience/bloomz-7b1-mt --question-file eval/data/question_instruct.jsonl --answer-file eval/data/answer_bloomz_org_instruct.jsonl
python -m eval.get_model_answer --base-model bigscience/bloomz-7b1-mt --question-file eval/data/question_instruct.jsonl --answer-file eval/data/answer_bloomz_org_doctor.jsonl --prompt-type 2

#eval gpt review
python -m eval.eval_gpt_review -q eval/data/question_instruct.jsonl -a eval/data/answer_gpt35_instruct.jsonl eval/data/answer_bloomz_instruct.jsonl -p eval/data/prompt.jsonl -r eval/data/reviewer.jsonl -o eval/data/review_bloomz_instruct.jsonl
python -m eval.eval_gpt_review -q eval/data/question_doctor.jsonl -a eval/data/answer_gpt35_doctor.jsonl eval/data/answer_bloomz_doctor.jsonl -p eval/data/prompt.jsonl -r eval/data/reviewer.jsonl -o eval/data/review_bloomz_doctor.jsonl
python -m eval.eval_gpt_review -q eval/data/question_instruct.jsonl -a eval/data/answer_gpt35_instruct.jsonl eval/data/answer_bloomz_org_instruct.jsonl -p eval/data/prompt.jsonl -r eval/data/reviewer.jsonl -o eval/data/review_bloomz_org_instruct.jsonl

#calc overal score
python -m eval.calc_overal_score -q eval/data/question_instruct.jsonl -r eval/data/review_bloomz_instruct.jsonl -o eval/data/score_bloomz_instruct.jsonl
python -m eval.calc_overal_score -q eval/data/question_doctor.jsonl -r eval/data/review_bloomz_doctor.jsonl -o eval/data/score_bloomz_doctor.jsonl
python -m eval.calc_overal_score -q eval/data/question_instruct.jsonl -r eval/data/review_bloomz_org_instruct.jsonl -o eval/data/score_bloomz_org_instruct.jsonl

#format and translate icliniq
python -m eval.convert_icliniq -i eval/data/icliniq.json -o eval/data/question_doctor.jsonl