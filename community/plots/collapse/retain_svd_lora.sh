#!/bin/bash
# Does the retain-side SVD do better on the elicited (LoRA-attached) model or on the
# plain one? Llama-3.1-8B / Sycophancy, retain / both / quadratic, LoRA elicitation ON.
#   retain_svd_with_lora=false: retain SVD before the LoRA elicitation (the current code path)
#   retain_svd_with_lora=true:  retain SVD after it, LoRA active in the forward (as for forget)
# Two LoRA LRs from the good region of the Llama/Sycophancy Optuna search
# (sycophancy/results/Llama-3.1-8B.json: top-1 0.041; 0.005-0.05 all in the top 10, >=0.1 bad).
# Step size via RepSelectAdaptive (bisected onto the KL budget), like the collapse grids.
# Reference without LoRA: collapse2_sycophancy_Llama-3.1-8B_retain_both_quadratic (18.3%).

source .venv/bin/activate
run() {
  bash verda_runner.sh $*
}

model=${1:-Llama-3.1-8B}
common="python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/sycophancy/default model=${model} trainer=RepSelectSimple trainer.handler=RepSelectAdaptive ~eval.general_caps ~eval.fewshot_attack_5 trainer.method_args.distribution=retain trainer.method_args.collapse_on=both trainer.method_args.hard_soft=quadratic trainer.method_args.n_pcs=512 trainer.method_args.use_lora=true"

for lora_lr in 0.04 0.01; do
  for with_lora in false true; do
    run ${common} \
      trainer.method_args.lora_lr=${lora_lr} \
      trainer.method_args.retain_svd_with_lora=${with_lora} \
      task_name=retainsvd_sycophancy_${model}_loralr${lora_lr}_svdlora${with_lora}
  done
done
