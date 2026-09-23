# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default relearning_trainer=RepSelectAttack task_name=SAMPLE_ATTACK
import logging

import torch as pt
from peft import LoraConfig, get_peft_model

from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.repselect_simple import _collapse, _prep_batch, _train_on

logging.basicConfig(level=logging.INFO)


class RepSelectAttack(UnlearnTrainer):
    """
    Adaptive relearning attack: assumes the attacker knows the RepSelectSimple
    mechanism and mirrors it against the attacker's own relearning set instead of
    the (unknown) forget set.
    1. Optional LoRA adversarial pretrain, then accumulate one full-pass
       weight-gradient over the relearning set (descent direction), over the same
       MLP gate/up/down projections RepSelect targets.
    2. SVD that accumulated gradient once, exactly like RepSelect does.
    3. Standard per-batch SGD over the relearning set: each batch's fresh gradient
       is passed through RepSelect's own `_collapse` transform (using the fixed
       SVD basis from step 2) before being applied to the weights, concentrating
       every update on the same subspace RepSelect's own filtered_grad lives in.
    """

    def __init__(
        self,
        n_pcs,
        lora_lr,
        collapse_on="both",
        use_lora=True,
        hard_soft="soft",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_pcs = n_pcs
        self.lora_lr = lora_lr
        self.collapse_on = collapse_on
        self.use_lora = use_lora
        self.hard_soft = hard_soft
        assert collapse_on in ["act", "grad", "both", "none"]
        assert hard_soft in ["hard", "soft"]

        is_moe = any(hasattr(layer.mlp, "experts") for layer in self.model.model.layers)
        if is_moe:
            lora_config = LoraConfig(target_parameters=["mlp.experts.gate_up_proj"])
            self.model = get_peft_model(self.model, lora_config)
            self.base_trainable_params = [
                layer.mlp.experts.base_layer.gate_up_proj
                for layer in self.model.base_model.model.model.layers
                if hasattr(layer.mlp, "experts")
            ]
        else:
            lora_config = LoraConfig(
                target_modules=["gate_proj", "up_proj", "down_proj"]
            )
            self.model = get_peft_model(self.model, lora_config)
            self.base_trainable_params = [
                module.base_layer.weight
                for layer in self.model.base_model.model.model.layers
                for module in [
                    layer.mlp.gate_proj,
                    layer.mlp.up_proj,
                    layer.mlp.down_proj,
                ]
            ]

        self.lora_params = [p for n, p in self.model.named_parameters() if "lora_" in n]

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        self.model = self.accelerator.prepare(self.model)
        self.control = self.callback_handler.on_train_begin(
            self.args, self.state, self.control
        )
        self.model.train()

        # LoRA adversarial pretrain: one epoch, SGD descent on relearn NLL
        if self.use_lora:
            _train_on(self.lora_params, self.model)
            for batch in self.get_train_dataloader():
                self.model.zero_grad(set_to_none=True)
                output = self.model(**_prep_batch(batch))
                output.loss.backward()
                for p in self.lora_params:
                    p.data -= self.lora_lr * p.grad

        # one full pass: accumulate relearn weight-gradient with LoRA active
        self.model.zero_grad(set_to_none=True)
        _train_on(self.base_trainable_params, self.model)
        for batch in self.get_train_dataloader():
            output = self.model(**_prep_batch(batch))
            output.loss.backward()

        # strip LoRA and release its freed memory before the SVD's own working memory spike
        self.model = self.model.unload()
        pt.cuda.empty_cache()

        # SVD of the accumulated relearn gradient -- fixed basis reused every batch below
        for weight in self.base_trainable_params:
            weight.USV = pt.svd_lowrank(weight.grad.float(), q=self.n_pcs)
            weight.grad = None

        self._apply_relearn_loop()

        self.control = self.callback_handler.on_train_end(
            self.args, self.state, self.control
        )

    def _apply_relearn_loop(self):
        self.evaluate()
        for epoch in range(self.args.num_train_epochs):
            for batch in self.get_train_dataloader():
                self.model.zero_grad(set_to_none=True)
                output = self.model(**_prep_batch(batch))
                output.loss.backward()

                for weight in self.base_trainable_params:
                    grad = weight.grad.float()
                    U, S, V = weight.USV
                    if self.collapse_on in ["act", "both"]:
                        grad = _collapse(grad, V, S, self.hard_soft)  # filter D_in side
                    if self.collapse_on in ["grad", "both"]:
                        grad = _collapse(grad.mT, U, S, self.hard_soft).mT  # filter D_out side
                    # sign update: like raw SGD but scale-invariant per parameter (the
                    # Adam/signSGD connection), since raw grad magnitudes are ~1e-4 to
                    # 1e-6 here and a plain SGD step at this lr barely moves the weights
                    weight.data -= self.args.learning_rate * grad.sign().to(weight.dtype)
                    weight.grad = None

            self.state.epoch = epoch + 1
            self.evaluate()
            if self.control.should_training_stop:
                break
