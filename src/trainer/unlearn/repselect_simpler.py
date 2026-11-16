"""Simplified RepSelectSimple, for reference (not registered, no config).

Fixed choices: SVD of the retain gradient, collapse on both sides, quadratic
collapse, no LoRA elicitation. Equals RepSelectSimple with distribution=retain,
collapse_on=both, hard_soft=quadratic, use_lora=false.
"""

import torch as pt

from trainer.unlearn.base import UnlearnTrainer


def _collapse(mat: pt.Tensor, eig_vec: pt.Tensor, S: pt.Tensor) -> pt.Tensor:
    """Divide mat's components along the eig_vec columns by S^2, normalized so
    that the smallest one maps to 1; the rest of mat is left as is.
    The dim arguments keep it working on the batched (per-expert) MoE weights."""
    S = S**2
    S = S / S.amin(dim=-1, keepdim=True)
    projected = mat @ eig_vec
    return mat - (projected - projected / S.unsqueeze(-2)) @ eig_vec.mT


class RepSelectSimpler(UnlearnTrainer):
    """Single-shot unlearning update of the MLP gate/up/down projections:
    1. Accumulate the retain weight-gradient over one pass and SVD it.
    2. Accumulate the forget weight-gradient over one pass.
    3. Collapse the forget gradient along the top retain PCs, on D_in (via V)
       and on D_out (via U).
    4. Each training epoch: weight -= filtered_grad * lr, then evaluate.
    """

    def __init__(self, n_pcs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_pcs = n_pcs
        layers = self.model.model.layers
        if any(hasattr(layer.mlp, "experts") for layer in layers):  # MoE
            # note that here the act and grad dimensions may be transposed
            self.base_trainable_params = [
                layer.mlp.experts.gate_up_proj
                for layer in layers
                if hasattr(layer.mlp, "experts")
            ]
        else:
            self.base_trainable_params = [
                module.weight
                for layer in layers
                for module in [
                    layer.mlp.gate_proj,
                    layer.mlp.up_proj,
                    layer.mlp.down_proj,
                ]
            ]

    def _accumulate_grad(self, split):
        """One pass over `split` (forget | retain); leaves the summed gradient of
        minus the loss in the .grad of the trainable params."""
        self.model.zero_grad(set_to_none=True)
        for batch_pair in self.get_train_dataloader():
            batch = batch_pair[split]
            batch = {k: batch[k] for k in ("input_ids", "attention_mask", "labels")}
            (-self.model(**batch).loss).backward()

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        self.model = self.accelerator.prepare(self.model)
        self.control = self.callback_handler.on_train_begin(
            self.args, self.state, self.control
        )
        self.model.train()
        self.model.requires_grad_(False)
        for weight in self.base_trainable_params:
            weight.requires_grad_(True)

        self._accumulate_grad("retain")
        USVs = [
            pt.svd_lowrank(weight.grad.float(), q=self.n_pcs)
            for weight in self.base_trainable_params
        ]

        self._accumulate_grad("forget")
        for weight, (U, S, V) in zip(self.base_trainable_params, USVs):
            grad = weight.grad.float()
            grad = _collapse(grad, V, S)  # D_in side
            grad = _collapse(grad.mT, U, S).mT  # D_out side
            weight.filtered_grad = grad.to(weight.dtype)
            weight.grad = None

        self._apply_unlearn_loop()

        self.control = self.callback_handler.on_train_end(
            self.args, self.state, self.control
        )

    def _apply_unlearn_loop(self):
        # perform dummy epochs, simply applying the filtered gradient
        self.evaluate()
        for epoch in range(self.args.num_train_epochs):
            for weight in self.base_trainable_params:
                weight.data -= weight.filtered_grad * self.args.learning_rate

            self.state.epoch = epoch + 1
            self.evaluate()
            if self.control.should_training_stop:
                break
