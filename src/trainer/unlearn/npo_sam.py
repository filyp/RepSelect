"""NPO + sharpness-aware minimization (Fan et al., ICML 2025,
"Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware
Minimization Perspective and Beyond", github.com/OPTML-Group/Unlearn-Smooth).

Per training step:
  1. gradient g of the NPO forget loss at the current weights theta
  2. perturb theta <- theta + rho * g / ||g||_2  (norm over all trainable params)
  3. forget gradient at the perturbed weights
  4. restore theta
  5. retain gradient at the original weights
  6. update with  gamma * grad_forget(theta + eps) + alpha * grad_retain(theta)

The NPO loss is minimised when the forget NLL is *high*, so ascending it (step 2)
moves the weights one small step toward re-learning the forget set. The unlearning
gradient is then taken from that point: a one-step adversarial inner maximisation,
the same idea as RepSelect's LoRA elicitation but re-done at every step in full
weight space. Their NPO loss (-2/beta * logsigmoid(beta * (nll - nll_ref))) is the
same as ours, so their beta and rho are on the same scale as this codebase's.

Extra cost versus NPO: one more forward/backward on the forget batch per step and
one parameter-sized buffer (the perturbation) held between steps 2 and 4.
"""

import torch

from trainer.unlearn.npo import NPO
from trainer.utils import compute_dpo_loss


class NPOSAM(NPO):
    def __init__(self, sam_rho=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam_rho = sam_rho

    def _forget_loss(self, model, inputs):
        forget_loss, _ = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=inputs["forget"],
            beta=self.beta,
        )
        return forget_loss

    def _retain_loss(self, model, inputs):
        retain_inputs = {
            "input_ids": inputs["retain"]["input_ids"],
            "attention_mask": inputs["retain"]["attention_mask"],
            "labels": inputs["retain"]["labels"],
        }
        return self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)

        # HF divides the loss by the number of accumulation steps in this case (our
        # nested forget/retain batches never carry a top-level "labels", so
        # num_items_in_batch is None and this is the branch NPO runs through too)
        accum = getattr(
            self, "current_gradient_accumulation_steps", self.args.gradient_accumulation_steps
        )
        scale = 1.0 / accum if (
            (not self.model_accepts_loss_kwargs or num_items_in_batch is None)
            and self.compute_loss_func is None
        ) else 1.0

        params = [p for p in model.parameters() if p.requires_grad]

        # 1. forget gradient at theta, kept out of p.grad (which may hold earlier micro-batches)
        with self.compute_loss_context_manager():
            forget_loss = self._forget_loss(model, inputs)
        eps = list(torch.autograd.grad(forget_loss, params, allow_unused=True))
        del forget_loss
        grad_norm = torch.norm(
            torch.stack([g.float().norm(2) for g in eps if g is not None]), 2
        )

        # 2. theta <- theta + rho * g / ||g||   (the buffer `eps` is scaled in place)
        with torch.no_grad():
            for i, (p, g) in enumerate(zip(params, eps)):
                if g is None:
                    continue
                g.mul_(self.sam_rho / (grad_norm + 1e-12))
                p.add_(g)

        # 3. forget gradient at the perturbed point, accumulated into p.grad
        with self.compute_loss_context_manager():
            forget_loss_p = self._forget_loss(model, inputs)
        self.accelerator.backward(self.gamma * forget_loss_p * scale)

        # 4. restore theta
        with torch.no_grad():
            for p, g in zip(params, eps):
                if g is not None:
                    p.sub_(g)
        del eps

        # 5. retain gradient at the original weights, accumulated into p.grad
        with self.compute_loss_context_manager():
            retain_loss = self._retain_loss(model, inputs)
        self.accelerator.backward(self.alpha * retain_loss * scale)

        loss = self.gamma * forget_loss_p.detach() + self.alpha * retain_loss.detach()
        return loss * scale
