"""RepSelectSimpler with per-token retain covariances in place of the SVD of the
summed retain gradient (not registered, no config).

Per module, over the retain tokens t (a_t: input activation, g_t: output gradient):
    unweighted:  cov_act = sum_t a_t a_t',          cov_grad = sum_t g_t g_t'
    weighted:    cov_act = sum_t |g_t|^2 a_t a_t',  cov_grad = sum_t |a_t|^2 g_t g_t'
The forget gradient G is then filtered with them, in one of two ways:
    inverse=False:  collapsed along the top n_pcs eigenvectors of each, dividing by
                    the eigenvalues (they take the place of S^2 of the gradient's SVD)
    inverse=True:   cov_grad^-1 G cov_act^-1, full rank, each cov with a ridge of
                    PRE_EPS x its mean diagonal (the init of als.solve)

Dense MLPs only, no LoRA elicitation.
"""

import torch as pt

from trainer.unlearn.base import UnlearnTrainer


PANEL_ROWS = 1024

# A cov (d, d) is symmetric, so only its upper triangle is computed and stored, as
# a list of row panels: the panel of rows r0:r1 holds the columns r0: (its
# diagonal block in full). With P panels this is ~(P + 1) / (2 P) of the square.


def _zero_cov(d, dtype, device) -> list[pt.Tensor]:
    return [
        pt.zeros(min(PANEL_ROWS, d - r0), d - r0, dtype=dtype, device=device)
        for r0 in range(0, d, PANEL_ROWS)
    ]


def _add_cov(panels: list[pt.Tensor], x: pt.Tensor):
    """panels += x' x of x (tokens, d). Each panel's product is fp32 (a plain bf16
    matmul returns bf16; out_dtype is CUDA only)."""
    for i, panel in enumerate(panels):
        r0 = i * PANEL_ROWS
        r1 = r0 + panel.shape[0]
        panel += pt.mm(x[:, r0:r1].mT, x[:, r0:], out_dtype=pt.float32)


def _full_cov(panels: list[pt.Tensor]) -> pt.Tensor:
    """the square fp32 cov (d, d): the lower triangle mirrored from the upper"""
    d = panels[0].shape[1]
    cov = pt.zeros(d, d, device=panels[0].device)
    for i, panel in enumerate(panels):
        r0 = i * PANEL_ROWS
        cov[r0 : r0 + panel.shape[0], r0:] = panel
    cov.triu_()
    cov += cov.triu(1).mT
    return cov


PRE_EPS = 1e-3  # ridge on the covs before inverting, x mean diagonal (as als.solve)


def _inv(panels: list[pt.Tensor]) -> pt.Tensor:
    """(cov + PRE_EPS x mean diagonal)^-1, fp32, via Cholesky in fp64 (as als.solve._inv)"""
    cov = _full_cov(panels).double()
    cov.diagonal().add_(PRE_EPS * cov.diagonal().mean())
    return pt.cholesky_inverse(pt.linalg.cholesky(cov)).float()


def _collapse(mat: pt.Tensor, eig_vec: pt.Tensor, eig_val: pt.Tensor) -> pt.Tensor:
    """Divide mat's components along the eig_vec columns by eig_val, normalized
    so that the smallest one maps to 1; the rest of mat is left as is."""
    eig_val = eig_val / eig_val.min()
    projected = mat @ eig_vec
    return mat - (projected - projected / eig_val) @ eig_vec.mT


class RepSelectCov(UnlearnTrainer):
    """Single-shot unlearning update of the MLP gate/up/down projections:
    1. One pass over the retain set: accumulate each module's act and grad
       covariances (inverse=False: then keep only their top n_pcs eigenpairs).
    2. One pass over the forget set: accumulate the weight gradient.
    3. Filter the forget gradient on D_in (cov_act) and on D_out (cov_grad):
       collapse it along the top eigenvectors, or multiply by the inverses.
    4. Each training epoch: weight -= filtered_grad * lr, then evaluate.
    """

    def __init__(self, n_pcs, weighted, inverse, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_pcs = n_pcs  # unused if inverse
        self.weighted = weighted
        self.inverse = inverse
        # Of the accumulators; each batch's cov is fp32. The inverse needs fp32: the
        # rounding of a bf16 accumulator (~1%) is above the ridge, and the Cholesky
        # then fails on a cov that is no longer positive definite.
        self.cov_dtype = pt.float32 if inverse else pt.bfloat16
        layers = self.model.model.layers
        assert not any(hasattr(layer.mlp, "experts") for layer in layers), "no MoE"
        self.modules = [
            module
            for layer in layers
            for module in [layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj]
        ]
        self.base_trainable_params = [module.weight for module in self.modules]

    def _add_batch_cov(self, weight, act, grad):
        act, grad = act[self._mask], grad[self._mask]  # (tokens, D_in), (tokens, D_out)
        if self.weighted:
            act_norm = act.float().norm(dim=1, keepdim=True)
            grad_norm = grad.float().norm(dim=1, keepdim=True)
            act = (act * grad_norm).to(act.dtype)
            grad = (grad * act_norm).to(grad.dtype)
        _add_cov(weight.cov_act, act)
        _add_cov(weight.cov_grad, grad)

    def _accumulate_covs(self):
        """One pass over the retain set, accumulating weight.cov_act (D_in, D_in)
        and weight.cov_grad (D_out, D_out) over its real tokens, as row panels.
        The forward hook holds the module's input until its output gradient
        arrives in the backward, which consumes both."""

        def forward_hook(module, inputs, output):
            # no parameter requires grad, so this starts the graph at the lowest module
            output.requires_grad_(True)
            act = inputs[0].detach()
            output.register_hook(
                lambda grad: self._add_batch_cov(module.weight, act, grad)
            )

        self.model.requires_grad_(False)
        for weight in self.base_trainable_params:
            D_out, D_in = weight.shape
            kw = {"dtype": self.cov_dtype, "device": weight.device}
            weight.cov_act = _zero_cov(D_in, **kw)
            weight.cov_grad = _zero_cov(D_out, **kw)
        handles = [m.register_forward_hook(forward_hook) for m in self.modules]
        for batch_pair in self.get_train_dataloader():
            batch = batch_pair["retain"]
            batch = {k: batch[k] for k in ("input_ids", "attention_mask", "labels")}
            self._mask = batch["attention_mask"].bool()
            self.model(**batch).loss.backward()
        for handle in handles:
            handle.remove()

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        self.model = self.accelerator.prepare(self.model)
        self.control = self.callback_handler.on_train_begin(
            self.args, self.state, self.control
        )
        self.model.train()

        self._accumulate_covs()
        for weight in self.base_trainable_params:
            if self.inverse:
                break  # the covs are kept until they are inverted below
            # top eigenpairs; each cov is freed as soon as it is decomposed
            weight.V, weight.S_act, _ = pt.svd_lowrank(
                _full_cov(weight.cov_act), q=self.n_pcs
            )
            del weight.cov_act
            weight.U, weight.S_grad, _ = pt.svd_lowrank(
                _full_cov(weight.cov_grad), q=self.n_pcs
            )
            del weight.cov_grad

        # one pass over the forget set: accumulate the weight gradient in .grad
        for weight in self.base_trainable_params:
            weight.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        for batch_pair in self.get_train_dataloader():
            batch = batch_pair["forget"]
            batch = {k: batch[k] for k in ("input_ids", "attention_mask", "labels")}
            (-self.model(**batch).loss).backward()

        grad_sq, filtered_sq = 0, 0
        for weight in self.base_trainable_params:
            grad = weight.grad.float()
            grad_sq += grad.pow(2).sum()
            if self.inverse:
                grad = _inv(weight.cov_grad) @ grad @ _inv(weight.cov_act)
                del weight.cov_act, weight.cov_grad
            else:
                grad = _collapse(grad, weight.V, weight.S_act)  # D_in side
                grad = _collapse(grad.mT, weight.U, weight.S_grad).mT  # D_out side
            filtered_sq += grad.pow(2).sum()
            weight.filtered_grad = grad.to(weight.dtype)
            weight.grad = None

        if self.inverse:
            # The inverses carry the covs' raw scales. One global factor brings the
            # update back to the forget gradient's norm, so that the step sizes are
            # in the usual range; the modules keep their relative sizes.
            # This could be skipped: it is one global scale, which the step size
            # (learning_rate, or RepSelectAdaptive's search) absorbs.
            scale = (grad_sq / filtered_sq).sqrt()
            for weight in self.base_trainable_params:
                weight.filtered_grad *= scale

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
