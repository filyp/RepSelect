"""ROUGE-1 recall on free generations.

Two evaluators share the generation and logging code:

RougeGenEvaluator (forget side): prompt with a recall question only, generate greedily,
score against the reference answer. Addresses the objection that a teacher-forced answer
probability does not show whether the model can still *produce* the answer.

RougeContinuationEvaluator (retain side): plain-text passages (e.g. `retain_eval`) are
not Q&A, so each passage is split: the first `prompt_tokens` tokens are the prompt, the
next `reference_tokens` tokens are the reference, and the greedy continuation is scored
against that reference. Same metric, same generation length, on data that must be kept.

Recall (not precision/F1) is the right variant: a base model keeps rambling past the
answer, which precision would punish for reasons unrelated to unlearning.
"""

import json
import logging
from pathlib import Path

import torch as pt
from rouge_score import rouge_scorer

logger = logging.getLogger("evaluator")
# rouge_scorer logs one INFO line per call
logging.getLogger("absl").setLevel(logging.WARNING)


def _batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


class _RougeGenBase:
    """Shared gating, greedy generation from token ids, scoring and logging.

    Config keys common to both evaluators:
        max_new_tokens: generation cap (default 32).
        only_at_relearn: if true, skip the unlearning stage entirely. Nothing is lost:
                         the relearn stage's epoch-0 eval already measures the unlearned
                         model, and generation is the expensive part of this evaluator.
        only_at_relearn_start: stricter still -- only the relearn stage's epoch-0 eval.
        log_n_generations: how many (prompt, reference, generation) triples to keep for
                           inspection; null (the default) keeps all of them. They go to
                           wandb as a table and to a json in output_dir; only the first
                           few also go to stdout, to avoid spamming the runner's log.
    """

    STDOUT_N = 3

    def __init__(self, eval_cfg, data, **kwargs):
        self.dataset_name = eval_cfg.get("dataset_name")
        self.max_new_tokens = eval_cfg.get("max_new_tokens", 32)
        self.only_at_relearn = eval_cfg.get("only_at_relearn", False)
        self.only_at_relearn_start = eval_cfg.get("only_at_relearn_start", False)
        self.mode = kwargs.get("mode")
        self.log_n = eval_cfg.get("log_n_generations", None)  # None = all
        self.scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        self.logged_rows = []  # accumulated across evals, re-logged as one growing table

    def _skip(self, trainer):
        if (self.only_at_relearn or self.only_at_relearn_start) and self.mode != "relearn":
            return True
        if self.only_at_relearn_start and trainer.state.epoch:
            return True
        return False

    def _generate(self, model, tokenizer, prompt_ids, batch_size):
        """Greedy continuations for a list of 1-D prompt id tensors (left-padded)."""
        model.eval()
        model.zero_grad(set_to_none=True)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        texts = []
        for batch in _batched(prompt_ids, batch_size):
            width = max(len(ids) for ids in batch)
            input_ids = pt.full((len(batch), width), pad_id, dtype=pt.long)
            attention_mask = pt.zeros((len(batch), width), dtype=pt.long)
            for i, ids in enumerate(batch):  # left padding: generation starts at one index
                input_ids[i, width - len(ids) :] = ids
                attention_mask[i, width - len(ids) :] = 1
            with pt.no_grad():
                out = model.generate(
                    input_ids=input_ids.to(model.device),
                    attention_mask=attention_mask.to(model.device),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
            gen = out[:, width:]
            texts.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        return texts

    def _score_and_log(self, prompts, references, generations, trainer, output_dir):
        recalls, samples = [], []
        for prompt, ref, gen in zip(prompts, references, generations):
            recall = self.scorer.score(ref, gen)["rouge1"].recall
            recalls.append(recall)
            if self.log_n is None or len(samples) < self.log_n:
                samples.append((prompt, ref, gen.strip(), recall))
        if samples:
            self._log_generations(samples, trainer, output_dir)
        return sum(recalls) / len(recalls)

    def _log_generations(self, samples, trainer, output_dir):
        """Keep generations for inspection: wandb table, stdout, and a json file."""
        epoch = trainer.state.epoch or 0
        stage = self.mode or "unlearn"
        for prompt, ref, gen, recall in samples:
            self.logged_rows.append([stage, epoch, prompt, ref, gen, recall])

        for _, ref, gen, recall in samples[: self.STDOUT_N]:
            logger.info(f"[{self.dataset_name} {stage} ep{epoch}] r1={recall:.2f} ref={ref!r} gen={gen!r}")

        try:  # wandb is the durable copy; the runner's stdout may not be kept
            import wandb

            if wandb.run is not None:
                table = wandb.Table(
                    columns=["stage", "epoch", "prompt", "reference", "generation", "rouge1_recall"],
                    data=[list(r) for r in self.logged_rows],
                )
                # commit=False: attach to the step the trainer is about to commit
                wandb.log({f"{self.dataset_name}_generations": table}, commit=False)
        except Exception as e:  # never fail a run over logging
            logger.warning(f"could not log generations to wandb: {e}")

        if output_dir:
            path = Path(output_dir) / f"{self.dataset_name}_generations.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.logged_rows, indent=1))


class RougeGenEvaluator(_RougeGenBase):
    """Question-only prompt, greedy answer, ROUGE-1 recall against the reference answer.

    Config keys (plus the common ones above):
        dataset_name: raw-question split in `data` (e.g. `eval_qs`), each item with
                      `question`, `choices`, `answer` (the wmdp_low_mi format).
        max_new_tokens: answers are ~6 tokens (p90 13, max 26), so 32 is generous.
    """

    def __init__(self, eval_cfg, data, **kwargs):
        super().__init__(eval_cfg, data, **kwargs)
        self.dataset_name = eval_cfg.get("dataset_name", "eval_qs")
        self.questions = data[self.dataset_name]

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        trainer = kwargs["trainer"]
        if self._skip(trainer):
            return {}
        tokenizer = kwargs["tokenizer"]

        prompts = [f"{q['question'].strip()}\nAnswer:" for q in self.questions]
        references = [q["choices"][q["answer"]] for q in self.questions]
        prompt_ids = [tokenizer(p, return_tensors="pt")["input_ids"][0] for p in prompts]

        generations = self._generate(
            model, tokenizer, prompt_ids, trainer.args.per_device_eval_batch_size
        )
        mean_recall = self._score_and_log(prompts, references, generations, trainer, output_dir)
        return {f"{self.dataset_name}_rouge1_recall": mean_recall}


class RougeContinuationEvaluator(_RougeGenBase):
    """Passage split into prompt / reference continuation, greedy continuation scored
    with ROUGE-1 recall against the reference. For plain-text splits such as `retain_eval`.

    Config keys (plus the common ones above):
        dataset_name: tokenized split in `data` (items with `input_ids`), e.g. `retain_eval`.
        prompt_tokens: number of leading tokens used as the prompt (default 64).
        reference_tokens: number of tokens after the prompt used as the reference
                          (default = max_new_tokens, so reference and generation have
                          the same length). Passages shorter than prompt_tokens +
                          reference_tokens are skipped.
    """

    def __init__(self, eval_cfg, data, **kwargs):
        super().__init__(eval_cfg, data, **kwargs)
        self.dataset_name = eval_cfg.get("dataset_name", "retain_eval")
        self.prompt_tokens = eval_cfg.get("prompt_tokens", 64)
        self.reference_tokens = eval_cfg.get("reference_tokens", self.max_new_tokens)
        need = self.prompt_tokens + self.reference_tokens
        self.samples = [s for s in data[self.dataset_name] if len(s["input_ids"]) >= need]
        skipped = len(data[self.dataset_name]) - len(self.samples)
        if skipped:
            logger.info(f"{self.dataset_name}: skipping {skipped} passages shorter than {need} tokens")
        assert self.samples, f"{self.dataset_name}: no passage has >= {need} tokens"

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        trainer = kwargs["trainer"]
        if self._skip(trainer):
            return {}
        tokenizer = kwargs["tokenizer"]

        prompt_ids = [s["input_ids"][: self.prompt_tokens] for s in self.samples]
        ref_ids = [
            s["input_ids"][self.prompt_tokens : self.prompt_tokens + self.reference_tokens]
            for s in self.samples
        ]
        prompts = tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        references = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)

        generations = self._generate(
            model, tokenizer, prompt_ids, trainer.args.per_device_eval_batch_size
        )
        mean_recall = self._score_and_log(prompts, references, generations, trainer, output_dir)
        return {f"{self.dataset_name}_rouge1_recall": mean_recall}
