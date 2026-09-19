"""ROUGE-1 recall on free generations from the recall questions.

Addresses the objection that a teacher-forced answer probability does not show
whether the model can still *produce* the answer: here the model is prompted with
the question only, generates greedily, and the generation is scored against the
reference answer.

Recall (not precision/F1) is the right variant: a base model keeps rambling past
the answer, which precision would punish for reasons unrelated to unlearning.
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


class RougeGenEvaluator:
    """Greedy-generation ROUGE-1 recall against the reference answer.

    Config keys:
        dataset_name: raw-question split in `data` (e.g. `eval_qs`), each item with
                      `question`, `choices`, `answer` (the wmdp_low_mi format).
        max_new_tokens: generation cap; answers are ~6 tokens (p90 13, max 26),
                        so the default 32 is generous without being wasteful.
        only_at_relearn: if true, skip the unlearning stage entirely. Nothing is lost:
                         the relearn stage's epoch-0 eval already measures the unlearned
                         model, and generation is the expensive part of this evaluator.
        only_at_relearn_start: stricter still -- only the relearn stage's epoch-0 eval.
        log_n_generations: how many (question, reference, generation) triples to keep for
                           inspection; null (the default) keeps all of them. They go to
                           wandb as a table and to a json in output_dir; only the first
                           few also go to stdout, to avoid spamming the runner's log.
    """

    STDOUT_N = 3

    def __init__(self, eval_cfg, data, **kwargs):
        self.dataset_name = eval_cfg.get("dataset_name", "eval_qs")
        self.max_new_tokens = eval_cfg.get("max_new_tokens", 32)
        self.only_at_relearn = eval_cfg.get("only_at_relearn", False)
        self.only_at_relearn_start = eval_cfg.get("only_at_relearn_start", False)
        self.mode = kwargs.get("mode")
        self.log_n = eval_cfg.get("log_n_generations", None)  # None = all
        self.questions = data[self.dataset_name]
        self.scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        self.logged_rows = []  # accumulated across evals, re-logged as one growing table

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        trainer = kwargs["trainer"]
        if (self.only_at_relearn or self.only_at_relearn_start) and self.mode != "relearn":
            return {}
        if self.only_at_relearn_start and trainer.state.epoch:
            return {}

        tokenizer = kwargs["tokenizer"]
        model.eval()
        model.zero_grad(set_to_none=True)

        prompts = [f"{q['question'].strip()}\nAnswer:" for q in self.questions]
        references = [q["choices"][q["answer"]] for q in self.questions]
        samples = []  # (prompt, reference, generation, recall) kept for inspection

        # left padding so that every sequence's generation starts at the same index
        orig_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        recalls = []
        try:
            for batch_prompts, batch_refs in zip(
                _batched(prompts, trainer.args.per_device_eval_batch_size),
                _batched(references, trainer.args.per_device_eval_batch_size),
            ):
                enc = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(
                    model.device
                )
                with pt.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                gen = out[:, enc["input_ids"].shape[1] :]
                texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
                for prompt, text, ref in zip(batch_prompts, texts, batch_refs):
                    score = self.scorer.score(ref, text)["rouge1"]
                    recalls.append(score.recall)
                    if self.log_n is None or len(samples) < self.log_n:
                        samples.append((prompt, ref, text.strip(), score.recall))
        finally:
            tokenizer.padding_side = orig_side

        mean_recall = sum(recalls) / len(recalls)
        if samples:
            self._log_generations(samples, trainer, output_dir)
        return {f"{self.dataset_name}_rouge1_recall": mean_recall}

    def _log_generations(self, samples, trainer, output_dir):
        """Keep a few generations for inspection: wandb table, stdout, and a json file."""
        epoch = trainer.state.epoch or 0
        stage = self.mode or "unlearn"
        for prompt, ref, gen, recall in samples:
            self.logged_rows.append([stage, epoch, prompt, ref, gen, recall])

        for _, ref, gen, recall in samples[: self.STDOUT_N]:
            logger.info(f"[{stage} ep{epoch}] r1={recall:.2f} ref={ref!r} gen={gen!r}")

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
            path = Path(output_dir) / "generations.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.logged_rows, indent=1))
