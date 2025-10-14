from pathlib import Path
from typing import Optional, Callable

import torch
from torchmetrics import AUROC
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_utils import EvalPrediction

from thesis import constants as c, utils
from thesis.dicova2.datasets import Dicova2Dataset, Dicova2DevFoldHandler
from thesis.coughvid.datasets import CoughvidDataset, SplitManager
from thesis.cough_detection.datasets import get_fold


def d2_cough_experiment(
    experiment_name: str,
    parent_dir: Path,
    create_model: Callable,
    train_args: TrainingArguments,
    transform: Optional[Callable] = None,
    early_stopping_patience: Optional[int] = None,
):
    output_dir = parent_dir / experiment_name
    audio_type = "cough"

    results = []

    dataset = Dicova2Dataset(audio_type, transform=transform)
    fold_handler = Dicova2DevFoldHandler(dataset)

    for fold in range(5):
        train_dataset, val_dataset = fold_handler.get_fold(fold)
        trainer = create_trainer(
            create_model(),
            train_dataset,
            val_dataset,
            train_args,
            early_stopping_patience,
        )
        trainer.train()
        utils.save_log_history(
            trainer.state.log_history, f"{output_dir}/fold_{fold}.json"
        )
        evaluation = trainer.evaluate()
        results.append(("cough", fold, evaluation["eval_auroc"]))

    utils.save_results(results, "results.txt", output_dir)


def compute_metrics(eval_pred: EvalPrediction):
    auroc_metric = AUROC(task="binary")
    labels = eval_pred.label_ids
    logits = eval_pred.predictions[:, 1]  # 1 is the index of the positive class
    auroc_score = auroc_metric(torch.tensor(logits), torch.tensor(labels))
    return {"auroc": auroc_score}


def create_trainer(
    model,
    train_dataset,
    eval_dataset,
    training_args,
    early_stopping_patience: Optional[int] = None,
):
    if early_stopping_patience is not None:
        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        ]
    else:
        callbacks = None

    return Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )


def coughvid_experiment(
    experiment_name: str,
    parent_dir: Path,
    create_model: Callable,
    train_args: TrainingArguments,
    transform: Optional[Callable] = None,
    early_stopping_patience: Optional[int] = None,
):
    output_dir = parent_dir / experiment_name

    results = []

    dataset = CoughvidDataset(transform=transform)
    split_manager = SplitManager(dataset)
    for fold in range(5):
        train_dataset, eval_dataset = split_manager.get_fold(fold)
        trainer = create_trainer(
            create_model(),
            train_dataset,
            eval_dataset,
            train_args,
            early_stopping_patience=early_stopping_patience,
        )
        trainer.train()
        utils.save_log_history(
            trainer.state.log_history, f"{output_dir}/fold-{fold}-log.json"
        )
        evaluation = trainer.evaluate()
        results.append(("wet-dry", fold, evaluation["eval_auroc"]))

    utils.save_results(results, f"{experiment_name}.txt", output_dir)


def cough_detection_experiment(
    experiment_name: str,
    parent_dir: Path,
    create_model: Callable,
    train_args: TrainingArguments,
    transform: Optional[Callable] = None,
    early_stopping_patience: Optional[int] = None,
):
    output_dir = parent_dir / experiment_name

    results = []

    for fold in range(5):
        train_dataset, eval_dataset = get_fold(fold, transform=transform)
        trainer = create_trainer(
            create_model(),
            train_dataset,
            eval_dataset,
            train_args,
            early_stopping_patience=early_stopping_patience,
        )
        trainer.train()
        utils.save_log_history(
            trainer.state.log_history, f"{output_dir}/fold-{fold}-log.json"
        )
        evaluation = trainer.evaluate()
        results.append(("detection", fold, evaluation["eval_auroc"]))

    utils.save_results(results, f"{experiment_name}.txt", output_dir)
