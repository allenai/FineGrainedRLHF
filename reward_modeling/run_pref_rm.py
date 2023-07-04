import logging
from dataclasses import dataclass, field

import transformers
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments, HfArgumentParser
from transformers import set_seed

import torch
import json
import os
import sys
import random
import numpy as np

from typing import Any, Dict, List, Optional
import evaluate
import argparse
import datasets
from datasets import Dataset

from my_longformer import LongformerForSequenceClassification

logger = logging.getLogger(__name__)
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_PROJECT"] = "RM"


class PairDataCollatorWithPadding:
    
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.singe_data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=pad_to_multiple_of)
    
    def __call__(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        inputs1 = [x['pred1'] for x in inputs]
        inputs2 = [x['pred2'] for x in inputs]
        labels = [x['label'] for x in inputs]
            
        batch_inputs1 = self.singe_data_collator(inputs1)
        batch_inputs2 = self.singe_data_collator(inputs2)
        
        batch = {}
        batch["labels"] = torch.tensor(labels)
        batch["pred1"] = batch_inputs1
        batch["pred2"] = batch_inputs2
        
        return batch

    
class PreferenceTrainer(Trainer):

    def __init__(
        self, 
        model=None, 
        args=None, 
        train_dataset=None, 
        eval_dataset=None, 
        tokenizer=None, 
        data_collator=None, 
        compute_metrics=None, 
        cal_score_mean_std=False
    ):
        super().__init__(
            model=model, 
            args=args, 
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer, 
            data_collator=data_collator, 
            compute_metrics=compute_metrics)
        self.cal_score_mean_std = cal_score_mean_std
    

    def compute_loss(self, model, inputs, return_outputs=False):
        
        # label 1 -> pred 1 better than pred 2, label -1 -> the other way around
        labels = inputs.pop("labels")
        
        inputs1 = inputs["pred1"]
        outputs1 = model(**inputs1)
        scores1 = outputs1[0]  # BS x 1
        
        inputs2 = inputs["pred2"]
        outputs2 = model(**inputs2)
        scores2 = outputs2[0]  # BS x 1
        
        # preference RM loss as described in instructGPT
        outputs = (scores1 - scores2).view(-1)
        loss = - torch.mean(torch.log(torch.sigmoid(outputs*labels.view(-1))))

        if self.cal_score_mean_std:
            if random.random() > 0.5:
                outputs = scores1.view(-1)
            else:
                outputs = scores2.view(-1)
        
        # put a placeholder number at the beginning of outputs
        # as the hf eval loop cuts the first element ..
        outputs_new = torch.zeros(len(outputs)+1).type_as(outputs)
        outputs_new[1:] = outputs
        
        return (loss, outputs_new) if return_outputs else loss


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    prediction_output_filename: Optional[str] = field(
        default="predictions.txt",
        metadata={"help": "An optional input file name for the output prediction file."},
    )
    lm_output_column_prefix: Optional[str] = field(
        default="prediction", metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    ) # TODO
    preference_column: Optional[str] = field(
        default="preference", metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    ) # TODO
    num_lm_outputs: Optional[int] = field(
        default=5, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    ) # TODO
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    n_lm_outputs: int = field(
        default=4,
        metadata={
            "help": (
                "The number of LM outputs being generated for each input, included in the json files."
            )
        },
    )
    cal_score_mean_std: bool = field(
        default=False,
        metadata={
            "help": (
                ""
            )
        },
    )


def main():

    # See all possible arguments in src/transformers/training_args.py

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize config, model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True
    )

    config.num_labels = 1
    model = LongformerForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir
    )

    if torch.cuda.is_available():
        model.cuda()
    
    # Read the datasets
    def gen(filename):
        with open(filename, "r") as f:
            content = json.loads(f.read())
            for e in content:
                yield e

    raw_data = {}
    if data_args.train_file is not None:
        with open(data_args.train_file, "r") as f:
            raw_data["train"] = json.load(f)
    if data_args.validation_file is not None:
        with open(data_args.validation_file, "r") as f:
            raw_data["validation"] = json.load(f)
    if data_args.test_file is not None:
        with open(data_args.test_file, "r") as f:
            raw_data["test"] = json.load(f)
    

    def process_annotations(annotations):
        examples = []

        prediction_keys = [f"{data_args.lm_output_column_prefix} {i+1}" for i in range(data_args.n_lm_outputs)]
        n_comparisons = (data_args.n_lm_outputs * (data_args.n_lm_outputs - 1)) // 2

        for ann in annotations:    
            predictions = [ann[key] for key in prediction_keys]

            # construct prompt
            tokens = ["question:"] + ann["question"].strip().split()
            for i, p in enumerate(ann['passages']):
                if i == 0:
                    tokens += ["context:"]
                title_string = f"wikipage: {p[0]}"
                p_tokens = title_string.split()
                p_tokens += ['text:'] + ' '.join(p[1:]).split()
                tokens += p_tokens
            tokens += ['answer:']
            prompt = ' '.join(tokens)

            preferences = ann[data_args.preference_column]

            assert len(preferences) == n_comparisons

            pair_id = 0

            for i in range(len(predictions)):
                for j in range(i+1, len(predictions)):
                    pref = preferences[pair_id] # 0 equal, 1 first better, 2 second better
                    pair_id += 1
                    example = {}
                    if pref == 0:
                        continue
                    elif pref == 1:
                        example["pred1"] = tokenizer(prompt + ' ' + predictions[i], truncation=True)
                        example["pred2"] = tokenizer(prompt + ' ' + predictions[j], truncation=True)
                        example["label"] = 1
                    elif pref == 2:
                        example["pred1"] = tokenizer(prompt + ' ' + predictions[j], truncation=True)
                        example["pred2"] = tokenizer(prompt + ' ' + predictions[i], truncation=True)
                        example["label"] = 1
                    else:
                        raise("unknown preference")

                    examples.append(example)

        return examples
    
    if training_args.do_train:
        train_dataset = raw_data["train"]
        train_dataset = process_annotations(train_dataset)
    
    if training_args.do_eval:
        eval_dataset = raw_data["validation"]
        eval_dataset = process_annotations(eval_dataset)

    if training_args.do_predict:
        predict_dataset = raw_data["test"]
        predict_dataset = process_annotations(predict_dataset)

    data_collator = PairDataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    accuracy = evaluate.load("accuracy")
    
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = (predictions > 0).astype(int)
        result = accuracy.compute(predictions=predictions, references=labels)
        return result
    

    # Initialize our Trainer
    trainer = PreferenceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        cal_score_mean_std=data_args.cal_score_mean_std
    )
    

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

        scores_to_write = []
        for i, p in enumerate(predictions):
            e = predict_dataset[i]
            scores_to_write += [float(p)]
        
        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, data_args.prediction_output_filename)
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for score in scores_to_write:
                    writer.write(str(score) + '\n')
            with open(os.path.join(training_args.output_dir, 'mean_std.txt'), "w") as writer:
                writer.write(str(np.mean(scores_to_write)) + '\n')
                writer.write(str(np.std(scores_to_write)) + '\n')


if __name__ == "__main__":
    main()