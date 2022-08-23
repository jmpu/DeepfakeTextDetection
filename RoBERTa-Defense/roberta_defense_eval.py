# -*- coding: utf-8 -*-
"""
RoBERTa Testing
"""

import os
import torch
import datetime
import jsonlines
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaForSequenceClassification, AdamW, RobertaConfig, get_linear_schedule_with_warmup, \
    Trainer, RobertaTokenizer, TrainingArguments, HfArgumentParser, set_seed, EvalPrediction
import random
import numpy as np
import logging
from torch.utils.data import TensorDataset
from typing import Dict
import argparse
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_auc_score


def dummy_data_collector(features):
    batch = {}
    batch['input_ids'] = torch.stack([f[0] for f in features])
    batch['attention_mask'] = torch.stack([f[1] for f in features])
    batch['labels'] = torch.stack([f[2] for f in features])

    return batch


def compute_metrics(p: EvalPrediction) -> Dict:
    probs = p.predictions
    preds = np.argmax(p.predictions, axis=1)
    accc = np.mean(preds == p.label_ids.reshape(-1))
    print("Flip rate is", 1 - accc)

    precision, recall, fscore, support = score(p.label_ids.reshape(-1), preds)
    auc_test = roc_auc_score(p.label_ids.reshape(-1), probs[:, 1])
    return {"acc": np.mean(preds == p.label_ids.reshape(-1)),
            "auc": auc_test,
            "precision_human": precision[0],
            "recall_human": recall[0],
            "fscore_human": fscore[0],
            "support_human": float(support[0]),
            "precision_machine": precision[1],
            "recall_machine": recall[1],
            "fscore_machine": fscore[1],
            "support_machine": float(support[1]),
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        required=True,

    )

    parser.add_argument(
        "--prediction_output",
        default=None,
        type=str,
        required=True,
        help="The file that saves the prediction metric of the test_dir"
    )

    parser.add_argument(
        "--test_dir",
        default=None,
        type=str,
        required=True,
        help="The dataset path for evaluation"
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The folder that saves model checkpoints while training. not used when run testing only"
    )

    parser.add_argument(
        "--logging_file",
        default=None,
        type=str,
        required=True,

    )

    parser.add_argument(
        "--train_batch_size",
        default=None,
        type=int,
        required=True,

    )

    parser.add_argument(
        "--val_batch_size",
        default=None,
        type=int,
        required=True,

    )

    parser.add_argument(
        "--model_ckpt_path",
        default=None,
        type=str,
        required=True,
        help="The detection model checkpoint to be loaded for evaluation"
    )

    parser.add_argument(
        "--num_train_epochs",
        default=None,
        type=int,
        required=True,

    )
    parser.add_argument(
        "--tensor_logging_dir",
        default=None,
        type=str,
        required=True,

    )

    parser.add_argument(
        "--save_steps",
        default=None,
        type=int,
        required=True,
        help="not used if evaluation only"
    )

    args = parser.parse_args()

    print('Loading Roberta tokenizer...')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir=args.cache_dir)

    '''Loading Test Data'''

    labels = []
    all_articles_test = []
    with jsonlines.open(args.test_dir, 'r') as input_articles:
        for article in input_articles:
            # if((article['split'] == 'test') and article['orig_split'] == 'gen'):
            all_articles_test.append(article['article'])
            labels.append(article['label'])

    encoded_article = tokenizer.batch_encode_plus(
        all_articles_test,
        truncation=True,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        padding='longest',
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    input_ids_test = [input_ids.reshape(1, -1) for input_ids in encoded_article['input_ids']]
    attention_masks_test = [masks.reshape(1, -1) for masks in encoded_article['attention_mask']]

    labels = np.asarray(labels)
    labels = np.expand_dims(np.where((labels == 'machine'), 1, 0), 1)
    labels_test = torch.from_numpy(labels)

    input_ids_test = torch.cat(input_ids_test, dim=0)
    attention_masks_test = torch.cat(attention_masks_test, dim=0)

    test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)

    model = RobertaForSequenceClassification.from_pretrained(
        args.model_ckpt_path,  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.))
        cache_dir=args.cache_dir
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=False,
        do_eval=False,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.save_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        do_predict=True,
        eval_steps=args.save_steps, 
        logging_dir=args.tensor_logging_dir

    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=args.logging_file
    )

    logger = logging.getLogger(__name__)

    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Number of GPUS available : {}".format(torch.cuda.device_count()))

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=dummy_data_collector,
        compute_metrics=compute_metrics
    )


    with jsonlines.open(args.prediction_output, 'w') as pred_out:
        preds_man = trainer.predict(test_dataset)
        print(preds_man[2])
        pred_out.write(preds_man[2])


if __name__ == "__main__":
    main()
