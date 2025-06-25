import pandas as pd
import numpy as np

import torch
from transformers import TrainerCallback, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, roc_auc_score

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter

from typing import List, Callable

import re
import os

all_txt_sem_rep_dir = os.path.join('data', 'semantic_representations')
all_txt_pickle_prefix = 'rep_list_'
dialogue_only_pickle_prefix = 'dialogue_only_rep_list_'

pooling_models = [
    'cardiffnlp/twitter-roberta-large-sensitive-multilabel', # 0
    'cardiffnlp/twitter-roberta-base-offensive',
    'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'GroNLP/hateBERT',
    'microsoft/deberta-v3-large',
    'joeddav/distilbert-base-uncased-go-emotions-student', # 5
    'mrm8488/t5-base-finetuned-imdb-sentiment'
]

def convert_col_to_ordinal(series: pd.Series, compact: bool = True) -> pd.Series:
    
    low_cat = ['none', 'very mild', 'mild']
    med_cat = ['moderate']
    high_cat = ['strong', 'high']
    
    if compact:
        new_series = np.select(
            [series.isin(low_cat), series.isin(med_cat), series.isin(high_cat)],
            [0, 1, 2],
            default=-1
        )
    else:
        new_series = np.select(
            [series.eq(low_cat[0]), series.eq(low_cat[1]), series.eq(low_cat[2]), series.eq(med_cat[0]), series.eq(high_cat[0]), series.eq(high_cat[1])],
            [0, 1, 2, 3, 4, 5],
            default=-1
        )
    
    return new_series


class FilmDataset(torch.utils.data.Dataset):
    """Dataset for film embeddings and ratings"""
    
    def __init__(self, embeddings, ratings):
        self.embeddings = torch.stack(embeddings) if isinstance(embeddings[0], torch.Tensor) else torch.tensor(embeddings)
        self.ratings = torch.tensor(ratings, dtype=torch.long)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.ratings[idx]
    

class FractionalTrainEvalCallback(TrainerCallback):
    def __init__(self):
        self.trainer = None  # will be set at init

    def on_init_end(self, args, state, control, model=None, **kwargs):
        # Save trainer reference after init
        self.trainer = kwargs.get("trainer", None)

    def on_evaluate(self, args, state, control, **kwargs):
        if self.trainer is not None:
            train_metrics = self.trainer.evaluate(
                eval_dataset=self.trainer.train_dataset,
                metric_key_prefix="train"
            )
            self.trainer.log(train_metrics)
        return control


# from peft import get_peft_model, LoraConfig, TaskType
# from transformers import AutoModelForSequenceClassification

# base_model = AutoModelForSequenceClassification.from_pretrained(enc_model, num_labels=1)

# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     target_modules=["query", "value"],  # key attention layers
#     lora_dropout=0.1,
#     bias="none",
#     task_type=TaskType.SEQ_CLS
# )

# model = get_peft_model(base_model, lora_config)


class CustomLossModel(torch.nn.Module):
    def __init__(self, model_name: str, loss_fn: Callable, unfrozen_layers: List[int], dropout: float):
        super().__init__()
        args = {'classifier_dropout': dropout} if 'deberta' not in model_name.lower() else {'hidden_dropout_prob': dropout}
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            **args
        )
        
        #, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to('cuda')
        self.loss_fn = loss_fn
        
        # Freeze early layers
        
        for name, param in self.model.named_parameters():
            # Match patterns like encoder.layer.X or encoder.layers.X
            lm_head_names = ('classifier', 'head', 'final', 'pooler')
            match = re.search(r'\.(?:layer|layers)\.(\d+)\.', name)
            if any(x in name for x in lm_head_names):
                param.requires_grad = True
            elif match:
                layer_num = int(match.group(1))
                param.requires_grad = layer_num in unfrozen_layers
            else:
                # Freeze everything else by default (e.g., embeddings, pooler)
                param.requires_grad = False
        # for name, param in self.model.named_parameters():
        #     param.requires_grad = any([f"encoder.layer.{i}." in name for i in unfrozen_layers])

    def forward(self, input_ids=None, attention_mask=None, labels=None, num_items_in_batch=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs  # <-- important
        )
        logits = outputs.logits

        if labels is not None:
            labels = labels.float().view(-1, 1)  # shape [batch_size, 1]
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}


def compute_metrics(eval_pred):
    logits, labels = eval_pred  # logits shape: (N, 1), labels shape: (N,)
    
    # Convert logits to probabilities using sigmoid
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int).reshape(-1)
    labels = labels.reshape(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_macro": f1_score(labels, preds, average='macro'),
        "weighted_score": 0.8 * recall + 0.2 * precision
    }
    


def check_trainable_layers(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"[FROZEN] {name}")
        else:
            print(f"[TRAINABLE] {name}")
            

def process_text(text: str, excl_stopwords: bool):
    
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalpha()]
    if excl_stopwords:
        filtered_tokens = [word for word in filtered_tokens if word not in stop_words]
        
    return filtered_tokens


def get_ngram_counts(text, n, top_n=10, excl_stopwords: bool = True):
    
    tokens = process_text(text, excl_stopwords)
    ngram_list = list(ngrams(tokens, n))
    ngram_counts = Counter(ngram_list)
    ngram_df = pd.DataFrame(ngram_counts.most_common(top_n), columns=['Ngram', 'Frequency'])
    ngram_df['Ngram'] = ngram_df['Ngram'].apply(lambda x: ' '.join(x))
    
    return ngram_df


