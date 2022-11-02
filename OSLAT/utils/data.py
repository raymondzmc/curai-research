import os
import json
import torch
import spacy
import collections

from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize.treebank import TreebankWordTokenizer
import re
import pdb


class HNLPContrastiveNERDataset(Dataset):

    def __init__(self, data, tokenizer, id2synonyms):
        super(HNLPContrastiveNERDataset, self).__init__()
        nlp = spacy.load("en_core_web_sm")
        spacy_tokenizer = nlp.tokenizer

        

        self.data = []
        for example in data:
            text = example['text']
            text = re.sub(r"\[\*(.*?)\*\]", '', text)
            tokens = [token.text for token in spacy_tokenizer(text)]
            text_inputs = tokenizer(tokens, return_tensors="pt", is_split_into_words=True, truncation=True, max_length=512)
            synonym_inputs = []
            ground_truth_masks = []

            for entity_id, entity_annotations in example['entities']:
                try:
                    synonyms = id2synonyms[entity_id]
                except:
                    continue

                # For generating the ground-truth masks for evaluating linking upperbound
                matched_str = [x[1] for x in entity_annotations]
                matched_str_len = [len(x[1].split()) for x in entity_annotations]
                gt_mask = np.zeros(len(tokens))
                for tok_idx in range(len(tokens)):
                    for matched_len in matched_str_len:
                        if (tok_idx + matched_len) <= len(tokens) and \
                           (' '.join(tokens[tok_idx: tok_idx + matched_len]) in matched_str):
                            gt_mask[tok_idx: tok_idx + matched_len] = 1

                # Convert to subword token mask
                token2word = text_inputs.word_ids()
                gt_token_mask = np.zeros(len(token2word))
                for token_idx, word_idx in enumerate(token2word):
                    if word_idx != None and gt_mask[word_idx] == 1:
                         gt_token_mask[token_idx] = 1

                ground_truth_masks.append(gt_token_mask)


                synonym_inputs.append(tokenizer(synonyms, return_tensors="pt", padding=True))


            # Keep examples with at least one concept
            if len(synonym_inputs) > 0:
                self.data.append({
                    'id': example['id'],
                    'text_inputs': text_inputs,
                    'synonym_inputs': synonym_inputs,
                    'entity_ids': [entity[0] for entity in example['entities']],
                    'tokens': tokens,
                    'multispan': [len(entity[1]) > 1 for entity in example['entities']],
                    'ground_truth_masks': ground_truth_masks,
                })

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class ContrastiveNERDataset(Dataset):

    def __init__(self, args, indices=[]):
        super(ContrastiveNERDataset, self).__init__()

        processed_data = torch.load(args.processed_data_path)
        self.data = processed_data['data']

        if len(indices):
            self.indices = set(indices)
            self.data = [x for i, x in enumerate(self.data) if i in set(self.indices)]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def ner_collate_fn(batch):
    results = collections.defaultdict(list)
    results['input'] = collections.defaultdict(list)
    results['ids'] = []
    for example in batch:
        for k, v in example['input'].items():
            results['input'][k].append(v.squeeze(0))

        results['entities'].append(example['entities'])
        results['ids'].append(example['id'])
    results['input']['input_ids'] = pad_sequence(results['input']['input_ids'], batch_first=True, padding_value=0)
    results['input']['token_type_ids'] = pad_sequence(results['input']['token_type_ids'], batch_first=True, padding_value=0)
    results['input']['attention_mask'] = pad_sequence(results['input']['attention_mask'], batch_first=True, padding_value=0)
    results['text'] = [example['text'] for example in batch]
    return results

def get_dataloaders_ner(args, train_ids, test_ids):

    train_set = ContrastiveNERDataset(args, train_ids)
    test_set = ContrastiveNERDataset(args, test_ids)
    train_loader = DataLoader(train_set,
                              batch_size=1,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=ner_collate_fn)

    test_loader = DataLoader(test_set,
                             batch_size=args.eval_batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             collate_fn=ner_collate_fn)

    return train_loader, test_loader



def process_json_file_ner(file_path, tokenizer, k_folds):
    with open(file_path) as f:
        json_data = json.load(f)

    data = json_data['data']
    for example in json_data['data']:
        example['ground_truth_masks'] = dict()

        has_annotations = any([v for k, v in example['annotations'].items()])
        if has_annotations:
            try:
                annotations = example['annotations'][example['entities'][0]][0]
            except:
                pdb.set_trace()
            input_text = annotations[0].split(tokenizer.sep_token)[0].strip()
            annotated_tokens = list(TreebankWordTokenizer().tokenize(input_text))
            example['input'] = tokenizer(annotated_tokens, return_tensors="pt", is_split_into_words=True, truncation=True,)
        else:
            example['input'] = tokenizer(example['text'], return_tensors="pt")

        for entity_idx, name in enumerate(example['entities']):
            
            # Not all entities have annotations
            if has_annotations and example['annotations'][name]:
                annotation_string = ' '.join(annotated_tokens)

                # For aligning annotations to tokens
                text_len = 0
                start2idx, end2idx = {}, {}
                for tok_idx, tok in enumerate(annotated_tokens):
                    tok_start = tok_idx + text_len
                    start2idx[tok_start] = tok_idx
                    end2idx[tok_start + len(tok)] = tok_idx
                    text_len += len(tok)
                spans = [(x[0], x[1]) for x in annotations[1]['entities']]

                try:
                    token_level_spans = [(start2idx[x[0]], end2idx[x[1]]) for x in spans]
                except:
                    continue
                    print("Couldn't find the token index of a span!")
                    pdb.set_trace()


                # Ground-truth token mask for the entity
                gt_mask = np.zeros(len(annotated_tokens))
                for span in token_level_spans:
                    gt_mask[span[0]: span[1] + 1] = 1

                # Convert to subword token mask
                token2word = example['input'].word_ids()
                gt_token_mask = np.zeros(len(token2word))
                for token_idx, word_idx in enumerate(token2word):
                    if word_idx != None and gt_mask[word_idx] == 1:
                         gt_token_mask[token_idx] = 1

                example['ground_truth_masks'][name] = gt_token_mask
                assert len(gt_token_mask) == len(example['input']['attention_mask'].squeeze(0))


    entity_synonyms = {}
    for entity_name, synonyms in json_data['entity_synonyms'].items():
        entity_synonyms[entity_name] = tokenizer(synonyms, return_tensors="pt", padding=True)

    pretrain_entity_synonyms = {}
    for entity_name, synonyms in json_data['pretrain_entity_synonyms'].items():
        pretrain_entity_synonyms[entity_name] = tokenizer(synonyms, return_tensors="pt", padding=True)
    # kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)
    # folds = [x for x in kfold.split(data)]
    results = {
        'data': data,
        'entity_synonyms': entity_synonyms,
        'pretrain_entity_synonyms': pretrain_entity_synonyms,
        'synonyms_names': json_data['entity_synonyms'],
    }
    return results


