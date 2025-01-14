import os, re, csv, json, torch, random, argparse, collections
from os.path import join as pjoin
import numpy as np
from run_hNLP import run_hnlp
from run_RFE import run_rfe


import pdb

ROOT_DIR = os.getcwd()

# Model names from HuggingFace library
encoder_names = {
    'sapbert-cls': 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
    'sapbert-mean': 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token',
    'scibert': 'allenai/scibert_scivocab_uncased',
    'bluebert': 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
    'clinicalbert': 'emilyalsentzer/Bio_ClinicalBERT',
    'biobert': 'dmis-lab/biobert-base-cased-v1.1',
    'pubmedbert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Optimization settings
    parser.add_argument('-optim', default='ADAM', type=str, choices=['ADAM', 'SGD'])
    parser.add_argument('-max_grad_norm', default=0, type=float)
    parser.add_argument('-lr', default=2e-4, type=float)
    parser.add_argument('-warmup_steps', default=100, type=int)
    parser.add_argument('-decay_method', default='noam', type=str, choices=['noam', 'none'])
    parser.add_argument('-beta1', default=0.9, type=float)
    parser.add_argument('-beta2', default=0.999, type=float)
    parser.add_argument('-epochs', default=10, type=int)


    # Effective batch size = batch_size * grad_accum_steps
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-grad_accum_steps', type=int, default=32)
    parser.add_argument('-eval_batch_size', type=int, default=1)
    parser.add_argument('-attention_heatmap_path', default=None)

    parser.add_argument('-finetune_encoder', action='store_true', default=True)
    parser.add_argument('-encoder', type=str, default='biobert', choices=encoder_names.keys())

    # Attention layer
    parser.add_argument('-use_attention', action='store_true', default=True)
    parser.add_argument('-use_multi_head', action='store_true', default=False)
    parser.add_argument('-num_attention_heads', type=int, default=6)
    parser.add_argument('-attention_head_size', type=int, default=64)
    parser.add_argument('-attention_probs_dropout_prob', type=float, default=0.1)
    parser.add_argument('-attention_type', type=int, default=0)
    parser.add_argument('-ignore_cls', help='Avoid attending to [CLS] tokens in the Entity Attention Layer', action='store_true', default=True)


    parser.add_argument('-normalize_query_embeddings', action='store_true', default=False)
    parser.add_argument('-entity_embedding_method', type=str, choices=['name', 'max', 'mean'], default='name')
    parser.add_argument('-pooling_method', type=str, default='cls', choices=['max', 'mean', 'cls'])

    # parser.add_argument('-freeze_weights', type=str, default='none', choices=['encoder', 'attention'])

    # Linear Projection Layer
    parser.add_argument('-use_projection_head', action='store_true', help="Apply a linear projection head on top of attention output", default=True)
    parser.add_argument('-projection_hidden_size', type=int, default=128)
    parser.add_argument('-projection_output_size', type=int, default=128)

    # Classifier layer
    parser.add_argument('-use_classification_loss', action='store_true', default=True)
    parser.add_argument('-classifier_hidden_size', type=int, default=200)
    parser.add_argument('-normalize_final_hidden', action='store_true', default=False)
    parser.add_argument('-share_classifier', action='store_true', default=False)

    # Entity Linking
    parser.add_argument('-hidden_states_pooling_method', type=str, default='mean', choices=['max', 'mean', 'cls'])

    # Contrastive learning
    parser.add_argument('-temperature', type=float, default=0.07)
    parser.add_argument('-use_contrastive_loss', action='store_true', default=True)
    parser.add_argument('-contrastive_lambda', type=float, default=0.1)
    parser.add_argument('-num_negatives', help='Number of negatives for contrastive loss', type=int, default=100)
    parser.add_argument('-retrieval_loss', action='store_true', help='Use retrieval loss by push together label-mention with label representation',)

    parser.add_argument('-combine_subwords', help='Combine subword tokens for visualization', action='store_true', default=True)
    
    parser.add_argument('-num_workers', type=int, default=0)
    parser.add_argument('-k_folds', type=int, default=5)
    parser.add_argument('-save_checkpoints', action='store_true', default=False)
    parser.add_argument('-checkpoints_dir', default=pjoin(ROOT_DIR, 'checkpoints'))

    parser.add_argument('-compare_rule_based', action='store_true', default=False)

    parser.add_argument('-load_model', type=str, default=None)
    parser.add_argument('-save_model', type=str, default=None)

    parser.add_argument('-random_seed', type=int, default=0)
    parser.add_argument('-folds', type=int, nargs='+', default=None)

    parser.add_argument('-append_query', action='store_true', help="Append query vector to the attention output")

    # Ablation Experiments
    parser.add_argument('-wo_pretraining', action='store_true')
    parser.add_argument('-wo_contrastive', action='store_true')
    parser.add_argument('-cross_dataset', action='store_true', help="Cross-dataset experiments")

    # Classifier
    parser.add_argument('-classification_loss', default='bce', type=str, choices=['bce', 'focal'])
    parser.add_argument('-freeze_weights', action='store_true')

    # Dataset to run
    parser.add_argument('-dataset', default='hnlp', type=str, choices=['rfe', 'hnlp'])

    parser.add_argument('-device', type=int, default=0)

    args = parser.parse_args()

    args.encoder_name = encoder_names[args.encoder]

    os.makedirs(args.checkpoints_dir, exist_ok=True)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    
    args.device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'hnlp':
        run_hnlp(args)
    elif args.dataset == 'rfe':
        run_rfe(args)
    else:
        raise NotImplementedError

    



