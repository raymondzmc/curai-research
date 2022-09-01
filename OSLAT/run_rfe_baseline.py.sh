
# RFE model on hNLP
echo "========= BioBERT (unsupervised) ========="
echo
python main.py -dataset rfe -retrieval_loss -wo_contrastive -wo_pretraining


echo "========= BioBERT (supervised) ========="
echo
python main.py -dataset rfe -retrieval_loss -wo_contrastive

echo "========= PubMedBERT (unsupervised) ========="
echo
python main.py -encoder pubmedbert -dataset rfe -retrieval_loss -wo_contrastive -wo_pretraining


echo "========= PubMedBERT (supervised) ========="
echo
python main.py -encoder pubmedbert -dataset rfe -retrieval_loss -wo_contrastive


echo "========= SAP-BERT (unsupervised) ========="
echo
python main.py -encoder sapbert-cls -dataset rfe -retrieval_loss -wo_contrastive -wo_pretraining


echo "========= SAP-BERT (supervised) ========="
echo
python main.py -encoder sapbert-cls -dataset rfe -retrieval_loss -wo_contrastive
