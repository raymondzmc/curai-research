
# RFE model on hNLP
echo "========= RFE on hNLP ========="
echo
python main.py -dataset hnlp -retrieval_loss -cross_dataset


echo "========= hNLP on RFE ========="
echo
python main.py -dataset rfe -retrieval_loss -cross_dataset