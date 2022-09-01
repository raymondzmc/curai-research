rm resources/CuRSA-FIXED-v0-processed-all.pth

echo "========= RFE (updated) ========="
echo
python main.py -dataset rfe -freeze_weights -classification_loss focal

echo "========= RFE W/O Contrastive Loss ========="
echo
python main.py -dataset rfe -freeze_weights -classification_loss focal -wo_contrastive


echo "========= RFE W/O Pretraining ========="
echo
python main.py -dataset rfe -freeze_weights -classification_loss focal -wo_pretraining


echo "========= RFE Evaluated on hNLP ========="
echo
python main.py -dataset hnlp -cross_dataset -freeze_weights -classification_loss focal 