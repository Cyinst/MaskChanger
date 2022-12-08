#CUDA_VISIBLE_DEVICES=2,3 \
PYTHONPATH="/":$PYTHONPATH \
bash tools/dist_train.sh configs/mask2former/m2f_r18_levircd.py 2
#bash tools/dist_train.sh configs/test/test_r18_levircd.py 2
#bash tools/dist_train.sh configs/pcam/graph_pcam_r18_512x512_40k_levircd.py 2
#bash tools/dist_train.sh configs/pcam/pcam_r18_512x512_40k_levircd.py 2