#python tools/train.py configs/pcam/pcam_r18_512x512_40k_levircd.py --work-dir ./pcam_r18_levir_workdir --gpu-id 3 --seed 307
python tools/train.py configs/test/test_r18_levircd.py --work-dir ./test --gpu-id 3 --seed 307
