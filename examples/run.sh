# CUDA_VISIBLE_DEVICES=0,1,2,3 python  drone_offline_cfc.py --epochs 400 --test

CUDA_VISIBLE_DEVICES=0,1,2,3 python  drone_offline_ltc_1order.py --epochs 400 --train --test

CUDA_VISIBLE_DEVICES=0,1,2,3 python  drone_offline_ltc_1order.py --epochs 400 --train --test
