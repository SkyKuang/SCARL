CUDA_VISIBLE_DEVICES=2 python eval.py \
    --init_model_pass=SCARL \
    --attack_method_list=natural \
    --dataset=cifar10 \
    --test_batch=100 \
    --net_type=pre-res \
    --activation='ReLU' \
    --depth=18 \
    --widen_factor=10 \
    --net_module='AT' \
    --benchmark=True \
    --save_name=cifar10-SCARL

