CUDA_VISIBLE_DEVICES=0 python3 train.py --train \
                                        --pred \
                                        --batch_size 32 \
                                        --n_epochs 10 \
                                        --hidden_size 768 \
                                        --finetuning \
                                        --early_stopping_step 15 \
                                        --dropout_rate 0.1
