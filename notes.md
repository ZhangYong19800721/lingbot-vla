conda activate lingbotvla

# 生成归一化统计文件的命令
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm.py configs/vla/so101_load20000h.yaml

# 训练命令
PYTHONPATH=. bash train.sh tasks/vla/train_lingbotvla.py configs/vla/so101_load20000h.yaml