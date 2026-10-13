conda activate lingbotvla

# 生成归一化统计文件的命令
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm.py configs/vla/so101_load20000h.yaml

# 训练命令
PYTHONPATH=. bash train.sh tasks/vla/train_lingbotvla.py configs/vla/so101_load20000h.yaml

# 开环评估

conda activate lingbotvla

export QWEN25_PATH=/home/zy/Downloads/models/Qwen2.5-VL-3B-Instruct

python scripts/open_loop_eval.py \
    --model_path /home/zy/workspace/lingbot-vla/output/so101/checkpoints/global_step_15000/hf_ckpt \
    --data_path /home/zy/.cache/huggingface/lerobot/ZhangYong/lerobot_dataset_a_20260921_164625 \
    --norm_path /home/zy/workspace/lingbot-vla/assets/norm_stats/so101.json \
    --traj_ids 0 1 2 3 4 \
    --use_length 50 \
    --num_denoising_step 10 \
    --save_plot_path ./open_loop_results/

# 真机部署
## 服务端部署

export TOKENIZERS_PARALLELISM=false
export QWEN25_PATH=/home/zy/Downloads/models/Qwen2.5-VL-3B-Instruct
python -m deploy.lingbot_vla_policy \
 --model_path /home/zy/workspace/lingbot-vla/output/so101/checkpoints/global_step_15000/hf_ckpt \
 --use_compile \
 --use_length 25 \
 --num_denoising_step 10 \
 --port 8006

 ## 客户端部署