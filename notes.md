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

这条命令只请求推理并打印预测动作，不会发送动作给机械臂，但会连接机械臂读取传感器和相机。如果服务端不在客户端这台机器上，把 --host 换成服务端的局域网 IP。若要使用文档中的远程 TLS 服务，则填写 --host "wss://服务器域名" 和对应的 --port；客户端已支持这种连接方式。

conda activate lingbotvla
cd /home/zy/workspace/lingbot-vla

python -m deploy.so101_websocket_client \
  --host 127.0.0.1 \
  --port 8006 \
  --robot-port /dev/ttyACM0 \
  --robot-id zihao_follower_arm \
  --side-camera /dev/video2 \
  --side-camera-width 1920 \
  --side-camera-height 1080 \
  --side-camera-fps 30 \
  --side-camera-fourcc MJPG \
  --wrist-camera /dev/video0 \
  --wrist-camera-width 1280 \
  --wrist-camera-height 720 \
  --wrist-camera-fps 30 \
  --wrist-camera-fourcc MJPG \
  --task "pick up a tape and put it into a drawer"

确认预测正常后，只执行动作块的第一步：

# 在上面命令末尾追加：
python -m deploy.so101_websocket_client --host 127.0.0.1 --port 8006 --robot-port /dev/ttyACM0 --robot-id zihao_follower_arm --side-camera /dev/video2 --side-camera-width 1920 --side-camera-height 1080 --side-camera-fps 30 --side-camera-fourcc MJPG  --wrist-camera /dev/video0 --wrist-camera-width 1280 --wrist-camera-height 720 --wrist-camera-fps 30 --wrist-camera-fourcc MJPG --task "pick up a tape and put it into a drawer" --execute --steps-per-chunk 1 --max-chunks 1

确认机械臂运动方向和幅度合适后，可用 --execute --steps-per-chunk 25 --max-chunks 0 持续运行，按 Ctrl+C 停止。单步执行仍会让机械臂移动，请先清空运动范围并保持随时停止的条件。

python -m deploy.so101_websocket_client --host 127.0.0.1 --port 8006 --robot-port /dev/ttyACM0 --robot-id zihao_follower_arm --side-camera /dev/video2 --side-camera-width 1920 --side-camera-height 1080 --side-camera-fps 30 --side-camera-fourcc MJPG  --wrist-camera /dev/video0 --wrist-camera-width 1280 --wrist-camera-height 720 --wrist-camera-fps 30 --wrist-camera-fourcc MJPG --task "pick up a tape and put it into a drawer" --execute --steps-per-chunk 25 --max-chunks 0