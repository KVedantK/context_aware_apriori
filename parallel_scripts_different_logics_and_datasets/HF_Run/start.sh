#!/bin/bash
# tiny web server so the Space passes Hugging Face's health check
mkdir -p /tmp/www && (cd /tmp/www && python -m http.server 7860 --bind 0.0.0.0) &

if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  # EDIT THE FLAGS HERE if you want: another model, or a quick test (add: --eval-n 50 --epochs 1)
  python train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --bs 4 --grad-accum 4 \
         --out /tmp/out_lora --hub-repo KVK2000/qwen-apriori-lora --pause-space
else
  echo "No GPU on this hardware, so nothing is training. Switch hardware to T4 small in Settings to start."
  sleep infinity
fi
