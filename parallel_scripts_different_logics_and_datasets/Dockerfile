FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    transformers \
    peft \
    datasets \
    accelerate \
    sentencepiece

COPY . /app

CMD ["bash", "-lc", "python train_lora.py --bs 4 --grad-accum 4 --out /tmp/out_lora"]