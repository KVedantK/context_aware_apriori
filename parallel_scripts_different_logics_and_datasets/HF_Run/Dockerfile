FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app
ENV HF_HOME=/tmp/hf
RUN pip install --no-cache-dir -U transformers peft datasets accelerate huggingface_hub
COPY . /app
RUN chmod +x /app/start.sh
CMD ["bash", "/app/start.sh"]
