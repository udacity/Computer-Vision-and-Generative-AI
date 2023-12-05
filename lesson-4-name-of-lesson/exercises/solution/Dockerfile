FROM python:3.10-slim
ENV PYTHONUNBUFFERED 1

RUN adduser --disabled-password --gecos '' --uid 1001 student

WORKDIR /home/student

RUN apt-get update && apt-get install -y \
    git \
    wget \
    libopencv-dev \
    python3-opencv \
 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/SysCV/sam-hq.git

WORKDIR /home/student/sam-hq

RUN pip install --no-cache-dir timm numpy matplotlib opencv-python-headless

ENV PYTHONPATH /home/student/sam-hq

RUN mkdir pretrained_checkpoint && \
    wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth -O pretrained_checkpoint/sam_hq_vit_l.pth

RUN chown -R student:student /home/student

USER student

COPY --chown=student:student vision_transformer_sam.py /home/student/

CMD ["python", "/home/student/vision_transformer_sam.py"]
