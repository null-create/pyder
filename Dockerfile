FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
  PYTHONFAULTHANDLER=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  LOGURU_LEVEL="INFO"

WORKDIR /app
COPY . /app

# system updates
RUN apt-get update -y && apt-get upgrade -y && \
  apt-get install -y --no-install-recommends curl wget && \
  rm -rf /var/lib/apt/lists/*

# install python dependencies
RUN pip install -r requirements.txt --no-cache

# minimize image
RUN apt-get autoremove -y && apt-get clean && \
  rm -rf /var/lib/apt/lists/*

CMD ["python", "crawler.py"]