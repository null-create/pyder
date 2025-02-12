FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
  PYTHONFAULTHANDLER=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY . /app
RUN rm -rf venv .venv 

# system updates
RUN apt-get update -y && apt-get upgrade -y && \
  apt-get install -y --no-install-recommends curl wget && \
  rm -rf /var/lib/apt/lists/*

# install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  libnss3 libatk1.0-0 libatk-bridge2.0-0 \
  libx11-xcb1 libxcomposite1 libxcursor1 libxdamage1 \
  libxfixes3 libxi6 libxrandr2 \
  libgbm1 libglib2.0-0 libasound2 \
  && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt --no-cache

# minimize image
RUN apt-get autoremove -y && apt-get clean && \
  rm -rf /var/lib/apt/lists/*

CMD ["python", "crawler.py"]