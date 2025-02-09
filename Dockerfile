FROM python:3.12-slim
ENV PYTHONUNBUFFERED=0

WORKDIR /app
COPY . /app
RUN rm -rf venv .venv 

RUN apt-get update -y && apt-get upgrade -y 
RUN pip install -r requirements.txt --no-cache

CMD ["python", "crawler.py"]