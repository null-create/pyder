FROM python:3.12-slim
ENV PYTHONUNBUFFERED=0

WORKDIR /app
COPY . /app

# Update the system
RUN apt-get update -y && apt-get upgrade -y 
RUN pip install -r requirements.txt --no-cache

EXPOSE 55555

CMD ["python", "crawler.py"]