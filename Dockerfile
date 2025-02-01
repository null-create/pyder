FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

# Update the system
RUN apt-get update -y && apt-get install -y 
RUN pip install -r requirements.txt --no-cache

EXPOSE 55555

CMD ["python", "spamalot.py"]