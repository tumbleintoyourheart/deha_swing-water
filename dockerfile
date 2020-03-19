FROM python:3.5
RUN apt-get update -y
COPY . /app
WORKDIR /app
ENTRYPOINT ["python3"]
CMD ["server.py"]