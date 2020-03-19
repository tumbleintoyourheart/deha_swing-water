FROM python:3.7-alpine
RUN apt-get update -y
COPY . /app
WORKDIR /app
ENTRYPOINT ["python"]
CMD ["server.py"]