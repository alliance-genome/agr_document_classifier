FROM python:3.8-slim

RUN apt-get update && apt-get install --no-install-recommends --yes build-essential

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python"]
