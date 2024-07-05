FROM python:3.11-slim

RUN apt-get update && apt-get install --no-install-recommends --yes build-essential

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

CMD ["python"]
