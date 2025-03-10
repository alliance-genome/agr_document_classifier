FROM python:3.11-slim

ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
WORKDIR /usr/src/app
ADD ./requirements.txt .
ADD abc_utils.py .
ADD agr_document_classifier.py .
ADD dataset_downloader.py .
ADD dataset_upload_from_csv.py .
ADD Makefile .
ADD models.py .
ADD crontab /etc/cron.d/agr_document_classifier_crontab
RUN chmod 0644 /etc/cron.d/agr_document_classifier_crontab
RUN apt-get update && apt-get install --no-install-recommends --yes build-essential git cron
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
RUN crontab /etc/cron.d/agr_document_classifier_crontab
#CMD /bin/bash -c 'declare -p' | grep -Ev 'BASHOPTS|BASH_VERSINFO|EUID|PPID|SHELLOPTS|UID' > /container.env && cron && tail -f /dev/null
CMD ["/bin/bash"]