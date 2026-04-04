FROM python:3.12.7-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/report_agent

COPY tools/sandbox/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY tools/sandbox/runner.py /opt/report_agent/runner.py

USER nobody
