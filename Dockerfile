FROM python:3.11-slim
WORKDIR /app
COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "rank.py", "--candidates", "data/candidates.jsonl", \
     "--team-id", "team_042", "--out", "submission.csv"]
