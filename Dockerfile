FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY openinterest ./openinterest

RUN pip install --no-cache-dir .[ml]

EXPOSE 8000
CMD ["uvicorn", "openinterest.api:app", "--host", "0.0.0.0", "--port", "8000"]
