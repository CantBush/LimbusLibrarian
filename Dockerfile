FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md NOTICE.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY data/fixtures ./data/fixtures
COPY data/eval/gold ./data/eval/gold
RUN pip install --no-cache-dir .
ENV LIMBUS_DATA_DIR=/app/data
EXPOSE 8000
CMD ["uvicorn", "limbus_librarian.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
