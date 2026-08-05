FROM python:3.13-slim
# this is like running in the terminal
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN uv venv
RUN uv pip install --no-cache-dir .
CMD ["uv", "run", "-m", "uvicorn", "classifier.app:app", "--host", "0.0.0.0", "--port", "8000"]
