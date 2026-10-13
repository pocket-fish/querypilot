FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

RUN pip install --no-cache-dir uv && useradd --create-home app

WORKDIR /app
COPY pyproject.toml uv.lock README.md .python-version ./
COPY src ./src
RUN uv sync --frozen --no-dev

RUN mkdir -p /app/data && chown -R app:app /app/data
USER app

EXPOSE 8000
CMD ["sh", "-c", "querypilot seed --if-missing && querypilot serve --host 0.0.0.0 --port 8000"]
