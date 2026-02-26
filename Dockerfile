FROM python:3.12-slim

WORKDIR /app

COPY . .

# Install normal Python deps from PyPI
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    numpy

# Install CPU-only PyTorch separately
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]