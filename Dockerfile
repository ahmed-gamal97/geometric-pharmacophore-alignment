FROM python:3.11-slim

WORKDIR /app

# Install build tools needed by some scipy/rdkit wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY solve.py .

# Expects:
#   /root/data/targets.json   (mounted or copied in)
#   /root/results/            (output directory)
CMD ["python", "solve.py"]
