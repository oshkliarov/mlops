# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

# Встановлюємо необхідні системні залежності для збірки
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копіюємо файл залежностей
COPY requirements.txt .

# Встановлюємо залежності у локальну папку (щоб потім перенести)
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runner
FROM python:3.10-slim AS runner

WORKDIR /app

# Встановлюємо git (потрібен для DVC та MLflow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копіюємо встановлені пакети з builder-стадії
COPY --from=builder /root/.local /root/.local

# Оновлюємо PATH, щоб система бачила встановлені пакети (напр., dvc, mlflow)
ENV PATH=/root/.local/bin:$PATH

# Копіюємо весь код проєкту
COPY . .

# Команда за замовчуванням
CMD ["python", "src/optimize.py"]
