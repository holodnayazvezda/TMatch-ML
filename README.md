# TMatch-ML

ML-сервис рекомендаций для TMatch на Python 3.12 и FastAPI. Формирует кандидатов по транзакционным признакам и взаимодействиям, использует FAISS для поиска и CatBoost для ранжирования. Включает объяснения рекомендаций, ограничения подбора, сбор признаков и подготовку офлайн-артефактов.

## Запуск через Docker Compose

```sh
cp .env.example .env
docker compose up --build
```

Заполните `.env`: подключение к PostgreSQL, порт сервера и собственный `REBUILD_API_KEY`. Compose запускает PostgreSQL, отдельную задачу подготовки артефактов и API. По умолчанию API доступен на `http://localhost:8080`, описание — `/docs`, готовность — `/health/ready`.

Для полной подготовки рекомендаций нужны исходные данные и артефакты, соответствующие путям в `src/core/data_config.py`: входные данные в `data_in/`, результаты в `data_out/`. Образцы CSV и ноутбук в репозитории не заменяют весь набор для развёртывания. Каталог `data/` также монтируется конфигурацией Compose.

## Локальная разработка

Требуются Python 3.12+, uv и PostgreSQL с настройками из `.env`.

```sh
uv sync
uv run build-offline-artifacts --all
uv run check-offline-artifacts-ready
uv run api
```

Для отдельного обучения моделей предусмотрена команда `uv run train-models`. Настройки признаков и путей находятся в `src/core/`, конфигурация сервера — в `src/ml/config.py`.

## Устройство проекта

- `src/ml/candidates/`, `retrieval.py`, `reranker.py` — кандидаты, поиск и ранжирование.
- `src/ml/services/` — признаки, события, модели и офлайн-артефакты.
- `src/ml/presentation/` — HTTP API и проверки состояния.
- [Контракт интеграции ML и бэкенда](contracts/ML%20Backend%20Integration%20Contract.md).
- [Контракты API](contracts/MVP%20Backend%20API%20Contracts.md).
- [Модель предметной области](contracts/Domain%20Model%20v1.md).

Подробные схемы запросов и взаимодействие с основным бэкендом описаны в контрактах.
