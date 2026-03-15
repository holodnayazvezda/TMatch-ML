# ML ↔ Backend API Contract (Current)

Статус: `draft`, отражает текущую реализацию сервиса в `src/ml/presentation/routes/*`.
Версия: `v1-current`.

## 1. Scope

Этот контракт покрывает только три endpoint:
1. `GET /health`
2. `POST /recommendation`
3. `POST /interactions/batch`

Важно:
1. Канонический ingestion путь сейчас `POST /interactions/batch` (во множественном числе).
2. Все payload для POST endpoint передаются как `application/json`.

## 2. Сущности контракта

### 2.1 `HealthResponse`

```json
{
  "status": "ok"
}
```

Поля:
1. `status`: `string`, фиксированное значение `"ok"`.

### 2.2 `RecommendationRequest`

```json
{
  "user_id": "string"
}
```

Поля:
1. `user_id`: `string`, обязательное, непустое.

### 2.3 `RecommendationItem`

```json
{
  "ml_recommendation_id": "01JXYZ...",
  "user_id": "string",
  "candidate_user_id": "string",
  "reasons": {
    "feature_name": 0.87,
    "another_feature": 0.41
  }
}
```

Поля:
1. `ml_recommendation_id`: `string` (ULID, генерируется ML сервисом).
2. `user_id`: `string` (входной пользователь, для кого сгенерирована рекомендация).
3. `candidate_user_id`: `string` (рекомендуемый кандидат).
4. `reasons`: `object<string, number>` (feature -> impact score).

### 2.4 `AuditEvent` (вход для батч-ингеста)

```json
{
  "id": "string",
  "created_at": "2026-03-15T07:20:00Z",
  "deleted_at": null,
  "event_type": "interaction.created",
  "actor_user_id": "string",
  "target_user_id": "string",
  "payload": {
    "action": "block"
  },
  "sent_to_ml": false
}
```

Поля:
1. `id`: `string | null`.
2. `created_at`: `string(datetime) | datetime`, для `block` используется как timestamp события.
3. `deleted_at`: `string(datetime) | null`.
4. `event_type`: `string`.
5. `actor_user_id`: `string | null`.
6. `target_user_id`: `string | null`.
7. `payload`: `object | null`, action читается из `payload.action`.
8. `sent_to_ml`: `boolean`.

### 2.5 `InteractionsBatchSummary`

```json
{
  "received_total": 7,
  "processed_total": 3,
  "ignored_unknown_action_count": 1,
  "ignored_non_interaction_count": 1,
  "ignored_sent_to_ml_true_count": 1,
  "invalid_rows_count": 1,
  "grouped_users_count": 2
}
```

Поля:
1. `received_total`: количество событий в батче.
2. `processed_total`: количество реально обработанных interaction-событий.
3. `ignored_unknown_action_count`: проигнорировано из-за неизвестного `payload.action`.
4. `ignored_non_interaction_count`: проигнорировано из-за `event_type != interaction.created`.
5. `ignored_sent_to_ml_true_count`: проигнорировано, т.к. `sent_to_ml=true`.
6. `invalid_rows_count`: невалидные строки (пустые id, self-interaction, пустой payload/action, пустой created_at и т.д.).
7. `grouped_users_count`: число уникальных `actor_user_id` в обработанных событиях.

## 3. Endpoint Contract

## 3.1 `GET /health`

Назначение: liveness probe.

Response `200`:
```json
{
  "status": "ok"
}
```

## 3.2 `POST /recommendation`

Назначение: получить список рекомендаций для пользователя.

Request body: `RecommendationRequest`.

Response `200`: `array[RecommendationItem]`.

Пример ответа:
```json
[
  {
    "ml_recommendation_id": "01JP9QV7JPKK44VX6ZVE8FJ7EM",
    "user_id": "5f9c4ab0...",
    "candidate_user_id": "8348b2f3...",
    "reasons": {
      "trx_cosine_similarity": 0.91,
      "same_region": 0.52
    }
  }
]
```

Ошибки:
1. `400 Bad Request` — бизнес-ошибка при обработке рекомендаций.
   Формат:
   ```json
   { "detail": "..." }
   ```
2. `422 Unprocessable Entity` — schema validation error.
3. `500 Internal Server Error` — внутренняя ошибка обработки.
   Формат:
   ```json
   { "detail": "Failed to process request: ..." }
   ```

Примечания:
1. `ml_recommendation_id` генерируется на каждую строку рекомендации.
2. Дополнительно выполняется внутреннее сохранение `recommendation_id -> source_row` (in-memory stub).
3. `reasons` всегда объект (`feature -> impact`), даже если в источнике список.
4. ML runtime-конфиг (режим реранка, top_n, ratios, candidate limits и т.д.) фиксируется внутри сервиса и не входит в API контракт с backend.

## 3.3 `POST /interactions/batch`

Назначение: batch ingestion `AuditEvent` от backend в ML.

Request body: `array[AuditEvent]`.

Нормализация/фильтрация:
1. Обрабатываются только события с `event_type == "interaction.created"`.
2. События с `sent_to_ml == true` игнорируются.
3. `action` берется из `payload.action`, нормализуется (`trim + lowercase`), алиас `ublock -> unblock`.
4. Поддерживаемые action: `block`, `skip`, `unblock`.
5. Обязательны `actor_user_id`, `target_user_id`; self-interaction (`actor == target`) не допускается.
6. Для `block` timestamp берется из `AuditEvent.created_at`.
7. После валидации выполняется только группировка по `actor_user_id` и вызов фиктивной функции инкрементального апдейта фичей пользователя.

Response `200`: `InteractionsBatchSummary`.

Ошибки:
1. `422 Unprocessable Entity` — некорректный JSON/типы верхнего уровня.
2. `500 Internal Server Error` — внутренняя ошибка обработки ingestion pipeline.

Пример запроса:
```json
[
  {
    "id": "evt-1",
    "created_at": "2026-03-15T07:20:00Z",
    "deleted_at": null,
    "event_type": "interaction.created",
    "actor_user_id": "user-a",
    "target_user_id": "user-b",
    "payload": { "action": "block" },
    "sent_to_ml": false
  },
  {
    "id": "evt-2",
    "created_at": "2026-03-15T07:21:00Z",
    "deleted_at": null,
    "event_type": "interaction.created",
    "actor_user_id": "user-a",
    "target_user_id": "user-c",
    "payload": { "action": "ublock" },
    "sent_to_ml": false
  }
]
```

## 4. Семантика полей и связь с доменной моделью

1. `ml_recommendation_id` в ML API = `ml_recommendation_id` в доменной модели backend.
2. `AuditEvent` используется как append-only транспорт событий из backend в ML.
3. `payload.action` для ingestion-контракта ожидается из множества `block|skip|unblock` (с алиасом `ublock`).
4. `sent_to_ml=true` трактуется как уже отправленное/обработанное событие и не должно попадать в повторный pipeline.

## 5. Минимальные curl-примеры

`health`:
```bash
curl -sS http://localhost:8080/health
```

`recommendation`:
```bash
curl -sS -X POST http://localhost:8080/recommendation \
  -H "Content-Type: application/json" \
  -d '{"user_id":"5f9c4ab08cac7457e9111a30e4664920607ea2c115a1433d7be98e97e64244ca"}'
```

`interactions/batch`:
```bash
curl -sS -X POST http://localhost:8080/interactions/batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "id":"evt-1",
      "created_at":"2026-03-15T07:20:00Z",
      "deleted_at":null,
      "event_type":"interaction.created",
      "actor_user_id":"user-a",
      "target_user_id":"user-b",
      "payload":{"action":"skip"},
      "sent_to_ml":false
    }
  ]'
```
