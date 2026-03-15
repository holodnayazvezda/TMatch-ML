# MVP Backend API Contracts

## Scope (v1)
- Регистрация/вход.
- Заполнение и редактирование профиля.
- Получение рекомендаций.
- Лайк/дизлайк кандидата.
- Аудит действий (append-only).

## OpenAPI 3.1 (draft)

```yaml
openapi: 3.1.0
info:
  title: T-Match MVP API
  version: 0.1.0
servers:
  - url: https://api.example.com

tags:
  - name: auth
  - name: profile
  - name: dating-profile
  - name: recommendations
  - name: interactions
  - name: notifications
  - name: metrics
  - name: health

paths:
  /auth/registration/email:
    post:
      tags: [auth]
      summary: Register by email/password
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SignUpEmailRequest'
      responses:
        '201':
          description: Created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AuthResponse'
        '400':
          $ref: '#/components/responses/BadRequest'
        '409':
          description: Email already exists
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /auth/registration/yandex:
    post:
      tags: [auth]
      summary: Register with Yandex
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/YandexSignUpRequest'
      responses:
        '201':
          description: Created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AuthResponse'
        '401':
          description: Authentication failed
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /auth/signing/email:
    post:
      tags: [auth]
      summary: Sign in (email/password)
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SignInEmailRequest'
      responses:
        '200':
          description: OK
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AuthResponse'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          description: Invalid credentials
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '404':
          description: User not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /auth/signing/yandex:
    post:
      tags: [auth]
      summary: Sign in (Yandex)
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/YandexSignInRequest'
      responses:
        '200':
          description: OK
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AuthResponse'
        '401':
          description: Invalid code or authentication failed
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '404':
          description: User not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /profile:
    put:
      tags: [profile]
      summary: Create or update own profile
      description: |
        Create: all fields from body (first_name, birth_date required; last_name, region, avatar_url optional).
        Update: only `avatar_url` is applied; first_name, last_name, birth_date, region are immutable and ignored.
        On every upsert an audit event `profile.updated` is emitted.
      security:
        - BearerToken: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UpsertProfileRequest'
      responses:
        '200':
          description: Profile upserted
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ProfileResponse'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'

    get:
      tags: [profile]
      summary: Get own profile
      security:
        - BearerToken: []
      responses:
        '200':
          description: OK
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ProfileResponse'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '404':
          description: Profile not set yet
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /dating-profile/me:
    get:
      tags: [dating-profile]
      summary: Get own dating profile
      security:
        - BearerToken: []
      responses:
        '200':
          description: OK
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DatingProfileResponse'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '404':
          description: Dating profile not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
    put:
      tags: [dating-profile]
      summary: Create or update own dating profile
      description: |
        Requires bank profile. At least one photo required.
        Replaces entire photo list.
      security:
        - BearerToken: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UpsertDatingProfileRequest'
      responses:
        '200':
          description: OK
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DatingProfileResponse'
        '400':
          description: Profile required or validation (min 1 photo)
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '401':
          $ref: '#/components/responses/Unauthorized'

  /dating-profile/me/photos:
    post:
      tags: [dating-profile]
      summary: Upload dating profile photo
      description: |
        Requires bank profile. Request body is raw image bytes.
        Content-Type header required (image/jpeg, image/png, or image/webp). Max 10 MB.
        Returns URL to add to dating profile photos; removal is via PUT /dating-profile/me (replace list).
      security:
        - BearerToken: []
      requestBody:
        required: true
        content:
          image/jpeg:
            schema:
              type: string
              format: binary
          image/png:
            schema:
              type: string
              format: binary
          image/webp:
            schema:
              type: string
              format: binary
      responses:
        '200':
          description: OK
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UploadDatingPhotoResponse'
        '400':
          description: Profile required to upload dating photo
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '413':
          description: Photo too large (max 10 MB)
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '415':
          description: Invalid or missing Content-Type / invalid payload
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '503':
          description: Photo storage unavailable
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /dating-profile/me/traits/visibility:
    patch:
      tags: [dating-profile]
      summary: Set trait visibility (hide/show)
      description: |
        User can only hide or show traits; cannot add or remove.
      security:
        - BearerToken: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SetTraitVisibilityRequest'
      responses:
        '204':
          description: No content
        '401':
          $ref: '#/components/responses/Unauthorized'
        '404':
          description: Dating profile or trait not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /recommendations:
    get:
      tags: [recommendations]
      summary: Get personalized candidates for current user
      description: |
        Requires a ready dating profile (exists and has at least one photo).
      security:
        - BearerToken: []
      parameters:
        - name: limit
          in: query
          required: false
          schema:
            type: integer
            minimum: 1
            maximum: 100
            default: 20
      responses:
        '200':
          description: OK
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/RecommendationsResponse'
        '400':
          description: Dating profile not ready (min 1 photo required)
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '422':
          $ref: '#/components/responses/ValidationError'
        '503':
          description: ML provider unavailable
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /interactions:
    post:
      tags: [interactions]
      summary: Append interaction event for candidate
      description: |
        Interaction history is append-only.
        Multiple interactions for same actor->candidate are allowed.
        Effective state for recommendations is defined by latest interaction.
      security:
        - BearerToken: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateInteractionRequest'
      responses:
        '201':
          description: Created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/InteractionResponse'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '404':
          description: Candidate not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '422':
          $ref: '#/components/responses/ValidationError'

  /notifications:
    post:
      tags: [notifications]
      summary: Send push notification
      security:
        - BearerToken: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SendNotificationRequest'
      responses:
        '200':
          description: OK
        '401':
          $ref: '#/components/responses/Unauthorized'
        '404':
          description: Notification device not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /notifications/devices:
    post:
      tags: [notifications]
      summary: Register FCM device token
      security:
        - BearerToken: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/RegisterDeviceRequest'
      responses:
        '200':
          description: OK
        '401':
          $ref: '#/components/responses/Unauthorized'

  /metrics:
    get:
      tags: [metrics]
      summary: Prometheus metrics
      responses:
        '200':
          description: OK
          content:
            text/plain:
              schema:
                type: string

  /health:
    get:
      tags: [health]
      summary: Liveness probe
      responses:
        '200':
          description: OK
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthResponse'

  /ready:
    get:
      tags: [health]
      summary: Readiness probe
      responses:
        '200':
          description: OK
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthResponse'
        '503':
          description: Unavailable
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ReadyResponse'

components:
  securitySchemes:
    BearerToken:
      type: http
      scheme: bearer
      bearerFormat: JWT

  responses:
    BadRequest:
      description: Bad request
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
    Unauthorized:
      description: Unauthorized
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
    ValidationError:
      description: Validation error
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'

  schemas:
    SignUpEmailRequest:
      type: object
      additionalProperties: false
      required: [email, password]
      properties:
        email:
          type: string
          format: email
        password:
          type: string
          minLength: 8
          maxLength: 128

    SignInEmailRequest:
      type: object
      additionalProperties: false
      required: [email, password]
      properties:
        email:
          type: string
          format: email
        password:
          type: string
          minLength: 8
          maxLength: 128

    YandexSignUpRequest:
      type: object
      additionalProperties: false
      required: [code]
      properties:
        code:
          type: string
          minLength: 1

    YandexSignInRequest:
      type: object
      additionalProperties: false
      required: [code]
      properties:
        code:
          type: string
          minLength: 1

    AuthResponse:
      type: object
      additionalProperties: false
      required: [user_id, access_token]
      properties:
        user_id:
          type: string
          format: uuid
        access_token:
          type: string

    UpsertProfileRequest:
      type: object
      additionalProperties: false
      required: [first_name, birth_date]
      properties:
        first_name:
          type: string
          minLength: 1
          maxLength: 80
        last_name:
          type: string
          minLength: 1
          maxLength: 80
          nullable: true
        birth_date:
          type: string
          format: date
        region:
          type: string
          minLength: 1
          maxLength: 120
          nullable: true
        avatar_url:
          type: string
          format: uri
          maxLength: 2048
          nullable: true

    ProfileResponse:
      type: object
      additionalProperties: false
      required: [user_id, first_name, birth_date]
      properties:
        user_id:
          type: string
          format: uuid
        first_name:
          type: string
        last_name:
          type: string
          nullable: true
        birth_date:
          type: string
          format: date
        region:
          type: string
          nullable: true
        avatar_url:
          type: string
          format: uri
          nullable: true

    Profile:
      type: object
      additionalProperties: false
      required: [first_name, birth_date]
      properties:
        first_name:
          type: string
        last_name:
          type: string
          nullable: true
        birth_date:
          type: string
          format: date
        region:
          type: string
          nullable: true
        avatar_url:
          type: string
          format: uri
          nullable: true

    RecommendationsResponse:
      type: object
      additionalProperties: false
      required: [items]
      properties:
        items:
          type: array
          items:
            $ref: '#/components/schemas/Recommendation'

    Recommendation:
      type: object
      additionalProperties: false
      required: [ml_recommendation_id, candidate_user_id, reasons]
      properties:
        ml_recommendation_id:
          type: string
        candidate_user_id:
          type: string
          format: uuid
        reasons:
          type: array
          minItems: 1
          items:
            $ref: '#/components/schemas/RecommendationReason'

    RecommendationReason:
      type: object
      additionalProperties: false
      required: [score, reason_type]
      properties:
        score:
          type: number
          format: float
        reason_type:
          type: string
          enum: [lifestyle]

    UpsertDatingProfileRequest:
      type: object
      additionalProperties: false
      required: [photos]
      properties:
        photos:
          type: array
          minItems: 1
          items:
            type: string
            maxLength: 2048

    DatingProfileResponse:
      type: object
      additionalProperties: false
      required: [user_id, photos, traits]
      properties:
        user_id:
          type: string
          format: uuid
        photos:
          type: array
          items:
            type: string
        traits:
          type: array
          items:
            type: object
            required: [trait_code, score, is_hidden]
            properties:
              trait_code:
                type: string
              score:
                type: number
                format: float
              is_hidden:
                type: boolean

    SetTraitVisibilityRequest:
      type: object
      additionalProperties: false
      required: [trait_code, is_hidden]
      properties:
        trait_code:
          type: string
          minLength: 1
          maxLength: 64
        is_hidden:
          type: boolean

    UploadDatingPhotoResponse:
      type: object
      additionalProperties: false
      required: [photo_url]
      properties:
        photo_url:
          type: string
          format: uri

    CreateInteractionRequest:
      type: object
      additionalProperties: false
      required: [candidate_user_id, action]
      properties:
        candidate_user_id:
          type: string
          format: uuid
        action:
          type: string
          enum: [like, dislike]
        ml_recommendation_id:
          type: string
          nullable: true
          description: Optional link to recommendation (e.g. like/dislike from feed); block/report may have no recommendation.

    InteractionResponse:
      type: object
      additionalProperties: false
      required: [interaction_id, actor_user_id, candidate_user_id, action, created_at]
      properties:
        interaction_id:
          type: string
          format: uuid
        actor_user_id:
          type: string
          format: uuid
        candidate_user_id:
          type: string
          format: uuid
        action:
          type: string
          enum: [like, dislike]
        created_at:
          type: string
          format: date-time
        ml_recommendation_id:
          type: string
          nullable: true

    SendNotificationRequest:
      type: object
      additionalProperties: false
      required: [title, body]
      properties:
        title:
          type: string
          minLength: 1
          maxLength: 100
        body:
          type: string
          minLength: 1
          maxLength: 500

    RegisterDeviceRequest:
      type: object
      additionalProperties: false
      required: [device_id]
      properties:
        device_id:
          type: string
          minLength: 1
          maxLength: 200

    HealthResponse:
      type: object
      additionalProperties: false
      required: [status]
      properties:
        status:
          type: string
          enum: [ok]

    ReadyResponse:
      type: object
      additionalProperties: false
      required: [status]
      properties:
        status:
          type: string
          enum: [ok, unavailable]

    Error:
      type: object
      additionalProperties: false
      required: [code, message]
      properties:
        code:
          type: string
        message:
          type: string
```

## Domain Invariants (MVP)
- Пользователь без заполненного `profile` не может получать рекомендации.
- `profile` хранится в домене как VO внутри `User`; обновление профиля не создает отдельную доменную сущность.
- После создания профиля менять можно только `avatar_url`; first_name, last_name, birth_date, region неизменяемы.
- При создании/обновлении профиля создается audit event `profile.updated`.
- Каждая рекомендация содержит `ml_recommendation_id`, пришедший от ML; backend сохраняет этот id в БД вместе с Recommendation.
- Для получения рекомендаций нужна готовая dating-анкета (существует и содержит минимум 1 фото).
- Создать dating-анкету можно только при наличии банковского профиля; при создании/обновлении обязательно минимум 1 фото.
- Трейты в анкете добавляет/пересчитывает только ML; пользователь может только скрывать/показывать трейты (hide/show).
- `interaction` пишется append-only, переписывание прошлых записей запрещено.
- Для фильтрации выдачи по паре `(actor_user_id, candidate_user_id)` используется последнее interaction-событие.
- Если последнее действие `dislike`, кандидат исключается из выдачи; если последнее `like`, кандидат может появляться снова по правилам ранжирования.
- Пользователь не может отправить interaction на самого себя.
- Привязка интеракции к рекомендации опциональна: для like/dislike из ленты передают `ml_recommendation_id`; для block/report связь с рекомендацией не обязательна.
- `AuditEvent` пишется append-only.
- Отправка в ML идет пачками через `AuditEvent` и флаг `sent_to_ml`.

## Event Names (audit)
- `profile.updated`
- `interaction.created`
