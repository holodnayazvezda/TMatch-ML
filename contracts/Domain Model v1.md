# Модель домена v1

## Сущности

User
* id: UserId
* created_at: datetime
* deleted_at: datetime | null
* profile: Profile | null

AuthIdentity
* id: AuthIdentityId
* created_at: datetime
* deleted_at: datetime | null
* user_id: UserId
* method: AuthMethod
* identifier: str
* secret_key: str | null

AccessToken
* id: AccessTokenId
* created_at: datetime
* deleted_at: datetime | null
* user_id: UserId
* revoked: bool
* expires_in: datetime

NotificationDevice
* id: NotificationDeviceId
* created_at: datetime
* deleted_at: datetime | null
* user_id: UserId
* device_id: str

Interaction
* id: InteractionId
* created_at: datetime
* deleted_at: datetime | null
* actor_user_id: UserId
* candidate_user_id: UserId
* action: InteractionType
* ml_recommendation_id: str | null

DatingProfile
* id: DatingProfileId
* created_at: datetime
* deleted_at: datetime | null
* user_id: UserId
* photos: list[str]
* traits: list[UserTrait]

Recommendation
* id: RecommendationId
* created_at: datetime
* deleted_at: datetime | null
* ml_recommendation_id: str
* user_id: UserId
* candidate_user_id: UserId
* reasons: list[RecommendationReason]

AuditEvent
* id: AuditEventId
* created_at: datetime
* deleted_at: datetime | null
* event_type: AuditEventType
* actor_user_id: UserId | null
* target_user_id: UserId | null
* payload: dict | null
* sent_to_ml: bool

## Value Object

Profile
* first_name: str
* last_name: str | null
* birth_date: date
* region: str | null
* avatar_url: str | null

UserTrait
* trait_code: str
* score: float
* is_hidden: bool

RecommendationReason
* score: float
* reason_type: CompatibilityType

## Enum

AuthMethod
* email
* yandex

InteractionType
* like
* dislike

CompatibilityType
* lifestyle

AuditEventType
* interaction.created
* profile.updated

## Связи

* User 1 -> N AuthIdentity
* User 1 -> N AccessToken
* User 1 -> N NotificationDevice
* User 1 -> N Interaction (actor_user_id)
* Interaction N -> 1 User (candidate_user_id)
* User 1 -> N Recommendation (user_id)
* Recommendation N -> 1 User (candidate_user_id)
* User 1 -> N AuditEvent (actor_user_id)
* AuditEvent N -> 1 User (target_user_id)
* User 1 -> 1 Profile (внутри User как VO)
* User 1 -> 1 DatingProfile (user_id)

## Инварианты

* Для получения рекомендаций нужна готовая dating-анкета (существует и минимум 1 фото).
* Создать dating-анкету можно только при наличии Profile; при создании/обновлении анкеты обязательно минимум 1 фото.
* Трейты добавляет/пересчитывает только ML; пользователь может только скрывать/показывать (is_hidden).
* Profile хранится как VO внутри User.
* После создания профиля изменяем только avatar_url; first_name, last_name, birth_date, region неизменяемы.
* При upsert профиля создается audit event profile.updated.
* Каждая Recommendation хранит ml_recommendation_id, полученный от ML.
* История Interaction append-only: старые записи не обновляются и не удаляются.
* Для одной пары пользователей допускается любое количество Interaction.
* Эффективное состояние пары определяется последним Interaction.
* Если последнее действие dislike, кандидат исключается из выдачи.
* Если последнее действие like, кандидат может снова попасть в выдачу по правилам ранжирования.
* Self-interaction запрещен (actor_user_id != candidate_user_id).
* Связь интеракции с рекомендацией опциональна (ml_recommendation_id может быть null).
* История AuditEvent append-only: старые записи не обновляются и не удаляются.
* Отправка в ML идет батчами через AuditEvent (`sent_to_ml=false` -> отправка -> `sent_to_ml=true`).

## Вне текущего объема

* Классический event sourcing как source of truth.
* Отдельная проекция PairState.
* Доменная сущность Transaction.
