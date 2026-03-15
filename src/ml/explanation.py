from collections.abc import Mapping
from typing import Any


FEATURE_REASON_MAPPER = {
    "txn_cnt": {
        "group": "activity",
        "label": "Общая активность",
        "reason": "Одинаково часто совершают финансовые операции"
    },
    "unique_transaction_cnt": {
        "group": "activity",
        "label": "Уникальные операции",
        "reason": "совершают много разных финансовых операций"
    },
    "active_days_cnt": {
        "group": "activity",
        "label": "Активные дни",
        "reason": "регулярно проявляют активность по дням"
    },
    "active_months_cnt": {
        "group": "activity",
        "label": "Активные месяцы",
        "reason": "сохраняют активность на длинном промежутке времени"
    },
    "active_weeks_cnt": {
        "group": "activity",
        "label": "Активные недели",
        "reason": "регулярно проявляют активность по неделям"
    },
    "first_txn_dttm": {
        "group": "activity_history",
        "label": "Первая транзакция",
        "reason": "имеют длительную историю финансовой активности"
    },
    "last_txn_dttm": {
        "group": "activity_history",
        "label": "Последняя транзакция",
        "reason": "были активны совсем недавно"
    },
    "activity_span_days": {
        "group": "activity_history",
        "label": "Длительность активности",
        "reason": "демонстрируют устойчивую активность в течение длительного времени"
    },
    "avg_txn_per_active_day": {
        "group": "intensity",
        "label": "Среднее число операций в активный день",
        "reason": "интенсивно совершают операции в активные дни"
    },
    "avg_txn_per_month_active": {
        "group": "intensity",
        "label": "Среднее число операций в активный месяц",
        "reason": "интенсивно совершают операции в активные месяцы"
    },
    "avg_txn_per_week_active": {
        "group": "intensity",
        "label": "Среднее число операций в активную неделю",
        "reason": "интенсивно совершают операции в активные недели"
    },
    "dominant_hour": {
        "group": "time_profile",
        "label": "Основной час активности",
        "reason": "обычно активны в определённое время суток"
    },
    "dominant_weekday_num": {
        "group": "time_profile",
        "label": "Основной день недели",
        "reason": "чаще проявляют активность в определённые дни недели"
    },
    "dominant_weekday": {
        "group": "time_profile",
        "label": "Основной день недели",
        "reason": "чаще проявляют активность в определённый день недели"
    },
    "share_weekend_txn": {
        "group": "time_profile",
        "label": "Активность в выходные",
        "reason": "чаще совершают операции в выходные"
    },
    "share_weekday_txn": {
        "group": "time_profile",
        "label": "Активность в будни",
        "reason": "чаще совершают операции в будние дни"
    },
    "share_night_txn": {
        "group": "time_profile",
        "label": "Ночная активность",
        "reason": "чаще совершают операции ночью"
    },
    "share_morning_txn": {
        "group": "time_profile",
        "label": "Утренняя активность",
        "reason": "чаще совершают операции утром"
    },
    "share_afternoon_txn": {
        "group": "time_profile",
        "label": "Дневная активность",
        "reason": "чаще совершают операции днём"
    },
    "share_evening_txn": {
        "group": "time_profile",
        "label": "Вечерняя активность",
        "reason": "чаще совершают операции вечером"
    },
    "dominant_daypart": {
        "group": "time_profile",
        "label": "Основная часть дня",
        "reason": "обычно активны в определённую часть дня"
    },
    "max_txn_in_one_day": {
        "group": "intensity",
        "label": "Пиковая дневная активность",
        "reason": "иногда совершают очень много операций за один день"
    },
    "median_txn_per_active_day": {
        "group": "intensity",
        "label": "Медиана операций в активный день",
        "reason": "обычно совершают стабильное число операций в активные дни"
    },
    "unique_category_cnt": {
        "group": "category_behavior",
        "label": "Разнообразие категорий",
        "reason": "тратят деньги в разных сферах"
    },
    "top_category_txn_share": {
        "group": "category_behavior",
        "label": "Доминирующая категория",
        "reason": "особенно часто совершают покупки в одной основной категории"
    },
    "category_entropy": {
        "group": "category_behavior",
        "label": "Распределение категорий",
        "reason": "имеют характерный профиль трат по категориям"
    },

    "cat_share_супермаркеты": {
        "group": "category_behavior",
        "label": "Супермаркеты",
        "reason": "Одинаково часто совершают покупки в супермаркетах"
    },
    "cat_share_фаст_фуд": {
        "group": "category_behavior",
        "label": "Фастфуд",
        "reason": "Одинаково часто тратят деньги на фастфуд"
    },
    "cat_share_транспорт": {
        "group": "category_behavior",
        "label": "Транспорт",
        "reason": "Одинаково часто пользуются транспортом"
    },
    "cat_share_разные_товары": {
        "group": "category_behavior",
        "label": "Разные товары",
        "reason": "Одинаково часто совершают покупки в сегменте разных товаров"
    },
    "cat_share_топливо": {
        "group": "category_behavior",
        "label": "Топливо",
        "reason": "Одинаково часто тратят деньги на топливо"
    },
    "cat_share_рестораны": {
        "group": "category_behavior",
        "label": "Рестораны",
        "reason": "Одинаково часто посещают рестораны или кафе"
    },
    "cat_share_аптеки": {
        "group": "category_behavior",
        "label": "Аптеки",
        "reason": "Одинаково часто совершают покупки в аптеках"
    },
    "cat_share_дом_ремонт": {
        "group": "category_behavior",
        "label": "Дом и ремонт",
        "reason": "Одинаково часто тратят деньги на товары для дома и ремонта"
    },
    "cat_share_развлечения": {
        "group": "category_behavior",
        "label": "Развлечения",
        "reason": "Одинаково часто тратят деньги на развлечения"
    },
    "cat_share_сервисные_услуги": {
        "group": "category_behavior",
        "label": "Сервисные услуги",
        "reason": "Одинаково часто пользуются сервисными услугами"
    },
    "cat_share_автоуслуги": {
        "group": "category_behavior",
        "label": "Автоуслуги",
        "reason": "Одинаково часто пользуются автоуслугами"
    },
    "cat_share_связь_телеком": {
        "group": "category_behavior",
        "label": "Связь и телеком",
        "reason": "Одинаково часто тратят деньги на связь и телеком-услуги"
    },
    "cat_share_одежда_обувь": {
        "group": "category_behavior",
        "label": "Одежда и обувь",
        "reason": "Одинаково часто совершают покупки одежды и обуви"
    },
    "cat_share_красота": {
        "group": "category_behavior",
        "label": "Красота",
        "reason": "Одинаково часто тратят деньги на товары и услуги в сфере красоты"
    },
    "cat_share_медицинские_услуги": {
        "group": "category_behavior",
        "label": "Медицинские услуги",
        "reason": "Одинаково часто пользуются медицинскими услугами"
    },
    "cat_share_госсборы": {
        "group": "category_behavior",
        "label": "Госсборы",
        "reason": "Одинаково часто совершают платежи, связанные с государственными сборами"
    },
    "cat_share_аренда_авто": {
        "group": "category_behavior",
        "label": "Аренда авто",
        "reason": "Одинаково часто пользуются арендой автомобилей"
    },
    "cat_share_частные_услуги": {
        "group": "category_behavior",
        "label": "Частные услуги",
        "reason": "Одинаково часто пользуются частными услугами"
    },
    "cat_share_нко": {
        "group": "category_behavior",
        "label": "НКО",
        "reason": "Одинаково часто совершают платежи в адрес некоммерческих организаций"
    },
    "cat_share_финансовые_услуги": {
        "group": "category_behavior",
        "label": "Финансовые услуги",
        "reason": "Одинаково часто пользуются финансовыми услугами"
    },
    "cat_share_цветы": {
        "group": "category_behavior",
        "label": "Цветы",
        "reason": "Одинаково часто совершают покупки в цветочных магазинах"
    },
    "cat_share_книги": {
        "group": "category_behavior",
        "label": "Книги",
        "reason": "Одинаково часто покупают книги"
    },
    "cat_share_животные": {
        "group": "category_behavior",
        "label": "Товары для животных",
        "reason": "Одинаково часто совершают покупки товаров и услуг для животных"
    },
    "cat_share_образование": {
        "group": "category_behavior",
        "label": "Образование",
        "reason": "Одинаково часто тратят деньги в сфере образования"
    },
    "cat_share_кино": {
        "group": "category_behavior",
        "label": "Кино",
        "reason": "Одинаково часто тратят деньги на кино"
    },
    "cat_share_турагентства": {
        "group": "category_behavior",
        "label": "Турагентства",
        "reason": "Одинаково часто пользуются услугами турагентств"
    },
    "cat_share_ж_д_билеты": {
        "group": "category_behavior",
        "label": "Ж/д билеты",
        "reason": "Одинаково часто покупают железнодорожные билеты"
    },
    "cat_share_авиабилеты": {
        "group": "category_behavior",
        "label": "Авиабилеты",
        "reason": "Одинаково часто покупают авиабилеты"
    },
    "cat_share_спорттовары": {
        "group": "category_behavior",
        "label": "Спорттовары",
        "reason": "Одинаково часто покупают спортивные товары"
    },
    "cat_share_unknown": {
        "group": "category_behavior",
        "label": "Неизвестная категория",
        "reason": "Одинаково часто совершают операции в категориях без точной расшифровки"
    },
    "cat_share_other": {
        "group": "category_behavior",
        "label": "Прочие категории",
        "reason": "Одинаково часто совершают операции в прочих категориях"
    },

    "unique_mcc_cnt": {
        "group": "mcc_behavior",
        "label": "Разнообразие MCC",
        "reason": "совершают покупки у разных типов продавцов"
    },
    "top_mcc_txn_share": {
        "group": "mcc_behavior",
        "label": "Доминирующий MCC",
        "reason": "особенно Одинаково часто совершают покупки у одного типа продавцов"
    },
    "mcc_entropy": {
        "group": "mcc_behavior",
        "label": "Распределение MCC",
        "reason": "имеют характерный профиль покупок по типам продавцов"
    },

    "mcc_share_5411": {
        "group": "mcc_behavior",
        "label": "Продуктовые магазины",
        "reason": "Одинаково часто совершают покупки в продуктовых магазинах"
    },
    "mcc_share_5814": {
        "group": "mcc_behavior",
        "label": "Фастфуд",
        "reason": "Одинаково часто совершают покупки в заведениях быстрого питания"
    },
    "mcc_share_5499": {
        "group": "mcc_behavior",
        "label": "Продукты и specialty food",
        "reason": "Одинаково часто совершают покупки в продуктовых и специализированных продовольственных магазинах"
    },
    "mcc_share_3990": {
        "group": "mcc_behavior",
        "label": "Прочая розница/услуги",
        "reason": "Одинаково часто совершают покупки у продавцов из прочих розничных или сервисных сегментов"
    },
    "mcc_share_4131": {
        "group": "mcc_behavior",
        "label": "Автобусные перевозки",
        "reason": "Одинаково часто пользуются автобусным транспортом"
    },
    "mcc_share_5912": {
        "group": "mcc_behavior",
        "label": "Аптеки",
        "reason": "Одинаково часто совершают покупки в аптеках"
    },
    "mcc_share_5541": {
        "group": "mcc_behavior",
        "label": "АЗС",
        "reason": "Одинаково часто совершают покупки на автозаправках"
    },
    "mcc_share_5921": {
        "group": "mcc_behavior",
        "label": "Алкогольные магазины",
        "reason": "Одинаково часто совершают покупки в специализированных магазинах напитков"
    },
    "mcc_share_5812": {
        "group": "mcc_behavior",
        "label": "Рестораны",
        "reason": "Одинаково часто посещают рестораны и кафе"
    },
    "mcc_share_4111": {
        "group": "mcc_behavior",
        "label": "Локальный транспорт",
        "reason": "Одинаково часто пользуются локальным пассажирским транспортом"
    },
    "mcc_share_5300": {
        "group": "mcc_behavior",
        "label": "Оптовые клубы / warehouse",
        "reason": "Одинаково часто совершают покупки в крупных универсальных торговых точках"
    },
    "mcc_share_4121": {
        "group": "mcc_behavior",
        "label": "Такси",
        "reason": "Одинаково часто пользуются такси"
    },
    "mcc_share_5200": {
        "group": "mcc_behavior",
        "label": "Товары для дома",
        "reason": "Одинаково часто покупают товары для дома и ремонта"
    },
    "mcc_share_7995": {
        "group": "mcc_behavior",
        "label": "Азартные/развлекательные заведения",
        "reason": "Одинаково часто тратят деньги в развлекательных заведениях"
    },
    "mcc_share_5331": {
        "group": "mcc_behavior",
        "label": "Универсальные магазины",
        "reason": "Одинаково часто совершают покупки в универсальных магазинах"
    },
    "mcc_share_5993": {
        "group": "mcc_behavior",
        "label": "Табачные магазины",
        "reason": "Одинаково часто совершают покупки в специализированных табачных магазинах"
    },
    "mcc_share_7542": {
        "group": "mcc_behavior",
        "label": "Автомойки",
        "reason": "Одинаково часто пользуются услугами автомоек"
    },
    "mcc_share_5964": {
        "group": "mcc_behavior",
        "label": "Прямой маркетинг",
        "reason": "Одинаково часто совершают покупки через direct marketing и дистанционные продажи"
    },
    "mcc_share_4900": {
        "group": "mcc_behavior",
        "label": "Коммунальные услуги",
        "reason": "Одинаково часто совершают платежи за коммунальные услуги"
    },
    "mcc_share_5816": {
        "group": "mcc_behavior",
        "label": "Цифровые товары",
        "reason": "Одинаково часто покупают цифровые товары или онлайн-контент"
    },
    "mcc_share_3991": {
        "group": "mcc_behavior",
        "label": "Прочая розница/услуги",
        "reason": "Одинаково часто совершают покупки у продавцов из прочих категорий"
    },
    "mcc_share_5815": {
        "group": "mcc_behavior",
        "label": "Цифровые сервисы",
        "reason": "Одинаково часто пользуются цифровыми сервисами"
    },
    "mcc_share_5311": {
        "group": "mcc_behavior",
        "label": "Универмаги",
        "reason": "Одинаково часто совершают покупки в универмагах"
    },
    "mcc_share_5462": {
        "group": "mcc_behavior",
        "label": "Пекарни",
        "reason": "Одинаково часто совершают покупки в пекарнях"
    },
    "mcc_share_7512": {
        "group": "mcc_behavior",
        "label": "Прокат авто",
        "reason": "Одинаково часто пользуются прокатом автомобилей"
    },
    "mcc_share_4814": {
        "group": "mcc_behavior",
        "label": "Телеком",
        "reason": "Одинаково часто совершают платежи за телеком-услуги"
    },
    "mcc_share_4816": {
        "group": "mcc_behavior",
        "label": "Компьютерные сети/онлайн-сервисы",
        "reason": "Одинаково часто пользуются онлайн- и сетевыми сервисами"
    },
    "mcc_share_5399": {
        "group": "mcc_behavior",
        "label": "Прочая розница",
        "reason": "Одинаково часто совершают покупки в прочих розничных магазинах"
    },
    "mcc_share_5977": {
        "group": "mcc_behavior",
        "label": "Косметика",
        "reason": "Одинаково часто покупают косметику и сопутствующие товары"
    },
    "mcc_share_5533": {
        "group": "mcc_behavior",
        "label": "Автозапчасти",
        "reason": "Одинаково часто покупают автозапчасти и автомобильные товары"
    },
    "mcc_share_unknown": {
        "group": "mcc_behavior",
        "label": "Неизвестный MCC",
        "reason": "Одинаково часто совершают операции у продавцов без точной MCC-расшифровки"
    },
    "mcc_share_other": {
        "group": "mcc_behavior",
        "label": "Прочие MCC",
        "reason": "Одинаково часто совершают операции у прочих типов продавцов"
    },

    "unique_merchant_cnt": {
        "group": "merchant_behavior",
        "label": "Разнообразие продавцов",
        "reason": "совершают покупки у большого числа разных продавцов"
    },
    "top_merchant_txn_share": {
        "group": "merchant_behavior",
        "label": "Любимый продавец",
        "reason": "особенно часто возвращаются к одному и тому же продавцу"
    },
    "merchant_diversity_ratio": {
        "group": "merchant_behavior",
        "label": "Доля разнообразия продавцов",
        "reason": "распределяют покупки между разными продавцами"
    },
    "category_diversity_ratio": {
        "group": "category_behavior",
        "label": "Разнообразие категорий",
        "reason": "распределяют траты между разными категориями"
    },
    "mcc_diversity_ratio": {
        "group": "mcc_behavior",
        "label": "Разнообразие MCC",
        "reason": "распределяют покупки между разными типами продавцов"
    }
}


def get_feature_label(feature_name: Any) -> str:
    key = str(feature_name)
    raw_meta = FEATURE_REASON_MAPPER.get(key)
    if isinstance(raw_meta, Mapping):
        label = raw_meta.get("label")
        if label is not None:
            label_text = str(label).strip()
            if label_text:
                return label_text
    return key


def map_feature_impacts_to_labels(raw_reasons: Any) -> dict[str, float]:
    feature_impacts = _normalize_feature_impacts(raw_reasons)
    if not feature_impacts:
        return {}

    labels_with_impacts: dict[str, float] = {}
    for feature_name, impact in feature_impacts.items():
        label = get_feature_label(feature_name)
        current = labels_with_impacts.get(label)
        if current is None or impact > current:
            labels_with_impacts[label] = impact
    return labels_with_impacts


def _normalize_feature_impacts(raw_reasons: Any) -> dict[str, float]:
    if raw_reasons is None:
        return {}

    if isinstance(raw_reasons, Mapping):
        normalized: dict[str, float] = {}
        for key, value in raw_reasons.items():
            normalized_key = str(key)
            try:
                normalized[normalized_key] = float(value)
            except (TypeError, ValueError):
                normalized[normalized_key] = 0.0
        return normalized

    if isinstance(raw_reasons, list):
        return {str(feature_name): 0.0 for feature_name in raw_reasons}

    return {str(raw_reasons): 0.0}
