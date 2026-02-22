"""
NLP Module - AI-анализ текста обращений.
Сценарий работы:
1) Попытка анализа локальной OSS LLM через Ollama.
2) Если LLM недоступна/ошиблась, используем rule-based fallback.
"""

import asyncio
import csv
import io
import json
import logging
import math
import os
import re
from collections import Counter
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Optional, cast
import osmnx as ox  # <--- ДОБАВЬ ЭТУ СТРОКУ
from collections import Counter
import re


import httpx
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
from langdetect import DetectorFactory, LangDetectException, detect

logger = logging.getLogger(__name__)
DetectorFactory.seed = 0

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_PREFERRED_MODELS = [
    "llama3.2:3b",
    "llama3.2:1b",
    "qwen2.5:3b",
    "gemma2:2b",
    "phi3:mini",
]

VALID_APPEAL_TYPES = [
    "Жалоба",
    "Смена данных",
    "Консультация",
    "Претензия",
    "Неработоспособность приложения",
    "Мошеннические действия",
    "Спам",
]
VALID_SENTIMENTS = ["Позитивный", "Нейтральный", "Негативный"]
VALID_LANGUAGES = ["KZ", "ENG", "RU"]

# Офисы компании (для расчета близости)
COMPANY_OFFICES = [
    {"name": "Алматы (Главный офис)", "lat": 43.2220, "lon": 76.8512},
    {"name": "Астана (Офис)", "lat": 51.1801, "lon": 71.4460},
    {"name": "Шымкент (Офис)", "lat": 42.3417, "lon": 69.5901},
]

CITY_ALIASES = {
    "Алматы": ["алматы", "almaty"],
    "Астана": ["астана", "astana", "нур-султан", "nur-sultan", "nursultan"],
    "Шымкент": ["шымкент", "shymkent"],
}

APPEAL_KEYWORDS = {
    "Мошеннические действия": [
        "мошен",
        "фрод",
        "взлом",
        "украл",
        "украли",
        "подозрительн",
        "подозрительная активность",
        "подозрительные операции",
        "несанкционирован",
        "неавторизован",
        "чужая транзакция",
        "чужие транзакции",
        "не совершал",
        "не совершала",
        "без моего ведома",
        "неизвестная транзакция",
        "неизвестные транзакции",
        "unknown transaction",
        "unauthorized transaction",
        "unauthorized",
        "scam",
        "fraud",
        "phishing",
        "suspicious",
        "алаяқ",
        "ұрлап",
    ],
    "Неработоспособность приложения": [
        "не работает",
        "не открывается",
        "ошибка",
        "сбой",
        "вылетает",
        "зависает",
        "завис",
        "crash",
        "crashing",
        "bug",
        "error",
        "cannot login",
        "can't login",
        "не могу войти",
        "не заходит",
    ],
    "Претензия": [
        "претензи",
        "требую",
        "требуем",
        "компенсаци",
        "верните деньги",
        "refund",
        "возместить",
    ],
    "Жалоба": [
        "жалоб",
        "недоволен",
        "отвратительно",
        "ужасно",
        "плохо",
        "безобразие",
        "service is bad",
        "bad service",
    ],
    "Смена данных": [
        "изменить",
        "сменить",
        "поменять",
        "обновить",
        "телефон",
        "номер",
        "email",
        "e-mail",
        "адрес доставки",
        "profile data",
        "update details",
    ],
    "Спам": [
        "спам",
        "рассылк",
        "реклам",
        "spam",
        "unsubscribe",
        "отписать",
        "отпишите",
    ],
    "Консультация": [
        "как",
        "подскажите",
        "интересует",
        "условия",
        "стоимость",
        "вопрос",
        "help",
        "how to",
        "information",
        "кеңес",
    ],
}

NEGATIVE_WORDS = [
    "плохо",
    "ужасно",
    "отвратительно",
    "недоволен",
    "недовольна",
    "злой",
    "зла",
    "возмущен",
    "проблема",
    "ошибка",
    "не работает",
    "cannot",
    "can't",
    "bad",
    "terrible",
]
POSITIVE_WORDS = [
    "спасибо",
    "благодарю",
    "отлично",
    "хорошо",
    "доволен",
    "довольна",
    "прекрасно",
    "замечательно",
    "thank you",
    "great",
    "perfect",
]
URGENT_WORDS = [
    "срочно",
    "немедленно",
    "asap",
    "urgent",
    "прямо сейчас",
    "сейчас",
    "today",
]
MONEY_RISK_WORDS = [
    "деньги",
    "деньг",
    "списали",
    "транзакц",
    "оплата",
    "карта",
    "счет",
    "balance",
    "payment",
]
RESOLVED_POSITIVE_HINTS = [
    "решили",
    "решена",
    "решен",
    "все в порядке",
    "всё в порядке",
    "претензий нет",
    "вопрос закрыт",
    "проблема решена",
    "issue resolved",
    "resolved",
    "fixed",
    "помогли",
    "thanks for help",
]
RESOLVED_STRONG_PHRASES = [
    "претензий нет",
    "все в порядке",
    "всё в порядке",
    "проблема решена",
    "вопрос закрыт",
    "issue resolved",
]
UNRESOLVED_ISSUE_HINTS = [
    "не работает",
    "не могу",
    "не заходит",
    "ошибка",
    "сбой",
    "вылетает",
    "зависает",
    "украли",
    "взлом",
    "подозрительн",
    "несанкционирован",
    "неавторизован",
    "cannot",
    "can't",
    "crash",
    "urgent",
    "срочно",
]
FRAUD_UNAUTHORIZED_HINTS = [
    "не совершал",
    "не совершала",
    "не совершали",
    "без моего ведома",
    "несанкционирован",
    "неавторизован",
    "unauthorized",
    "unknown transaction",
]
FRAUD_TRANSACTION_HINTS = [
    "транзакц",
    "операци",
    "перевод",
    "платеж",
    "списани",
    "списали",
    "карта",
    "счет",
    "аккаунт",
]

MANAGER_RECOMMENDATIONS = {
    "Мошеннические действия": "сразу эскалировать в antifraud и временно ограничить рисковые операции клиента",
    "Неработоспособность приложения": "передать в техподдержку L2/L3 с деталями ошибки и проверить массовость инцидента",
    "Претензия": "зафиксировать претензию, согласовать срок официального ответа и назначить ответственного",
    "Жалоба": "связаться с клиентом, снять детали проблемы и предложить план решения",
    "Смена данных": "провести верификацию клиента и выполнить обновление данных по регламенту",
    "Спам": "удалить контакт из промо-рассылок и подтвердить клиенту отписку",
    "Консультация": "дать краткий пошаговый ответ и отправить релевантную инструкцию",
}

APPEAL_TYPE_ALIASES = {
    "жалоба": "Жалоба",
    "complaint": "Жалоба",
    "смена данных": "Смена данных",
    "изменение данных": "Смена данных",
    "data change": "Смена данных",
    "consultation": "Консультация",
    "консультация": "Консультация",
    "претензия": "Претензия",
    "claim": "Претензия",
    "неработоспособность приложения": "Неработоспособность приложения",
    "app issue": "Неработоспособность приложения",
    "outage": "Неработоспособность приложения",
    "мошеннические действия": "Мошеннические действия",
    "fraud": "Мошеннические действия",
    "scam": "Мошеннические действия",
    "спам": "Спам",
    "spam": "Спам",
}
SENTIMENT_ALIASES = {
    "позитивный": "Позитивный",
    "positive": "Позитивный",
    "нейтральный": "Нейтральный",
    "neutral": "Нейтральный",
    "негативный": "Негативный",
    "negative": "Негативный",
}
LANGUAGE_ALIASES = {
    "ru": "RU",
    "rus": "RU",
    "russian": "RU",
    "eng": "ENG",
    "en": "ENG",
    "english": "ENG",
    "kz": "KZ",
    "kk": "KZ",
    "kaz": "KZ",
    "kazakh": "KZ",
}

ANALYSIS_PROMPT_TEMPLATE = """Ты — система анализа клиентских обращений. Проанализируй обращение и верни ТОЛЬКО валидный JSON.

Обращение: "{text}"

Формат JSON:
{{
  "appeal_type": "<одна из: Жалоба, Смена данных, Консультация, Претензия, Неработоспособность приложения, Мошеннические действия, Спам>",
  "sentiment": "<одна из: Позитивный, Нейтральный, Негативный>",
  "priority": <число от 1 до 10>,
  "language": "<одна из: KZ, ENG, RU>",
  "summary": "<1-2 предложения: суть + рекомендация менеджеру>",
  "address": "<адрес из текста если есть, иначе null>"
}}

Правила:
- priority 8-10: мошенничество, критические инциденты, срочные жалобы
- priority 5-7: претензии и обычные жалобы
- priority 1-4: консультации, смена данных, спам
- Если язык не определен, верни RU
- Без markdown и пояснений, только JSON"""

GEOCODE_CACHE: dict[str, Optional[dict[str, Any]]] = {}
MODEL_DIR = Path(__file__).parent / "models"
TRAINED_MODEL_PATH = MODEL_DIR / "appeals_nb_model.json"
TRAINED_MODEL_VERSION = 2
MODEL_TOKEN_PATTERN = re.compile(r"[a-zA-Zа-яА-ЯёЁәіңғүұқөһӘІҢҒҮҰҚӨҺ0-9]{2,}")
COORDINATE_PAIR_PATTERN = re.compile(
    r"(?P<first>-?\d{1,3}(?:[.,]\d+)?)\s*[,;]\s*(?P<second>-?\d{1,3}(?:[.,]\d+)?)"
)
TOKEN_PROFILE_DEFAULT = "default"
TOKEN_PROFILE_INTENT = "intent_v2"
TOKEN_PROFILE_SENTIMENT = "sentiment_v2"
MAX_HINT_TOKEN_REPETITIONS = 2
INTENT_NB_ALPHA = 1.2
INTENT_NB_USE_CLASS_WEIGHTS = True
SENTIMENT_NB_ALPHA = 2.0
SENTIMENT_NB_USE_CLASS_WEIGHTS = False
INTENT_HINT_TOKEN_BY_TYPE = {
    "Жалоба": "intent_hint:complaint",
    "Смена данных": "intent_hint:data_change",
    "Консультация": "intent_hint:consultation",
    "Претензия": "intent_hint:claim",
    "Неработоспособность приложения": "intent_hint:app_issue",
    "Мошеннические действия": "intent_hint:fraud",
    "Спам": "intent_hint:spam",
}
SENTIMENT_HINT_NEGATIVE = "sentiment_hint:negative"
SENTIMENT_HINT_POSITIVE = "sentiment_hint:positive"
ADDRESS_STATUS_FOUND = "found"
ADDRESS_STATUS_NOT_FOUND = "not_found"
ADDRESS_STATUS_NOT_PROVIDED = "not_provided"

try:
    TRAINED_INTENT_MIN_CONFIDENCE = float(
        os.getenv("TRAINED_INTENT_MIN_CONFIDENCE", "0.60")
    )
except ValueError:
    TRAINED_INTENT_MIN_CONFIDENCE = 0.60
try:
    TRAINED_SENTIMENT_MIN_CONFIDENCE = float(
        os.getenv("TRAINED_SENTIMENT_MIN_CONFIDENCE", "0.50")
    )
except ValueError:
    TRAINED_SENTIMENT_MIN_CONFIDENCE = 0.50


class TextNaiveBayesClassifier:
    """Простой multinomial Naive Bayes для текстов без внешних ML-зависимостей."""

    def __init__(
        self,
        class_doc_counts: dict[str, float],
        class_token_counts: dict[str, dict[str, float]],
        total_tokens_per_class: dict[str, float],
        vocabulary: set[str],
        total_docs: float,
        feature_profile: str = TOKEN_PROFILE_DEFAULT,
        alpha: float = 1.0,
    ):
        self.class_doc_counts = class_doc_counts
        self.class_token_counts = class_token_counts
        self.total_tokens_per_class = total_tokens_per_class
        self.vocabulary = vocabulary
        self.total_docs = float(total_docs)
        self.feature_profile = feature_profile
        self.alpha = max(1e-6, float(alpha))
        self.labels = sorted(class_doc_counts.keys())

    @classmethod
    def train(
        cls,
        texts: list[str],
        labels: list[str],
        feature_profile: str = TOKEN_PROFILE_DEFAULT,
        alpha: float = 1.0,
        use_class_weights: bool = False,
    ) -> "TextNaiveBayesClassifier":
        if len(texts) != len(labels):
            raise ValueError("Количество текстов и меток должно совпадать")
        if not texts:
            raise ValueError("Нет данных для обучения")

        raw_class_counts: Counter[str] = Counter(labels)
        if use_class_weights and raw_class_counts:
            max_class_count = max(raw_class_counts.values())
            class_weights = {
                label: math.sqrt(max_class_count / count) if count > 0 else 1.0
                for label, count in raw_class_counts.items()
            }
        else:
            class_weights = {label: 1.0 for label in raw_class_counts}

        class_doc_counts: dict[str, float] = {
            label: 0.0 for label in raw_class_counts.keys()
        }
        class_token_counts: dict[str, dict[str, float]] = {
            label: {} for label in raw_class_counts.keys()
        }
        total_tokens_per_class: dict[str, float] = {
            label: 0.0 for label in raw_class_counts.keys()
        }
        vocabulary: set[str] = set()
        total_docs = 0.0

        for text, label in zip(texts, labels):
            weight = float(class_weights.get(label, 1.0))
            class_doc_counts[label] = class_doc_counts.get(label, 0.0) + weight
            total_docs += weight

            tokens = tokenize_for_model(text, feature_profile=feature_profile)
            if tokens:
                token_counter = Counter(tokens)
                label_token_counts = class_token_counts.setdefault(label, {})
                for token, count in token_counter.items():
                    weighted_count = count * weight
                    label_token_counts[token] = (
                        label_token_counts.get(token, 0.0) + weighted_count
                    )
                    total_tokens_per_class[label] = (
                        total_tokens_per_class.get(label, 0.0) + weighted_count
                    )
                    vocabulary.add(token)

        return cls(
            class_doc_counts=class_doc_counts,
            class_token_counts=class_token_counts,
            total_tokens_per_class=total_tokens_per_class,
            vocabulary=vocabulary,
            total_docs=total_docs if total_docs > 0 else float(len(texts)),
            feature_profile=feature_profile,
            alpha=alpha,
        )

    def predict(self, text: str) -> tuple[str, float]:
        if not self.labels:
            raise ValueError("Классификатор не обучен")

        tokens = tokenize_for_model(text, feature_profile=self.feature_profile)
        if not tokens:
            tokens = ["__empty__"]

        vocab_size = max(1, len(self.vocabulary))
        labels_count = max(1, len(self.labels))
        alpha = self.alpha
        scores: dict[str, float] = {}

        for label in self.labels:
            prior = (self.class_doc_counts.get(label, 0.0) + alpha) / (
                self.total_docs + alpha * labels_count
            )
            score = math.log(prior)
            token_counts = self.class_token_counts.get(label, {})
            denominator = (
                self.total_tokens_per_class.get(label, 0.0) + alpha * vocab_size
            )
            for token in tokens:
                score += math.log((token_counts.get(token, 0.0) + alpha) / denominator)
            scores[label] = score

        best_label = max(scores, key=lambda item: scores[item])
        max_score = max(scores.values())
        normalizer = sum(
            math.exp(current_score - max_score) for current_score in scores.values()
        )
        confidence = (
            math.exp(scores[best_label] - max_score) / normalizer if normalizer else 0.0
        )
        return best_label, round(confidence, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_doc_counts": dict(self.class_doc_counts),
            "class_token_counts": {
                label: dict(counter)
                for label, counter in self.class_token_counts.items()
            },
            "total_tokens_per_class": dict(self.total_tokens_per_class),
            "vocabulary": sorted(self.vocabulary),
            "total_docs": self.total_docs,
            "feature_profile": self.feature_profile,
            "alpha": self.alpha,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TextNaiveBayesClassifier":
        class_doc_counts = {
            k: float(v) for k, v in payload.get("class_doc_counts", {}).items()
        }
        class_token_counts = {
            label: {token: float(count) for token, count in counter.items()}
            for label, counter in payload.get("class_token_counts", {}).items()
        }
        total_tokens_per_class = {
            k: float(v) for k, v in payload.get("total_tokens_per_class", {}).items()
        }
        vocabulary = set(payload.get("vocabulary", []))
        total_docs = float(payload.get("total_docs", 0))
        feature_profile = str(payload.get("feature_profile") or TOKEN_PROFILE_DEFAULT)
        alpha = float(payload.get("alpha", 1.0))
        return cls(
            class_doc_counts=class_doc_counts,
            class_token_counts=class_token_counts,
            total_tokens_per_class=total_tokens_per_class,
            vocabulary=vocabulary,
            total_docs=total_docs,
            feature_profile=feature_profile,
            alpha=alpha,
        )


TRAINED_CLASSIFIERS: dict[str, TextNaiveBayesClassifier] = {}
TRAINED_MODEL_META: dict[str, Any] = {}


def _build_intent_hint_tokens(text_lower: str) -> list[str]:
    tokens: list[str] = []
    for appeal_type, keywords in APPEAL_KEYWORDS.items():
        hits = _keyword_score(text_lower, keywords)
        if hits <= 0:
            continue
        token = INTENT_HINT_TOKEN_BY_TYPE.get(appeal_type)
        if token:
            tokens.extend([token] * min(MAX_HINT_TOKEN_REPETITIONS, hits))
    return tokens


def _build_sentiment_hint_tokens(text_lower: str) -> list[str]:
    tokens: list[str] = []
    negative_hits = _keyword_score(text_lower, NEGATIVE_WORDS)
    positive_hits = _keyword_score(text_lower, POSITIVE_WORDS)
    if negative_hits > 0:
        tokens.extend(
            [SENTIMENT_HINT_NEGATIVE] * min(MAX_HINT_TOKEN_REPETITIONS, negative_hits)
        )
    if positive_hits > 0:
        tokens.extend(
            [SENTIMENT_HINT_POSITIVE] * min(MAX_HINT_TOKEN_REPETITIONS, positive_hits)
        )
    return tokens


def tokenize_for_model(
    text: str, feature_profile: str = TOKEN_PROFILE_DEFAULT
) -> list[str]:
    normalized = _normalize_whitespace(text).lower()
    if not normalized:
        return []

    words = MODEL_TOKEN_PATTERN.findall(normalized)
    if not words:
        return []

    tokens = list(words)
    for idx in range(len(words) - 1):
        tokens.append(f"{words[idx]}_{words[idx + 1]}")
    if feature_profile == TOKEN_PROFILE_INTENT:
        tokens.extend(_build_intent_hint_tokens(normalized))
    elif feature_profile == TOKEN_PROFILE_SENTIMENT:
        tokens.extend(_build_sentiment_hint_tokens(normalized))
    return tokens


def _calc_classification_metrics(
    true_labels: list[str], predicted_labels: list[str]
) -> dict[str, Any]:
    if not true_labels or len(true_labels) != len(predicted_labels):
        return {}

    labels = sorted(set(true_labels) | set(predicted_labels))
    total = len(true_labels)
    correct = sum(
        1 for true, pred in zip(true_labels, predicted_labels) if true == pred
    )
    accuracy = correct / total if total else 0.0

    macro_f1 = 0.0
    weighted_f1 = 0.0
    per_label: dict[str, dict[str, Any]] = {}
    for label in labels:
        tp = sum(
            1
            for true, pred in zip(true_labels, predicted_labels)
            if true == label and pred == label
        )
        fp = sum(
            1
            for true, pred in zip(true_labels, predicted_labels)
            if true != label and pred == label
        )
        fn = sum(
            1
            for true, pred in zip(true_labels, predicted_labels)
            if true == label and pred != label
        )
        support = sum(1 for true in true_labels if true == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )
        macro_f1 += f1
        weighted_f1 += f1 * support
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    labels_count = len(labels) if labels else 1
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1 / labels_count, 4),
        "weighted_f1": round(weighted_f1 / total if total else 0.0, 4),
        "labels": per_label,
    }


def _build_cv_splits(
    sample_count: int,
) -> tuple[str, list[tuple[list[int], list[int]]]]:
    if sample_count <= 1:
        return "none", []

    indices = list(range(sample_count))
    if sample_count <= 120:
        splits = [
            ([idx for idx in indices if idx != test_idx], [test_idx])
            for test_idx in indices
        ]
        return "loocv", splits

    folds = 5
    splits: list[tuple[list[int], list[int]]] = []
    for fold in range(folds):
        test_indices = [idx for idx in indices if idx % folds == fold]
        train_indices = [idx for idx in indices if idx % folds != fold]
        if test_indices and train_indices:
            splits.append((train_indices, test_indices))
    return f"{folds}-fold", splits


def _evaluate_classifier_cv(
    texts: list[str],
    labels: list[str],
    feature_profile: str,
    alpha: float,
    use_class_weights: bool,
) -> dict[str, Any]:
    if len(texts) < 2 or len(set(labels)) < 2:
        return {}

    strategy, splits = _build_cv_splits(len(texts))
    if not splits:
        return {}

    predictions: list[Optional[str]] = [None] * len(texts)
    for train_indices, test_indices in splits:
        train_texts = [texts[idx] for idx in train_indices]
        train_labels = [labels[idx] for idx in train_indices]
        classifier = TextNaiveBayesClassifier.train(
            train_texts,
            train_labels,
            feature_profile=feature_profile,
            alpha=alpha,
            use_class_weights=use_class_weights,
        )
        for test_idx in test_indices:
            predicted_label, _ = classifier.predict(texts[test_idx])
            predictions[test_idx] = predicted_label

    if any(label is None for label in predictions):
        return {}

    metrics = _calc_classification_metrics(
        labels, [label or "" for label in predictions]
    )
    if not metrics:
        return {}

    metrics["strategy"] = strategy
    metrics["samples"] = len(texts)
    return metrics


def _normalize_appeal_type_label(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = _normalize_whitespace(value)
    if not normalized:
        return None
    if normalized in VALID_APPEAL_TYPES:
        return normalized
    return APPEAL_TYPE_ALIASES.get(normalized.lower())


def _normalize_sentiment_label(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = _normalize_whitespace(value)
    if not normalized:
        return None
    if normalized in VALID_SENTIMENTS:
        return normalized
    return SENTIMENT_ALIASES.get(normalized.lower())


def _resolve_csv_column(
    fieldnames: list[str], primary: str, fallbacks: tuple[str, ...] = ()
) -> Optional[str]:
    mapping = {
        _normalize_whitespace(field).lower(): field
        for field in fieldnames
        if isinstance(field, str) and field.strip()
    }
    for candidate in (primary, *fallbacks):
        key = _normalize_whitespace(candidate).lower()
        if key in mapping:
            return mapping[key]
    return None


def _save_trained_model() -> None:
    if "intent" not in TRAINED_CLASSIFIERS:
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": TRAINED_MODEL_VERSION,
        "meta": TRAINED_MODEL_META,
        "models": {
            name: classifier.to_dict()
            for name, classifier in TRAINED_CLASSIFIERS.items()
        },
    }
    TRAINED_MODEL_PATH.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def load_trained_model() -> dict[str, Any]:
    """Загрузка обученной модели с диска при старте backend."""
    global TRAINED_CLASSIFIERS, TRAINED_MODEL_META

    if not TRAINED_MODEL_PATH.exists():
        TRAINED_CLASSIFIERS = {}
        TRAINED_MODEL_META = {}
        return {"available": False, "path": str(TRAINED_MODEL_PATH), "loaded": False}

    try:
        payload = json.loads(TRAINED_MODEL_PATH.read_text(encoding="utf-8"))
        models_payload = payload.get("models", {})
        loaded: dict[str, TextNaiveBayesClassifier] = {}
        for model_name in ("intent", "sentiment"):
            model_data = models_payload.get(model_name)
            if isinstance(model_data, dict):
                loaded[model_name] = TextNaiveBayesClassifier.from_dict(model_data)

        if "intent" not in loaded:
            raise ValueError("В файле модели отсутствует intent-классификатор")

        TRAINED_CLASSIFIERS = loaded
        TRAINED_MODEL_META = payload.get("meta", {})
        return get_trained_model_status()
    except Exception as exc:
        logger.error("Failed to load trained model: %s", exc)
        TRAINED_CLASSIFIERS = {}
        TRAINED_MODEL_META = {}
        return {
            "available": False,
            "path": str(TRAINED_MODEL_PATH),
            "loaded": False,
            "error": str(exc),
        }


def get_trained_model_status() -> dict[str, Any]:
    intent_model = TRAINED_CLASSIFIERS.get("intent")
    sentiment_model = TRAINED_CLASSIFIERS.get("sentiment")
    return {
        "available": intent_model is not None,
        "path": str(TRAINED_MODEL_PATH),
        "loaded": bool(TRAINED_CLASSIFIERS),
        "trained_at": TRAINED_MODEL_META.get("trained_at"),
        "samples": TRAINED_MODEL_META.get("samples", {}),
        "metrics": TRAINED_MODEL_META.get("metrics", {}),
        "training_config": TRAINED_MODEL_META.get("training_config", {}),
        "intent_labels": intent_model.labels if intent_model else [],
        "sentiment_labels": sentiment_model.labels if sentiment_model else [],
    }


def train_models_from_csv_bytes(
    csv_bytes: bytes,
    text_column: str = "text",
    appeal_type_column: str = "appeal_type",
    sentiment_column: Optional[str] = "sentiment",
) -> dict[str, Any]:
    """Обучение моделей intent/sentiment из CSV и сохранение на диск."""
    if not csv_bytes:
        raise ValueError("Пустой CSV файл")

    decoded_csv: Optional[str] = None
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            decoded_csv = csv_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if decoded_csv is None:
        raise ValueError("Не удалось декодировать CSV. Используйте UTF-8 или CP1251.")

    reader = csv.DictReader(io.StringIO(decoded_csv))
    if not reader.fieldnames:
        raise ValueError("CSV должен содержать заголовки колонок")

    fieldnames = [field for field in reader.fieldnames if isinstance(field, str)]
    text_col = _resolve_csv_column(
        fieldnames,
        text_column,
        (
            "message",
            "body",
            "appeal",
            "request",
            "описание",
            "описание обращения",
            "текст",
            "обращение",
        ),
    )
    appeal_col = _resolve_csv_column(
        fieldnames,
        appeal_type_column,
        (
            "intent",
            "label",
            "type",
            "category",
            "тип обращения",
            "категория обращения",
            "категория",
        ),
    )
    sentiment_col = None
    if sentiment_column and sentiment_column.strip():
        sentiment_col = _resolve_csv_column(
            fieldnames,
            sentiment_column,
            ("sentiment_label", "tone", "emotion", "тональность", "эмоция"),
        )

    if not text_col:
        raise ValueError(f"Колонка текста '{text_column}' не найдена")
    if not appeal_col:
        raise ValueError(f"Колонка типа обращения '{appeal_type_column}' не найдена")

    intent_texts: list[str] = []
    intent_labels: list[str] = []
    sentiment_texts: list[str] = []
    sentiment_labels: list[str] = []
    rows_total = 0

    for row in reader:
        rows_total += 1
        raw_text = row.get(text_col)
        if raw_text is None:
            continue
        text = _normalize_whitespace(str(raw_text))
        if not text:
            continue

        normalized_appeal_type = _normalize_appeal_type_label(row.get(appeal_col))
        if normalized_appeal_type:
            intent_texts.append(text)
            intent_labels.append(normalized_appeal_type)

        if sentiment_col:
            normalized_sentiment = _normalize_sentiment_label(row.get(sentiment_col))
            if normalized_sentiment:
                sentiment_texts.append(text)
                sentiment_labels.append(normalized_sentiment)

    if len(intent_texts) < 10:
        raise ValueError(
            "Недостаточно данных для обучения intent (минимум 10 размеченных строк)"
        )
    if len(set(intent_labels)) < 2:
        raise ValueError("Для intent нужны как минимум 2 разных класса")

    trained_models: dict[str, TextNaiveBayesClassifier] = {
        "intent": TextNaiveBayesClassifier.train(
            intent_texts,
            intent_labels,
            feature_profile=TOKEN_PROFILE_INTENT,
            alpha=INTENT_NB_ALPHA,
            use_class_weights=INTENT_NB_USE_CLASS_WEIGHTS,
        )
    }
    samples = {"rows_total": rows_total, "intent_samples": len(intent_texts)}
    metrics: dict[str, Any] = {
        "intent": _evaluate_classifier_cv(
            intent_texts,
            intent_labels,
            feature_profile=TOKEN_PROFILE_INTENT,
            alpha=INTENT_NB_ALPHA,
            use_class_weights=INTENT_NB_USE_CLASS_WEIGHTS,
        )
    }

    if (
        sentiment_texts
        and len(set(sentiment_labels)) >= 2
        and len(sentiment_texts) >= 10
    ):
        trained_models["sentiment"] = TextNaiveBayesClassifier.train(
            sentiment_texts,
            sentiment_labels,
            feature_profile=TOKEN_PROFILE_SENTIMENT,
            alpha=SENTIMENT_NB_ALPHA,
            use_class_weights=SENTIMENT_NB_USE_CLASS_WEIGHTS,
        )
        samples["sentiment_samples"] = len(sentiment_texts)
        metrics["sentiment"] = _evaluate_classifier_cv(
            sentiment_texts,
            sentiment_labels,
            feature_profile=TOKEN_PROFILE_SENTIMENT,
            alpha=SENTIMENT_NB_ALPHA,
            use_class_weights=SENTIMENT_NB_USE_CLASS_WEIGHTS,
        )
    else:
        samples["sentiment_samples"] = len(sentiment_texts)

    global TRAINED_CLASSIFIERS, TRAINED_MODEL_META
    TRAINED_CLASSIFIERS = trained_models
    TRAINED_MODEL_META = {
        "trained_at": datetime.now().isoformat(),
        "source": "csv_upload",
        "columns": {
            "text": text_col,
            "appeal_type": appeal_col,
            "sentiment": sentiment_col,
        },
        "samples": samples,
        "metrics": metrics,
        "training_config": {
            "intent": {
                "feature_profile": TOKEN_PROFILE_INTENT,
                "alpha": INTENT_NB_ALPHA,
                "use_class_weights": INTENT_NB_USE_CLASS_WEIGHTS,
            },
            "sentiment": {
                "feature_profile": TOKEN_PROFILE_SENTIMENT,
                "alpha": SENTIMENT_NB_ALPHA,
                "use_class_weights": SENTIMENT_NB_USE_CLASS_WEIGHTS,
            },
        },
    }
    _save_trained_model()

    return {
        "status": "ok",
        "message": "Модель обучена и сохранена",
        **get_trained_model_status(),
    }


def predict_with_trained_model(text: str) -> Optional[dict[str, Any]]:
    intent_model = TRAINED_CLASSIFIERS.get("intent")
    if not intent_model:
        return None

    normalized = _normalize_whitespace(text)
    if not normalized:
        return None

    appeal_type, intent_confidence = intent_model.predict(normalized)
    sentiment = None
    sentiment_confidence = None
    sentiment_model = TRAINED_CLASSIFIERS.get("sentiment")
    if sentiment_model:
        sentiment, sentiment_confidence = sentiment_model.predict(normalized)

    return {
        "appeal_type": appeal_type,
        "sentiment": sentiment,
        "intent_confidence": intent_confidence,
        "sentiment_confidence": sentiment_confidence,
    }


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _truncate_text(text: str, max_len: int = 180) -> str:
    text = _normalize_whitespace(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip(" ,;:") + "..."


def _normalize_choice(
    value: Any, aliases: dict[str, str], valid_values: list[str], default: str
) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in valid_values:
            return normalized
        mapped = aliases.get(normalized.lower())
        if mapped:
            return mapped
    return default


def _select_ollama_model(models: list[str]) -> Optional[str]:
    if not models:
        return None

    for preferred in [OLLAMA_MODEL] + OLLAMA_PREFERRED_MODELS:
        for available in models:
            if (
                preferred == available
                or preferred in available
                or available in preferred
            ):
                return available
    return models[0]


async def call_ollama(prompt: str, model: Optional[str] = None) -> str:
    """Вызов локального Ollama API."""
    selected_model = model or OLLAMA_MODEL
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": selected_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 512},
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")


async def check_ollama_available() -> dict:
    """Проверка доступности Ollama и списка моделей."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status()
        data = response.json()
        models = [
            item.get("name") for item in data.get("models", []) if item.get("name")
        ]
        return {"available": True, "models": models}
    except Exception as exc:
        logger.warning("Ollama not available: %s", exc)
        return {"available": False, "models": []}


def extract_json_from_response(text: str) -> dict:
    """Извлечение первого валидного JSON-объекта из ответа LLM."""
    cleaned = re.sub(r"```(?:json)?", "", (text or ""), flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()
    if not cleaned:
        return {}

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[idx:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {}


def detect_language_fallback(text: str) -> str:
    """Определение языка (KZ/ENG/RU), если не удалось - RU."""
    normalized = _normalize_whitespace(text).lower()
    if not normalized:
        return "RU"

    kz_chars = set("әіңғүұқөһ")
    cyrillic = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    latin = set("abcdefghijklmnopqrstuvwxyz")

    if set(normalized) & kz_chars:
        return "KZ"

    kz_hint_words = [
        "қалай",
        "өтінемін",
        "тапсырыс",
        "жеткізілмеді",
        "көмек",
        "мекенжай",
    ]
    if any(word in normalized for word in kz_hint_words):
        return "KZ"

    try:
        lang = detect(normalized)
        if lang == "kk":
            return "KZ"
        if lang == "en":
            return "ENG"
        if lang == "ru":
            return "RU"
    except LangDetectException:
        pass

    latin_count = sum(1 for ch in normalized if ch in latin)
    cyrillic_count = sum(1 for ch in normalized if ch in cyrillic)
    if latin_count > cyrillic_count:
        return "ENG"
    return "RU"


def _keyword_score(text_lower: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if keyword in text_lower)


def _has_strong_positive_feedback(text_lower: str) -> bool:
    positive_hits = _keyword_score(text_lower, POSITIVE_WORDS)
    negative_hits = _keyword_score(text_lower, NEGATIVE_WORDS)
    unresolved_hits = _keyword_score(text_lower, UNRESOLVED_ISSUE_HINTS)
    urgent_hits = _keyword_score(text_lower, URGENT_WORDS)
    money_hits = _keyword_score(text_lower, MONEY_RISK_WORDS)
    fraud_hits = _keyword_score(
        text_lower, APPEAL_KEYWORDS.get("Мошеннические действия", [])
    )

    return (
        positive_hits >= 2
        and negative_hits == 0
        and unresolved_hits == 0
        and urgent_hits == 0
        and money_hits == 0
        and fraud_hits == 0
    )


def _has_resolved_positive_feedback(text_lower: str) -> bool:
    positive_hits = _keyword_score(text_lower, POSITIVE_WORDS)
    resolution_hits = _keyword_score(text_lower, RESOLVED_POSITIVE_HINTS)
    unresolved_hits = _keyword_score(text_lower, UNRESOLVED_ISSUE_HINTS)
    urgent_hits = _keyword_score(text_lower, URGENT_WORDS)

    strong_resolution = any(phrase in text_lower for phrase in RESOLVED_STRONG_PHRASES)

    if strong_resolution and positive_hits > 0 and urgent_hits == 0:
        return True
    if positive_hits >= 2 and resolution_hits > 0 and unresolved_hits <= 1 and urgent_hits == 0:
        return True
    return False


def _has_strong_fraud_signal(text_lower: str) -> bool:
    if _has_resolved_positive_feedback(text_lower) or _has_strong_positive_feedback(
        text_lower
    ):
        return False

    fraud_hits = _keyword_score(
        text_lower, APPEAL_KEYWORDS.get("Мошеннические действия", [])
    )
    unauthorized_hits = _keyword_score(text_lower, FRAUD_UNAUTHORIZED_HINTS)
    transaction_hits = _keyword_score(text_lower, FRAUD_TRANSACTION_HINTS)
    money_hits = _keyword_score(text_lower, MONEY_RISK_WORDS)

    # Явные сценарии несанкционированных транзакций/активности.
    if unauthorized_hits > 0 and (transaction_hits > 0 or money_hits > 0):
        return True
    if "подозрительн" in text_lower and (transaction_hits > 0 or money_hits > 0):
        return True
    if fraud_hits > 0 and (transaction_hits > 0 or money_hits > 0):
        return True
    return False


def _detect_appeal_type(text_lower: str) -> str:
    if _has_strong_fraud_signal(text_lower):
        return "Мошеннические действия"

    scores = {
        appeal_type: _keyword_score(text_lower, words)
        for appeal_type, words in APPEAL_KEYWORDS.items()
    }
    if not any(scores.values()):
        return "Консультация"

    # Приоритет категорий при равных score.
    precedence = [
        "Мошеннические действия",
        "Неработоспособность приложения",
        "Спам",
        "Претензия",
        "Жалоба",
        "Смена данных",
        "Консультация",
    ]
    max_score = max(scores.values())
    for appeal_type in precedence:
        if scores.get(appeal_type, 0) == max_score:
            return appeal_type
    return "Консультация"


def _detect_sentiment(text_lower: str, appeal_type: str) -> str:
    if _has_resolved_positive_feedback(text_lower) or _has_strong_positive_feedback(
        text_lower
    ):
        return "Позитивный"

    negative_score = _keyword_score(text_lower, NEGATIVE_WORDS)
    positive_score = _keyword_score(text_lower, POSITIVE_WORDS)

    if negative_score > positive_score:
        return "Негативный"
    if positive_score > negative_score:
        return "Позитивный"
    if appeal_type in {
        "Мошеннические действия",
        "Неработоспособность приложения",
        "Претензия",
        "Жалоба",
    }:
        return "Негативный"
    return "Нейтральный"


def _calculate_priority(appeal_type: str, sentiment: str, text_lower: str) -> int:
    base_priority = {
        "Мошеннические действия": 9,
        "Неработоспособность приложения": 8,
        "Претензия": 7,
        "Жалоба": 6,
        "Смена данных": 3,
        "Консультация": 2,
        "Спам": 1,
    }.get(appeal_type, 5)

    urgency_hits = _keyword_score(text_lower, URGENT_WORDS)
    money_risk_hits = _keyword_score(text_lower, MONEY_RISK_WORDS)

    if sentiment == "Негативный":
        base_priority += 1
    if urgency_hits >= 2:
        base_priority += 2
    elif urgency_hits == 1:
        base_priority += 1
    if money_risk_hits > 0 and appeal_type in {
        "Мошеннические действия",
        "Неработоспособность приложения",
        "Претензия",
    }:
        base_priority += 1

    if appeal_type == "Спам":
        return 1

    priority = max(1, min(10, base_priority))
    if sentiment == "Негативный":
        priority = max(priority, 7)
    if _has_resolved_positive_feedback(text_lower) or _has_strong_positive_feedback(
        text_lower
    ):
        priority = min(priority, 4)
        if sentiment == "Позитивный":
            priority = min(priority, 3)
    return priority


def _clean_address_candidate(candidate: str) -> Optional[str]:
    cleaned = _normalize_whitespace(candidate.strip(" ,;:.\"'"))
    cleaned = re.sub(
        r"^(?:по адресу|адрес|address|мекенжай)\s*[:\-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip(" ,;:.\"'")
    if not cleaned:
        return None
    if len(cleaned) < 3 or len(cleaned) > 140:
        return None
    if cleaned.lower() in {"нет", "null", "none", "не указан"}:
        return None
    if cleaned.lower() in {"г", "г.", "город", "ул", "ул.", "улица", "д", "д.", "дом"}:
        return None
    return cleaned


def _normalize_address_component(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = _normalize_whitespace(str(value))
    if not normalized:
        return None
    return normalized


def build_structured_address_candidate(
    address_data: Optional[dict[str, Any]],
) -> Optional[str]:
    """Сборка адреса из структурированных полей."""
    if not isinstance(address_data, dict):
        return None

    for key in ("address", "full_address", "raw_address", "client_address"):
        candidate_raw = address_data.get(key)
        if isinstance(candidate_raw, str):
            cleaned = _clean_address_candidate(candidate_raw)
            if cleaned:
                return cleaned

    country = _normalize_address_component(address_data.get("country"))
    region = _normalize_address_component(address_data.get("region"))
    city = _normalize_address_component(
        address_data.get("city")
        or address_data.get("locality")
        or address_data.get("settlement")
    )
    street = _normalize_address_component(address_data.get("street"))
    house = _normalize_address_component(
        address_data.get("house") or address_data.get("building")
    )

    street_part = None
    if street and house:
        street_part = f"{street} {house}"
    elif street:
        street_part = street
    elif house:
        street_part = f"дом {house}"

    parts = [part for part in (country, region, city, street_part) if part]
    if not parts:
        return None
    return _clean_address_candidate(", ".join(parts))


def extract_coordinates_from_structured_address(
    address_data: Optional[dict[str, Any]],
) -> Optional[tuple[float, float]]:
    """Извлечение координат из структурированных полей (lat/lon)."""
    if not isinstance(address_data, dict):
        return None

    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            normalized = value.strip().replace(",", ".")
            try:
                return float(normalized)
            except ValueError:
                return None
        return None

    lat = _to_float(
        address_data.get("lat")
        or address_data.get("latitude")
        or address_data.get("client_lat")
    )
    lon = _to_float(
        address_data.get("lon")
        or address_data.get("lng")
        or address_data.get("longitude")
        or address_data.get("client_lon")
    )
    if lat is None or lon is None:
        return None
    return _normalize_coordinates(lat, lon)


def extract_address_from_text(text: str) -> Optional[str]:
    """Извлечение адреса из текста без LLM."""
    normalized = _normalize_whitespace(text)
    if not normalized:
        return None

    def _candidate_score(candidate: str) -> int:
        lowered = candidate.lower()
        has_city_alias = any(
            alias in lowered for aliases in CITY_ALIASES.values() for alias in aliases
        )
        has_street_marker = bool(
            re.search(
                r"(?:\bул\.?\b|улиц|просп|пр-т|мкр|район|street|st\.?|avenue|ave|дом|\bд\.)",
                lowered,
            )
        )
        has_digits = any(char.isdigit() for char in candidate)
        score = 0
        if has_city_alias:
            score += 2
        if has_street_marker:
            score += 2
        if has_digits:
            score += 1
        if len(candidate) >= 12:
            score += 1
        # Снижаем приоритет для "шумных" хвостов вроде "адрес доставки. новый адрес: ..."
        if "адрес доставки" in lowered:
            score -= 2
        if "новый адрес" in lowered and ":" in candidate:
            score -= 1
        return score

    candidates: list[str] = []
    cue_patterns = [
        r"(?:новый\s+адрес|адрес\s+доставки|адрес|address|мекенжай)\s*[:\-]\s*([^\n!?]{5,180})",
        r"(?:нахожусь|локация|location)\s*(?:в|in)?\s*[:\-]?\s*([^\n!?]{5,180})",
        r"(?:офисе на|по адресу)\s*[:\-]?\s*([^\n!?]{5,180})",
    ]
    for pattern in cue_patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            candidate = _clean_address_candidate(match.group(1))
            if candidate:
                candidates.append(candidate)

    city_street_pattern = (
        r"((?:г\.?\s*)?(?:алматы|астана|шымкент|almaty|astana|shymkent)"
        r"[^\n!?]{0,80}(?:ул\.?|улиц|просп|пр-т|мкр|район|street|st\.|avenue|ave|дом|д\.)[^\n!?]{0,60})"
    )
    for match in re.finditer(city_street_pattern, normalized, flags=re.IGNORECASE):
        candidate = _clean_address_candidate(match.group(1))
        if candidate:
            candidates.append(candidate)

    if candidates:
        unique_candidates: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)

        # При одинаковом score предпочитаем более короткий и точный кандидат.
        best_candidate = max(
            unique_candidates, key=lambda candidate: (_candidate_score(candidate), -len(candidate))
        )
        if _candidate_score(best_candidate) >= 2:
            return best_candidate

    text_lower = normalized.lower()
    for city, aliases in CITY_ALIASES.items():
        if any(alias in text_lower for alias in aliases):
            return city
    return None


def _has_kazakhstan_address_hint(address: str) -> bool:
    lowered = address.lower()
    if "казахстан" in lowered or "kazakhstan" in lowered:
        return True
    return any(
        alias in lowered for aliases in CITY_ALIASES.values() for alias in aliases
    )


def _extract_city_from_address(address: str) -> Optional[str]:
    lowered = address.lower()
    for city, aliases in CITY_ALIASES.items():
        for alias in aliases:
            alias_pattern = rf"(?:^|[^\w]){re.escape(alias)}(?:$|[^\w])"
            if re.search(alias_pattern, lowered):
                return city
    return None


def _extract_street_and_house(address: str) -> tuple[Optional[str], Optional[str]]:
    lowered = address.lower()
    street_pattern = re.compile(
        r"(?:ул\.?|улица|street|st\.?|просп(?:ект)?|пр-т|avenue|ave|мкр|микрорайон)\s*([^\n,!?]{2,80})",
        flags=re.IGNORECASE,
    )
    street_match = street_pattern.search(lowered)
    if not street_match:
        return None, None

    raw_street_part = _normalize_whitespace(street_match.group(1))
    if not raw_street_part:
        return None, None

    house_match = re.search(
        r"(\d{1,5}[a-zа-я]?(?:[-/]\d{1,5}[a-zа-я]?)?)",
        raw_street_part,
        flags=re.IGNORECASE,
    )
    house = house_match.group(1) if house_match else None
    if house_match:
        street_name = _normalize_whitespace(
            raw_street_part[: house_match.start()].strip(" ,.;:")
        )
    else:
        street_name = _normalize_whitespace(raw_street_part.strip(" ,.;:"))

    if street_name and len(street_name) >= 2:
        return street_name, house
    return None, house


def _build_geocode_queries(address: str) -> list[str]:
    normalized = _normalize_whitespace(address)
    city = _extract_city_from_address(normalized)
    street, house = _extract_street_and_house(normalized)
    require_kazakhstan = _has_kazakhstan_address_hint(normalized)

    queries: list[str] = []
    if city and street and house:
        queries.append(f"{city}, {street} {house}, Казахстан")
        queries.append(f"{city}, {street}, {house}, Казахстан")
    if city and street:
        queries.append(f"{city}, {street}, Казахстан")
    if (
        require_kazakhstan
        and "казахстан" not in normalized.lower()
        and "kazakhstan" not in normalized.lower()
    ):
        queries.append(f"{normalized}, Казахстан")
        queries.append(f"{normalized}, Kazakhstan")
    queries.append(normalized)

    deduplicated: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = _normalize_whitespace(query)
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        deduplicated.append(cleaned)
    return deduplicated


def _normalize_geocode_candidates(location_result: Any) -> list[Any]:
    if location_result is None:
        return []
    if isinstance(location_result, list):
        return [item for item in location_result if item is not None]
    return [location_result]


def _candidate_matches_city(location_address: str, city: Optional[str]) -> bool:
    if not city:
        return False
    lowered = location_address.lower()
    aliases = CITY_ALIASES.get(city, [])
    return any(alias in lowered for alias in aliases)


def _score_geocode_candidate(
    location: Any,
    require_kazakhstan: bool,
    city: Optional[str],
    street: Optional[str],
    house: Optional[str],
) -> int:
    address_text = str(getattr(location, "address", "")).lower()
    score = 0

    if require_kazakhstan:
        if _is_kazakhstan_geocode_result(location):
            score += 5
        else:
            return -1

    if _candidate_matches_city(address_text, city):
        score += 3
    if street and street.lower() in address_text:
        score += 3
    if house and house.lower() in address_text:
        score += 2
    return score
    return None


def _is_kazakhstan_geocode_result(location: Any) -> bool:
    raw = getattr(location, "raw", {})
    if isinstance(raw, dict):
        address_payload = raw.get("address")
        if isinstance(address_payload, dict):
            country_code = str(address_payload.get("country_code") or "").lower()
            if country_code == "kz":
                return True
            country = str(address_payload.get("country") or "").lower()
            if "казахстан" in country or "kazakhstan" in country:
                return True

    location_address = str(getattr(location, "address", "")).lower()
    return "казахстан" in location_address or "kazakhstan" in location_address


def _generate_summary(
    text: str, appeal_type: str, sentiment: str, priority: int, address: Optional[str]
) -> str:
    essence = _truncate_text(text, max_len=170)
    recommendation = MANAGER_RECOMMENDATIONS.get(
        appeal_type, "проверить детали обращения и назначить ответственного сотрудника"
    )
    urgency_hint = " немедленная реакция обязательна." if priority >= 8 else ""
    location_hint = f" Указан адрес: {address}." if address else ""
    return (
        f"Клиент сообщает: {essence}. "
        f"Тип обращения: {appeal_type}, тональность: {sentiment}, приоритет: {priority}/10; "
        f"рекомендация менеджеру: {recommendation}.{urgency_hint}{location_hint}"
    )


def rule_based_analysis(text: str) -> dict:
    """Полный fallback-анализ на правилах и OSS-библиотеках."""
    normalized = _normalize_whitespace(text)
    text_lower = normalized.lower()

    appeal_type = _detect_appeal_type(text_lower)
    sentiment = _detect_sentiment(text_lower, appeal_type)
    priority = _calculate_priority(appeal_type, sentiment, text_lower)
    language = detect_language_fallback(normalized)
    address = extract_address_from_text(normalized)
    summary = _generate_summary(normalized, appeal_type, sentiment, priority, address)

    return {
        "appeal_type": appeal_type,
        "sentiment": sentiment,
        "priority": priority,
        "language": language,
        "summary": summary,
        "address": address,
    }


def _normalize_priority(value: Any, default: int) -> int:
    if isinstance(value, int):
        return max(1, min(10, value))
    if isinstance(value, float):
        return max(1, min(10, int(round(value))))
    if isinstance(value, str):
        match = re.search(r"\d{1,2}", value)
        if match:
            return max(1, min(10, int(match.group())))
    return default


def _normalize_analysis_result(raw: dict, fallback: dict, original_text: str) -> dict:
    normalized = dict(fallback)
    if not isinstance(raw, dict):
        return normalized

    normalized["appeal_type"] = _normalize_choice(
        raw.get("appeal_type"),
        APPEAL_TYPE_ALIASES,
        VALID_APPEAL_TYPES,
        fallback["appeal_type"],
    )
    normalized["sentiment"] = _normalize_choice(
        raw.get("sentiment"), SENTIMENT_ALIASES, VALID_SENTIMENTS, fallback["sentiment"]
    )
    normalized["language"] = _normalize_choice(
        raw.get("language"), LANGUAGE_ALIASES, VALID_LANGUAGES, fallback["language"]
    )
    normalized["priority"] = _normalize_priority(
        raw.get("priority"), fallback["priority"]
    )

    raw_summary = raw.get("summary")
    if isinstance(raw_summary, str) and raw_summary.strip():
        normalized["summary"] = _normalize_whitespace(raw_summary)

    raw_address = raw.get("address")
    if isinstance(raw_address, str):
        cleaned_address = _clean_address_candidate(raw_address)
        normalized["address"] = cleaned_address
    elif raw_address is None:
        normalized["address"] = fallback.get("address")

    if not normalized.get("language"):
        normalized["language"] = detect_language_fallback(original_text)
    if normalized.get("language") not in VALID_LANGUAGES:
        normalized["language"] = "RU"

    if not normalized.get("address"):
        normalized["address"] = extract_address_from_text(original_text)

    if not normalized.get("summary"):
        normalized["summary"] = _generate_summary(
            original_text,
            normalized["appeal_type"],
            normalized["sentiment"],
            normalized["priority"],
            normalized.get("address"),
        )

    return normalized


def _normalize_coordinates(
    first: float, second: float
) -> Optional[tuple[float, float]]:
    if -90 <= first <= 90 and -180 <= second <= 180:
        return first, second
    if -90 <= second <= 90 and -180 <= first <= 180:
        return second, first
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().replace(",", ".")
        if not normalized:
            return None
        try:
            return float(normalized)
        except ValueError:
            return None
    return None


def extract_coordinates_from_text(text: str) -> Optional[tuple[float, float]]:
    """Извлечение координат вида 'lat, lon' из текста."""
    normalized = _normalize_whitespace(text)
    if not normalized:
        return None

    for match in COORDINATE_PAIR_PATTERN.finditer(normalized):
        first_raw = match.group("first").replace(",", ".")
        second_raw = match.group("second").replace(",", ".")
        try:
            first = float(first_raw)
            second = float(second_raw)
        except ValueError:
            continue

        coords = _normalize_coordinates(first, second)
        if coords:
            return coords
    return None


def build_geo_payload(
    lat: float, lon: float, formatted_address: Optional[str] = None
) -> dict[str, Any]:
    nearest_office = find_nearest_office(lat, lon)
    return {
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "formatted_address": formatted_address,
        "nearest_office": nearest_office,
    }

ox.settings.use_cache = True
ox.settings.cache_folder = './osmnx_cache'
async def geocode_address(address: str) -> Optional[dict[str, Any]]:
    """Умный геокодер через osmnx с локальным кэшированием"""
    if not address:
        return None

    normalized_address = _normalize_whitespace(address)
    if normalized_address.lower() in {"null", "none"}:
        return None

    # Попытка 1: Ищем точный адрес через osmnx
    try:
        query = f"{normalized_address}, Казахстан"
        # ox.geocode возвращает кортеж (lat, lon)
        # Первый раз сходит в сеть, дальше будет брать из кэша
        lat, lon = ox.geocode(query)
        
        return {
            "lat": lat,
            "lon": lon,
            "formatted_address": normalized_address,
            "nearest_office": find_nearest_office(lat, lon)
        }
    except Exception as exc:
        logger.warning(f"osmnx не нашел точный адрес '{query}': {exc}")

    # Попытка 2 (Fallback): Если точный адрес не найден, ищем просто город
    city = _extract_city_from_address(normalized_address)
    if city:
        try:
            # Ищем центр города
            lat, lon = ox.geocode(f"город {city}, Казахстан")
            return {
                "lat": lat,
                "lon": lon,
                "formatted_address": normalized_address,
                "nearest_office": find_nearest_office(lat, lon)
            }
        except Exception:
            pass

    return None


async def resolve_client_address(
    address_candidate: Any,
) -> tuple[Optional[str], Optional[dict[str, Any]], str]:
    """
    Возвращает только валидный (существующий) адрес.
    - found: адрес геокодирован успешно
    - not_found: адрес указан, но не найден
    - not_provided: адрес не указан
    """
    if not isinstance(address_candidate, str) or not address_candidate.strip():
        return None, None, ADDRESS_STATUS_NOT_PROVIDED

    normalized_candidate = _normalize_whitespace(address_candidate)
    cleaned_candidate = _clean_address_candidate(normalized_candidate) or normalized_candidate

    geo_data = await geocode_address(cleaned_candidate)
    if not geo_data:
        # Fallback: если полный геокод не удался, но город распознан,
        # возвращаем приблизительные координаты по офису этого города.
        city = _extract_city_from_address(cleaned_candidate)
        if city:
            office = next(
                (item for item in COMPANY_OFFICES if city.lower() in item["name"].lower()),
                None,
            )
            if office:
                approx_geo = build_geo_payload(
                    office["lat"],
                    office["lon"],
                    formatted_address=cleaned_candidate,
                )
                return cleaned_candidate, approx_geo, ADDRESS_STATUS_NOT_FOUND
        return cleaned_candidate, None, ADDRESS_STATUS_NOT_FOUND

    formatted_address = geo_data.get("formatted_address")
    if isinstance(formatted_address, str) and formatted_address.strip():
        return _normalize_whitespace(formatted_address), geo_data, ADDRESS_STATUS_FOUND

    return cleaned_candidate, geo_data, ADDRESS_STATUS_FOUND


def find_nearest_office(lat: float, lon: float) -> Optional[dict[str, Any]]:
    """Нахождение ближайшего офиса по координатам."""

    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return radius * c

    nearest = None
    min_distance = float("inf")
    for office in COMPANY_OFFICES:
        distance = haversine(lat, lon, office["lat"], office["lon"])
        if distance < min_distance:
            min_distance = distance
            nearest = {**office, "distance_km": round(distance, 1)}
    return nearest


async def analyze_text(
    text: str, structured_address: Optional[dict[str, Any]] = None
) -> dict:
    """
    Основная функция анализа:
    - LLM (Ollama) при доступности;
    - rule-based fallback при любой ошибке.
    """
    if not text or not text.strip():
        return {
            "appeal_type": "Консультация",
            "sentiment": "Нейтральный",
            "priority": 1,
            "language": "RU",
            "summary": "Пустое обращение. Рекомендация менеджеру: запросить у клиента детали обращения.",
            "address": None,
            "address_status": ADDRESS_STATUS_NOT_PROVIDED,
            "client_lat": None,
            "client_lon": None,
            "geo": None,
            "method": "empty",
        }

    normalized_text = _normalize_whitespace(text)
    text_lower = normalized_text.lower()
    rule_result = rule_based_analysis(normalized_text)
    result = dict(rule_result)
    method = "rule_based"

    trained_prediction = predict_with_trained_model(normalized_text)
    if trained_prediction:
        strong_positive_feedback = _has_strong_positive_feedback(text_lower)
        resolved_positive_feedback = _has_resolved_positive_feedback(text_lower)
        should_block_negative_overrides = (
            strong_positive_feedback or resolved_positive_feedback
        ) and not _has_strong_fraud_signal(text_lower)

        intent_confidence = float(trained_prediction.get("intent_confidence") or 0.0)
        sentiment_confidence = float(
            trained_prediction.get("sentiment_confidence") or 0.0
        )
        used_trained = False

        if intent_confidence >= TRAINED_INTENT_MIN_CONFIDENCE:
            predicted_intent = trained_prediction.get("appeal_type")
            if (
                should_block_negative_overrides
                and predicted_intent in {
                    "Мошеннические действия",
                    "Претензия",
                    "Жалоба",
                    "Неработоспособность приложения",
                }
            ):
                logger.info(
                    "Blocked trained intent override for positive feedback: %s",
                    predicted_intent,
                )
            else:
                result["appeal_type"] = trained_prediction["appeal_type"]
                used_trained = True
        if (
            trained_prediction.get("sentiment")
            and sentiment_confidence >= TRAINED_SENTIMENT_MIN_CONFIDENCE
        ):
            predicted_sentiment = trained_prediction.get("sentiment")
            if should_block_negative_overrides and predicted_sentiment == "Негативный":
                logger.info(
                    "Blocked trained sentiment override for positive feedback: %s",
                    predicted_sentiment,
                )
            else:
                result["sentiment"] = trained_prediction["sentiment"]
                used_trained = True

        if used_trained:
            result["priority"] = _calculate_priority(
                result["appeal_type"], result["sentiment"], text_lower
            )
            result["summary"] = _generate_summary(
                normalized_text,
                result["appeal_type"],
                result["sentiment"],
                result["priority"],
                result.get("address"),
            )
            method = "trained_nb"

    baseline_result = dict(result)
    baseline_method = method

    ollama_status = await check_ollama_available()
    if ollama_status["available"] and ollama_status["models"]:
        model_to_use = _select_ollama_model(ollama_status["models"])
        if model_to_use:
            try:
                prompt = ANALYSIS_PROMPT_TEMPLATE.format(text=normalized_text[:2500])
                llm_response = await call_ollama(prompt, model_to_use)
                parsed = extract_json_from_response(llm_response)
                if parsed:
                    result = _normalize_analysis_result(
                        parsed, baseline_result, normalized_text
                    )
                    method = f"llm:{model_to_use}"
                else:
                    logger.warning("LLM returned invalid JSON: %s", llm_response[:200])
                    method = f"{baseline_method}_fallback"
            except Exception as exc:
                logger.error("LLM call failed: %s", exc)
                method = f"{baseline_method}_fallback"

    if result.get("appeal_type") not in VALID_APPEAL_TYPES:
        result["appeal_type"] = baseline_result["appeal_type"]
    if result.get("sentiment") not in VALID_SENTIMENTS:
        result["sentiment"] = baseline_result["sentiment"]
    if result.get("language") not in VALID_LANGUAGES:
        result["language"] = "RU"
    result["priority"] = _normalize_priority(
        result.get("priority"), baseline_result["priority"]
    )

    if _has_strong_fraud_signal(text_lower):
        result["appeal_type"] = "Мошеннические действия"
        if result.get("sentiment") == "Позитивный":
            result["sentiment"] = "Негативный"
        result["priority"] = _calculate_priority(
            result["appeal_type"], result["sentiment"], text_lower
        )
        if "fraud_guardrail" not in method:
            method = f"{method}_fraud_guardrail"

    if (
        (_has_strong_positive_feedback(text_lower) or _has_resolved_positive_feedback(text_lower))
        and not _has_strong_fraud_signal(text_lower)
    ):
        if result.get("appeal_type") in {
            "Мошеннические действия",
            "Претензия",
            "Жалоба",
            "Неработоспособность приложения",
        }:
            result["appeal_type"] = "Консультация"
        result["sentiment"] = "Позитивный"
        result["priority"] = _calculate_priority(
            result["appeal_type"], result["sentiment"], text_lower
        )
        if "positive_guardrail" not in method:
            method = f"{method}_positive_guardrail"

    # Бизнес-правило: консультации с нейтральной/позитивной тональностью всегда низкоприоритетные.
    if result.get("appeal_type") == "Консультация" and result.get("sentiment") in {
        "Нейтральный",
        "Позитивный",
    }:
        result["priority"] = max(
            1,
            min(4, _normalize_priority(result.get("priority"), 1)),
        )
        if "consultation_low_priority_guardrail" not in method:
            method = f"{method}_consultation_low_priority_guardrail"

    if not result.get("summary"):
        result["summary"] = _generate_summary(
            normalized_text,
            result["appeal_type"],
            result["sentiment"],
            result["priority"],
            result.get("address"),
        )

    model_address = result.get("address")
    raw_address: Optional[str] = None
    if isinstance(model_address, str):
        raw_address = _clean_address_candidate(model_address)
    structured_address_candidate = build_structured_address_candidate(
        structured_address
    )
    if structured_address_candidate:
        raw_address = structured_address_candidate
    if not raw_address:
        raw_address = extract_address_from_text(normalized_text)

    resolved_address, geo_data, address_status = await resolve_client_address(
        raw_address
    )

    coordinates = extract_coordinates_from_structured_address(structured_address)
    if not coordinates:
        coordinates = extract_coordinates_from_text(normalized_text)
    if not coordinates and isinstance(raw_address, str):
        coordinates = extract_coordinates_from_text(raw_address)
    if not geo_data and coordinates:
        geo_data = build_geo_payload(
            coordinates[0], coordinates[1], formatted_address=resolved_address
        )
        if address_status != ADDRESS_STATUS_FOUND:
            address_status = ADDRESS_STATUS_FOUND
        if not resolved_address and raw_address:
            resolved_address = _normalize_whitespace(raw_address)

    result["address"] = resolved_address
    result["address_status"] = address_status
    result["geo"] = geo_data
    if isinstance(geo_data, dict):
        result["client_lat"] = geo_data.get("lat")
        result["client_lon"] = geo_data.get("lon")
    else:
        result["client_lat"] = None
        result["client_lon"] = None
    result["method"] = method
    return result


def _record_city_label(record: dict) -> str:
    geo = record.get("geo")
    if isinstance(geo, dict):
        nearest = geo.get("nearest_office")
        if isinstance(nearest, dict) and nearest.get("name"):
            return nearest["name"]

    address = record.get("address")
    if isinstance(address, str):
        lower_address = address.lower()
        for city, aliases in CITY_ALIASES.items():
            if any(alias in lower_address for alias in aliases):
                return city
    return "Не определён"


async def generate_chart_data(query: str, appeals_data: list) -> dict:
    """AI-ассистент: генерация данных графика по текстовому запросу."""
    if not appeals_data:
        return {"error": "Нет данных для анализа", "chart_type": "none"}

    query_lower = query.lower()
    has_type = any(word in query_lower for word in ["тип", "категори", "вид", "type"])
    has_city = any(
        word in query_lower for word in ["город", "регион", "геогр", "city", "location"]
    )
    has_sentiment = any(
        word in query_lower for word in ["тональн", "настроен", "sentiment", "эмоц"]
    )
    has_priority = any(
        word in query_lower for word in ["приоритет", "срочн", "priority"]
    )
    has_language = any(word in query_lower for word in ["язык", "language"])
    has_time = any(
        word in query_lower for word in ["время", "дата", "динамик", "trend", "time"]
    )

    if has_type and has_city:
        return _chart_type_by_city(appeals_data, query)
    if has_type:
        return _chart_by_type(appeals_data, query)
    if has_city:
        return _chart_by_city(appeals_data, query)
    if has_sentiment:
        return _chart_by_sentiment(appeals_data, query)
    if has_priority:
        return _chart_by_priority(appeals_data, query)
    if has_language:
        return _chart_by_language(appeals_data, query)
    if has_time:
        return _chart_by_time(appeals_data, query)

    ollama_status = await check_ollama_available()
    if ollama_status["available"] and ollama_status["models"]:
        return await _llm_chart_interpretation(
            query, appeals_data, ollama_status["models"][0]
        )
    return _chart_overview(appeals_data)


def _chart_by_type(data: list, query: str) -> dict:
    counts = Counter(item.get("appeal_type", "Неизвестно") for item in data)
    return {
        "chart_type": "pie",
        "title": "Распределение по типам обращений",
        "labels": list(counts.keys()),
        "values": list(counts.values()),
        "description": f"Всего обращений: {len(data)}",
    }


def _chart_by_city(data: list, query: str) -> dict:
    counts = Counter(_record_city_label(item) for item in data)
    return {
        "chart_type": "bar",
        "title": "Распределение обращений по городам/офисам",
        "labels": list(counts.keys()),
        "values": list(counts.values()),
        "description": "Группировка по ближайшему офису/распознанному городу",
    }


def _chart_type_by_city(data: list, query: str) -> dict:
    types = sorted({item.get("appeal_type", "Неизвестно") for item in data})
    cities = sorted({_record_city_label(item) for item in data})
    city_index = {city: idx for idx, city in enumerate(cities)}
    series_map = {appeal_type: [0] * len(cities) for appeal_type in types}

    for item in data:
        appeal_type = item.get("appeal_type", "Неизвестно")
        city = _record_city_label(item)
        series_map[appeal_type][city_index[city]] += 1

    series = [
        {"name": appeal_type, "values": values}
        for appeal_type, values in series_map.items()
    ]
    totals = [
        sum(row[idx] for row in series_map.values()) for idx in range(len(cities))
    ]

    return {
        "chart_type": "bar",
        "title": "Распределение типов обращений по городам",
        "labels": cities,
        "values": totals,
        "series": series,
        "stacked": True,
        "description": "Series показывает количество каждого типа обращения в разрезе города",
    }


def _chart_by_sentiment(data: list, query: str) -> dict:
    counts = Counter(item.get("sentiment", "Нейтральный") for item in data)
    return {
        "chart_type": "doughnut",
        "title": "Тональность обращений",
        "labels": list(counts.keys()),
        "values": list(counts.values()),
        "description": "Эмоциональный фон клиентских обращений",
    }


def _chart_by_priority(data: list, query: str) -> dict:
    counts = Counter(str(item.get("priority", 5)) for item in data)
    sorted_keys = sorted(counts.keys(), key=lambda value: int(value))
    return {
        "chart_type": "bar",
        "title": "Распределение по приоритету",
        "labels": [f"Приоритет {value}" for value in sorted_keys],
        "values": [counts[value] for value in sorted_keys],
        "description": "Шкала от 1 (низкий) до 10 (критический)",
    }


def _chart_by_language(data: list, query: str) -> dict:
    counts = Counter(item.get("language", "RU") for item in data)
    return {
        "chart_type": "pie",
        "title": "Языки обращений",
        "labels": list(counts.keys()),
        "values": list(counts.values()),
        "description": "Распределение по языку обращения",
    }


def _chart_by_time(data: list, query: str) -> dict:
    dates = [item.get("created_at", "")[:10] for item in data if item.get("created_at")]
    if not dates:
        return _chart_overview(data)

    counts = Counter(dates)
    sorted_dates = sorted(counts.keys())
    return {
        "chart_type": "line",
        "title": "Динамика обращений по времени",
        "labels": sorted_dates,
        "values": [counts[current_date] for current_date in sorted_dates],
        "description": "Количество обращений по дням",
    }


def _chart_overview(data: list) -> dict:
    types = Counter(item.get("appeal_type", "Неизвестно") for item in data)
    sentiments = Counter(item.get("sentiment", "Нейтральный") for item in data)
    avg_priority = (
        sum(item.get("priority", 5) for item in data) / len(data) if data else 0
    )
    return {
        "chart_type": "bar",
        "title": "Общий обзор обращений",
        "labels": list(types.keys()),
        "values": list(types.values()),
        "description": (
            f"Всего: {len(data)} | Средний приоритет: {avg_priority:.1f} | "
            f"Негативных: {sentiments.get('Негативный', 0)}"
        ),
    }


async def _llm_chart_interpretation(query: str, data: list, model: str) -> dict:
    """Интерпретация произвольного графического запроса через LLM."""
    data_summary = {
        "total": len(data),
        "types": dict(Counter(item.get("appeal_type", "") for item in data)),
        "sentiments": dict(Counter(item.get("sentiment", "") for item in data)),
        "languages": dict(Counter(item.get("language", "") for item in data)),
        "cities": dict(Counter(_record_city_label(item) for item in data)),
        "avg_priority": (
            sum(item.get("priority", 5) for item in data) / len(data) if data else 0
        ),
    }

    prompt = f"""Пользователь запрашивает: "{query}"

Доступные данные об обращениях:
{json.dumps(data_summary, ensure_ascii=False, indent=2)}

Определи наиболее подходящий график и верни JSON:
{{
  "chart_type": "<pie|bar|line|doughnut>",
  "title": "<заголовок графика>",
  "labels": ["<метка1>", "<метка2>"],
  "values": [<число1>, <число2>],
  "description": "<описание>"
}}

Верни ТОЛЬКО JSON."""

    try:
        response = await call_ollama(prompt, model)
        parsed = extract_json_from_response(response)
        if parsed and "chart_type" in parsed:
            return parsed
    except Exception as exc:
        logger.error("LLM chart interpretation failed: %s", exc)

    return _chart_overview(data)


_model_load_status = load_trained_model()
if _model_load_status.get("available"):
    logger.info("Loaded trained CSV model from %s", _model_load_status.get("path"))


def generate_psychological_portrait(text: str, sentiment: str, priority: int) -> dict:
    """Генерирует психологический профиль клиента на основе текста обращения."""
    words = re.findall(r'\w+', text.lower())
    total_words = len(words)
    
    # Считаем повторы слов (от 3 букв и длиннее)
    word_counts = Counter([w for w in words if len(w) > 3])
    repetitions = {word: count for word, count in word_counts.items() if count > 1}
    max_repeats = max(repetitions.values()) if repetitions else 0
    
    # Анализ знаков препинания (восклицания/вопросы)
    exclamations = text.count('!')
    questions = text.count('?')
    
    # Определяем психотип
    if priority >= 8 or exclamations > 3:
        profile_type = "Взрывной/Эмоциональный"
        style = "Требует немедленного признания важности проблемы. Избегайте сухих скриптов."
    elif max_repeats > 2:
        profile_type = "Настойчивый/Акцентированный"
        style = "Зациклен на конкретной детали. Важно четко ответить на повторяющиеся вопросы."
    elif sentiment == "Позитивный":
        profile_type = "Лояльный/Конструктивный"
        style = "Расположен к диалогу. Можно предложить дополнительные услуги или кросс-продажи."
    else:
        profile_type = "Деловой/Нейтральный"
        style = "Ценит время и четкость. Минимум вежливости, максимум фактов."

    return {
        "profile_type": profile_type,
        "communication_recommendation": style,
        "metrics": {
            "word_repetition_count": len(repetitions),
            "max_repeats_of_single_word": max_repeats,
            "emotional_punctuation": exclamations + questions,
            "verbosity_level": "Высокая" if total_words > 50 else "Лаконичная"
        }
    }