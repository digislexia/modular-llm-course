# Урок 2: Эмбеддинги и векторные базы

## Введение

В предыдущем уроке мы узнали, что RAG состоит из Retriever и Generator. Сегодня мы погрузимся в сердце Retriever — **эмбеддинги** и **векторные базы данных**.

Представьте задачу: у вас 10 000 документов, и нужно найти те, которые отвечают на вопрос пользователя. Поиск по ключевым словам не сработает — "купить телефон" и "приобрести смартфон" имеют одинаковый смысл, но разные слова. Как искать по смыслу?

Ответ: **эмбеддинги** — превращение текста в "координаты смысла" в многомерном пространстве.

## Цели урока

После завершения урока вы сможете:

- ✅ Понимать, что такое эмбеддинги и как они работают
- ✅ Получать эмбеддинги через API
- ✅ Создавать и использовать векторные базы данных
- ✅ Реализовать семантический поиск
- ✅ Применять chunking для разбиения документов

## Ключевые термины

- **Эмбеддинг (Embedding)** — векторное представление текста в многомерном пространстве
- **Размерность (Dimension)** — количество чисел в векторе (обычно 384-1536)
- **Косинусное сходство** — мера близости двух векторов (-1 до 1)
- **Векторная база данных** — хранилище, оптимизированное для поиска похожих векторов
- **Chunking** — разбиение документа на фрагменты для индексации

## 1. Что такое эмбеддинги

### Аналогия: Координаты смысла

Представьте карту, где каждый город имеет координаты (широта, долгота). Близкие города — близкие координаты.

**Эмбеддинги работают так же**, но для текста:
- Каждый текст получает "координаты" в многомерном пространстве
- Похожие по смыслу тексты → близкие координаты
- Размерность: не 2 (как на карте), а 384-1536 измерений

```
Пример (упрощённо до 3 измерений):

"кот спит на диване"     → [0.8, 0.2, 0.5]
"кошка отдыхает на софе" → [0.79, 0.21, 0.48]  ← Близко!
"программирование на Python" → [0.1, 0.9, 0.3] ← Далеко!
```

### Как это работает?

Модель эмбеддингов (например, text-embedding-3-small от OpenAI):
1. Принимает текст
2. Пропускает через нейронную сеть
3. Возвращает вектор фиксированной длины

```python
"Привет, мир!" → [0.023, -0.041, 0.087, ..., 0.012]  # 1536 чисел
```

### Практика: Первые эмбеддинги

```python
"""
Получение эмбеддингов через OpenAI API.
"""

import os
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    """
    Получает эмбеддинг текста через OpenAI API.
    
    Args:
        text: Текст для эмбеддинга
        model: Модель эмбеддинга
        
    Returns:
        Вектор эмбеддинга (список чисел)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Если нет OpenAI ключа, пробуем OpenRouter
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")
        url = "https://openrouter.ai/api/v1/embeddings"
    else:
        url = "https://api.openai.com/v1/embeddings"
    
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "input": text
        }
    )
    
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        raise Exception(f"Ошибка API: {response.status_code} - {response.text}")


def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Вычисляет косинусное сходство двух векторов.
    
    Значения:
        1.0 = идентичные направления (очень похожи)
        0.0 = ортогональные (не связаны)
       -1.0 = противоположные направления
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    return dot_product / (norm1 * norm2)


# ═══════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИЯ: Похожие и разные тексты
# ═══════════════════════════════════════════════════════════════

print("="*60)
print("ДЕМОНСТРАЦИЯ ЭМБЕДДИНГОВ")
print("="*60)

# Тексты для сравнения
texts = [
    "Кот спит на диване",           # 0
    "Кошка отдыхает на софе",        # 1 - похоже на 0
    "Собака играет во дворе",        # 2 - другое животное
    "Программирование на Python",    # 3 - совсем другая тема
    "Python — популярный язык",      # 4 - похоже на 3
]

print("\nПолучаем эмбеддинги для текстов...")
embeddings = []
for i, text in enumerate(texts):
    emb = get_embedding(text)
    embeddings.append(emb)
    print(f"  [{i}] '{text[:30]}...' → вектор из {len(emb)} чисел")

print(f"\nРазмерность эмбеддингов: {len(embeddings[0])}")

# Сравниваем все пары
print("\n" + "-"*60)
print("МАТРИЦА СХОДСТВА (косинусное сходство):")
print("-"*60)

for i in range(len(texts)):
    for j in range(len(texts)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"  [{i}]-[{j}]: {sim:.3f}", end="")
        if sim > 0.8 and i != j:
            print(" ← Похожи!", end="")
        print()
    print()

print("="*60)
print("ИНТЕРПРЕТАЦИЯ:")
print("="*60)
print("""
• [0]-[1] высокое сходство: оба про кота/кошку на мебели
• [3]-[4] высокое сходство: оба про Python
• [0]-[3] низкое сходство: кот vs программирование
""")
```

### 🔍 Проверьте себя

Прежде чем запускать код:
1. Какие пары текстов будут иметь высокое сходство?
2. Какое сходство ожидаете между "кот" и "Python"?

<details>
<summary>Ожидаемые результаты</summary>

- [0] и [1]: ~0.85-0.95 (кот/кошка на мебели)
- [3] и [4]: ~0.80-0.90 (оба про Python)
- [0] и [3]: ~0.15-0.30 (совершенно разные темы)
- [2] и [0]: ~0.50-0.65 (оба про животных, но разных)
</details>

## 2. Класс для работы с эмбеддингами

Создадим удобный класс для получения и кэширования эмбеддингов:

```python
"""
Класс EmbeddingGenerator для удобной работы с эмбеддингами.
"""

import os
import json
import hashlib
import requests
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class EmbeddingGenerator:
    """
    Генератор эмбеддингов с кэшированием для экономии API вызовов.
    """
    
    def __init__(
        self, 
        model: str = "text-embedding-3-small",
        cache_dir: Optional[str] = ".embedding_cache"
    ):
        """
        Args:
            model: Название модели эмбеддинга
            cache_dir: Директория для кэша (None = без кэширования)
        """
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Создаём директорию кэша
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)
        
        # API настройки
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if os.getenv("OPENAI_API_KEY"):
            self.api_url = "https://api.openai.com/v1/embeddings"
        else:
            self.api_url = "https://openrouter.ai/api/v1/embeddings"
        
        # Статистика
        self.stats = {
            "api_calls": 0,
            "cache_hits": 0,
            "total_tokens": 0
        }
    
    def _get_cache_key(self, text: str) -> str:
        """Генерирует ключ кэша для текста"""
        content = f"{self.model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """Пытается получить эмбеддинг из кэша"""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{self._get_cache_key(text)}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                self.stats["cache_hits"] += 1
                return json.load(f)
        return None
    
    def _save_to_cache(self, text: str, embedding: List[float]):
        """Сохраняет эмбеддинг в кэш"""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{self._get_cache_key(text)}.json"
        with open(cache_file, 'w') as f:
            json.dump(embedding, f)
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Получает эмбеддинг для одного текста.
        
        Args:
            text: Текст для эмбеддинга
            
        Returns:
            Вектор эмбеддинга
        """
        # Проверяем кэш
        cached = self._get_from_cache(text)
        if cached:
            return cached
        
        # Запрос к API
        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "input": text
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        data = response.json()
        embedding = data["data"][0]["embedding"]
        
        # Обновляем статистику
        self.stats["api_calls"] += 1
        if "usage" in data:
            self.stats["total_tokens"] += data["usage"].get("total_tokens", 0)
        
        # Сохраняем в кэш
        self._save_to_cache(text, embedding)
        
        return embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Получает эмбеддинги для списка текстов.
        Использует батчинг для эффективности.
        
        Args:
            texts: Список текстов
            
        Returns:
            Список векторов эмбеддингов
        """
        embeddings = []
        texts_to_fetch = []
        indices_to_fetch = []
        
        # Проверяем кэш для каждого текста
        for i, text in enumerate(texts):
            cached = self._get_from_cache(text)
            if cached:
                embeddings.append((i, cached))
            else:
                texts_to_fetch.append(text)
                indices_to_fetch.append(i)
        
        # Запрашиваем отсутствующие в кэше (батчами по 100)
        batch_size = 100
        for batch_start in range(0, len(texts_to_fetch), batch_size):
            batch_texts = texts_to_fetch[batch_start:batch_start + batch_size]
            batch_indices = indices_to_fetch[batch_start:batch_start + batch_size]
            
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "input": batch_texts
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"API Error: {response.status_code}")
            
            data = response.json()
            self.stats["api_calls"] += 1
            
            for j, item in enumerate(data["data"]):
                emb = item["embedding"]
                original_idx = batch_indices[j]
                original_text = batch_texts[j]
                
                embeddings.append((original_idx, emb))
                self._save_to_cache(original_text, emb)
        
        # Сортируем по оригинальному индексу
        embeddings.sort(key=lambda x: x[0])
        return [emb for _, emb in embeddings]
    
    def similarity(self, text1: str, text2: str) -> float:
        """Вычисляет сходство между двумя текстами"""
        emb1 = np.array(self.get_embedding(text1))
        emb2 = np.array(self.get_embedding(text2))
        
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def find_most_similar(
        self, 
        query: str, 
        candidates: List[str], 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Находит наиболее похожие тексты из списка кандидатов.
        
        Args:
            query: Поисковый запрос
            candidates: Список текстов для поиска
            top_k: Количество результатов
            
        Returns:
            Список словарей с текстом и score
        """
        query_emb = np.array(self.get_embedding(query))
        candidate_embs = self.get_embeddings_batch(candidates)
        
        # Вычисляем сходство для всех кандидатов
        results = []
        for i, cand_emb in enumerate(candidate_embs):
            cand_emb = np.array(cand_emb)
            score = float(np.dot(query_emb, cand_emb) / 
                         (np.linalg.norm(query_emb) * np.linalg.norm(cand_emb)))
            results.append({
                "text": candidates[i],
                "score": score,
                "index": i
            })
        
        # Сортируем по убыванию score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def get_stats(self) -> Dict:
        """Возвращает статистику использования"""
        return self.stats.copy()


# ═══════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИЯ
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("ДЕМОНСТРАЦИЯ EmbeddingGenerator")
    print("="*60)
    
    generator = EmbeddingGenerator()
    
    # Тест 1: Базовое получение эмбеддинга
    print("\n📊 Тест 1: Получение эмбеддинга")
    text = "Python — отличный язык программирования"
    emb = generator.get_embedding(text)
    print(f"Текст: '{text}'")
    print(f"Размерность: {len(emb)}")
    print(f"Первые 5 значений: {emb[:5]}")
    
    # Тест 2: Сходство текстов
    print("\n📊 Тест 2: Сходство текстов")
    pairs = [
        ("машинное обучение", "искусственный интеллект"),
        ("машинное обучение", "приготовление пиццы"),
        ("кот", "котёнок"),
        ("кот", "автомобиль"),
    ]
    
    for text1, text2 in pairs:
        sim = generator.similarity(text1, text2)
        print(f"'{text1}' ↔ '{text2}': {sim:.3f}")
    
    # Тест 3: Поиск похожих
    print("\n📊 Тест 3: Поиск похожих текстов")
    candidates = [
        "Как установить Python на Windows",
        "Рецепт борща с мясом",
        "Основы машинного обучения",
        "Python для анализа данных",
        "Как выбрать ноутбук",
        "Введение в нейронные сети",
    ]
    
    query = "Программирование на Python"
    results = generator.find_most_similar(query, candidates, top_k=3)
    
    print(f"Запрос: '{query}'")
    print("Результаты:")
    for r in results:
        print(f"  {r['score']:.3f} - {r['text']}")
    
    # Статистика
    print(f"\n📈 Статистика: {generator.get_stats()}")
```

## 3. Векторные базы данных

### Зачем нужна специальная база?

При 10 000 документов сравнение запроса с каждым занимает секунды. При 1 000 000 — минуты. Нужен **быстрый поиск ближайших соседей**.

**Векторные базы данных** решают эту задачу:
- Индексируют векторы для быстрого поиска
- Используют алгоритмы ANN (Approximate Nearest Neighbors)
- Поиск за O(log n) вместо O(n)

### Сравнение библиотек

| Библиотека | Тип | Плюсы | Минусы |
|------------|-----|-------|--------|
| **FAISS** | Локальная | Очень быстрая, от Meta | Нет встроенной персистентности |
| **Chroma** | Локальная | Простой API, персистентность | Медленнее FAISS |
| **Qdrant** | Сервер | Продакшен-ready, фильтры | Нужен отдельный сервер |
| **Pinecone** | Облако | Managed, масштабируемость | Платный |

Мы начнём с **FAISS** (простота) и покажем **Chroma** (удобство).

### Реализация с FAISS

```python
"""
Простая векторная база данных на основе FAISS.
"""

import numpy as np
import faiss
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path


class SimpleVectorDB:
    """
    Простая векторная база данных на FAISS.
    Поддерживает добавление, поиск и сохранение/загрузку.
    """
    
    def __init__(self, dimension: int = 1536):
        """
        Args:
            dimension: Размерность векторов (1536 для text-embedding-3-small)
        """
        self.dimension = dimension
        
        # Создаём FAISS индекс (L2 = евклидово расстояние)
        # Для косинусного сходства нормализуем векторы
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (косинусное при нормализации)
        
        # Храним документы и метаданные
        self.documents: List[str] = []
        self.metadata: List[Dict] = []
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Нормализует векторы для косинусного сходства"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms
    
    def add(
        self, 
        texts: List[str], 
        embeddings: List[List[float]],
        metadata: Optional[List[Dict]] = None
    ):
        """
        Добавляет документы в базу.
        
        Args:
            texts: Тексты документов
            embeddings: Эмбеддинги документов
            metadata: Дополнительные метаданные (опционально)
        """
        if len(texts) != len(embeddings):
            raise ValueError("Количество текстов и эмбеддингов должно совпадать")
        
        # Конвертируем в numpy и нормализуем
        vectors = np.array(embeddings, dtype=np.float32)
        vectors = self._normalize(vectors)
        
        # Добавляем в индекс
        self.index.add(vectors)
        
        # Сохраняем документы и метаданные
        self.documents.extend(texts)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(texts))
    
    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Ищет наиболее похожие документы.
        
        Args:
            query_embedding: Эмбеддинг запроса
            top_k: Количество результатов
            
        Returns:
            Список словарей с документами и scores
        """
        if self.index.ntotal == 0:
            return []
        
        # Нормализуем запрос
        query = np.array([query_embedding], dtype=np.float32)
        query = self._normalize(query)
        
        # Ищем
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))
        
        # Формируем результаты
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS возвращает -1 если не найдено
                continue
            results.append({
                "text": self.documents[idx],
                "score": float(score),
                "index": int(idx),
                "metadata": self.metadata[idx]
            })
        
        return results
    
    def save(self, path: str):
        """Сохраняет базу на диск"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем индекс FAISS
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Сохраняем документы и метаданные
        with open(path / "data.json", 'w', encoding='utf-8') as f:
            json.dump({
                "documents": self.documents,
                "metadata": self.metadata,
                "dimension": self.dimension
            }, f, ensure_ascii=False, indent=2)
        
        print(f"💾 База сохранена в {path}")
    
    def load(self, path: str):
        """Загружает базу с диска"""
        path = Path(path)
        
        # Загружаем индекс
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Загружаем данные
        with open(path / "data.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            self.dimension = data["dimension"]
        
        print(f"📂 Загружено {len(self.documents)} документов")
    
    def __len__(self) -> int:
        return len(self.documents)


# ═══════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИЯ
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from embedding_generator import EmbeddingGenerator  # Из предыдущего примера
    
    print("="*60)
    print("ДЕМОНСТРАЦИЯ SimpleVectorDB")
    print("="*60)
    
    # Создаём генератор эмбеддингов и базу
    embedder = EmbeddingGenerator()
    db = SimpleVectorDB(dimension=1536)
    
    # Примеры документов
    documents = [
        "Python — высокоуровневый язык программирования общего назначения",
        "JavaScript используется для создания интерактивных веб-страниц",
        "Машинное обучение — раздел искусственного интеллекта",
        "SQL — язык для работы с реляционными базами данных",
        "Docker позволяет упаковывать приложения в контейнеры",
        "Git — распределённая система контроля версий",
        "REST API — архитектурный стиль для веб-сервисов",
        "Kubernetes оркестрирует контейнеризированные приложения",
    ]
    
    # Получаем эмбеддинги
    print("\n📊 Индексируем документы...")
    embeddings = embedder.get_embeddings_batch(documents)
    
    # Добавляем метаданные
    metadata = [{"category": "programming"} for _ in documents]
    
    # Добавляем в базу
    db.add(documents, embeddings, metadata)
    print(f"✅ Добавлено {len(db)} документов")
    
    # Поиск
    print("\n🔍 Тестируем поиск...")
    queries = [
        "Как работать с базами данных?",
        "Контейнеризация приложений",
        "Языки программирования для веба",
    ]
    
    for query in queries:
        print(f"\n📌 Запрос: '{query}'")
        query_emb = embedder.get_embedding(query)
        results = db.search(query_emb, top_k=3)
        
        for i, r in enumerate(results, 1):
            print(f"   {i}. [{r['score']:.3f}] {r['text'][:50]}...")
    
    # Сохранение и загрузка
    print("\n💾 Тестируем персистентность...")
    db.save("test_db")
    
    db2 = SimpleVectorDB()
    db2.load("test_db")
    print(f"✅ Загружено {len(db2)} документов")
```

### Альтернатива: Chroma

Chroma — более высокоуровневая библиотека с удобным API:

```python
"""
Использование Chroma для векторного хранилища.
pip install chromadb
"""

import chromadb
from chromadb.utils import embedding_functions


def demo_chroma():
    """Демонстрация работы с Chroma"""
    
    # Создаём клиент (персистентный)
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Используем OpenAI эмбеддинги (или свои)
    # openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    #     api_key="your-key",
    #     model_name="text-embedding-3-small"
    # )
    
    # Создаём или получаем коллекцию
    collection = client.get_or_create_collection(
        name="my_documents",
        # embedding_function=openai_ef  # Опционально
    )
    
    # Добавляем документы
    collection.add(
        documents=[
            "Python — отличный язык программирования",
            "JavaScript популярен для веб-разработки",
            "Машинное обучение меняет мир",
        ],
        metadatas=[
            {"category": "python"},
            {"category": "javascript"},
            {"category": "ml"},
        ],
        ids=["doc1", "doc2", "doc3"]
    )
    
    # Поиск
    results = collection.query(
        query_texts=["программирование на Python"],
        n_results=2
    )
    
    print("Результаты поиска:")
    for doc, dist in zip(results['documents'][0], results['distances'][0]):
        print(f"  [{dist:.3f}] {doc}")
    
    # Фильтрация по метаданным
    results_filtered = collection.query(
        query_texts=["языки программирования"],
        n_results=2,
        where={"category": "python"}  # Только Python
    )
    
    print("\nС фильтром (только Python):")
    for doc in results_filtered['documents'][0]:
        print(f"  {doc}")


if __name__ == "__main__":
    demo_chroma()
```

## 4. Семантический поиск end-to-end

Объединим всё в работающую систему поиска:

```python
"""
Полная система семантического поиска.
"""

import os
import numpy as np
from typing import List, Dict
from pathlib import Path


class SemanticSearch:
    """
    Система семантического поиска по документам.
    """
    
    def __init__(self, embedding_generator, vector_db):
        """
        Args:
            embedding_generator: Экземпляр EmbeddingGenerator
            vector_db: Экземпляр SimpleVectorDB
        """
        self.embedder = embedding_generator
        self.db = vector_db
    
    def index_documents(
        self, 
        documents: List[str],
        metadata: List[Dict] = None
    ):
        """
        Индексирует документы в векторную базу.
        
        Args:
            documents: Список документов
            metadata: Метаданные для каждого документа
        """
        print(f"📊 Индексирование {len(documents)} документов...")
        
        # Получаем эмбеддинги
        embeddings = self.embedder.get_embeddings_batch(documents)
        
        # Добавляем в базу
        self.db.add(documents, embeddings, metadata)
        
        print(f"✅ Индексировано {len(self.db)} документов")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Выполняет семантический поиск.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            
        Returns:
            Список результатов с текстом и score
        """
        # Получаем эмбеддинг запроса
        query_embedding = self.embedder.get_embedding(query)
        
        # Ищем в базе
        results = self.db.search(query_embedding, top_k)
        
        return results
    
    def search_with_threshold(
        self, 
        query: str, 
        threshold: float = 0.7,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Поиск с фильтрацией по минимальному score.
        
        Args:
            query: Поисковый запрос
            threshold: Минимальный score (0-1)
            top_k: Максимум результатов
            
        Returns:
            Отфильтрованные результаты
        """
        results = self.search(query, top_k)
        return [r for r in results if r["score"] >= threshold]


# ═══════════════════════════════════════════════════════════════
# ПОЛНЫЙ ПРИМЕР: Поиск по документации
# ═══════════════════════════════════════════════════════════════

def load_sample_documents() -> List[Dict]:
    """Загружает примеры документов из файла"""
    
    # Читаем файл с документами
    docs_path = Path(__file__).parent / "data" / "sample_docs.txt"
    
    if not docs_path.exists():
        # Если файла нет, используем встроенные примеры
        return [
            {"text": "Python — высокоуровневый язык программирования", "source": "doc1"},
            {"text": "Списки в Python — упорядоченные коллекции", "source": "doc2"},
            {"text": "Словари хранят пары ключ-значение", "source": "doc3"},
            {"text": "Функции определяются через def", "source": "doc4"},
            {"text": "Классы создаются ключевым словом class", "source": "doc5"},
        ]
    
    documents = []
    current_doc = ""
    doc_num = 0
    
    with open(docs_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "---":
                if current_doc.strip():
                    doc_num += 1
                    documents.append({
                        "text": current_doc.strip(),
                        "source": f"doc_{doc_num}"
                    })
                current_doc = ""
            else:
                current_doc += line
    
    # Добавляем последний документ
    if current_doc.strip():
        doc_num += 1
        documents.append({
            "text": current_doc.strip(),
            "source": f"doc_{doc_num}"
        })
    
    return documents


def main():
    """Демонстрация семантического поиска"""
    
    print("="*70)
    print("СИСТЕМА СЕМАНТИЧЕСКОГО ПОИСКА")
    print("="*70)
    
    # Инициализация
    from embedding_generator import EmbeddingGenerator
    from vector_db import SimpleVectorDB
    
    embedder = EmbeddingGenerator()
    db = SimpleVectorDB(dimension=1536)
    search_engine = SemanticSearch(embedder, db)
    
    # Загружаем документы
    print("\n📄 Загрузка документов...")
    docs = load_sample_documents()
    print(f"Загружено {len(docs)} документов")
    
    # Индексируем
    texts = [d["text"] for d in docs]
    metadata = [{"source": d["source"]} for d in docs]
    search_engine.index_documents(texts, metadata)
    
    # Интерактивный поиск
    print("\n" + "="*70)
    print("ИНТЕРАКТИВНЫЙ ПОИСК")
    print("="*70)
    print("Введите запрос (или 'выход' для завершения)")
    
    while True:
        query = input("\n🔍 Запрос: ").strip()
        
        if query.lower() in ['выход', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        results = search_engine.search(query, top_k=5)
        
        print(f"\n📊 Найдено {len(results)} результатов:")
        for i, r in enumerate(results, 1):
            score_bar = "█" * int(r['score'] * 10) + "░" * (10 - int(r['score'] * 10))
            print(f"\n{i}. [{score_bar}] {r['score']:.3f}")
            print(f"   {r['text'][:100]}...")
            if r['metadata']:
                print(f"   📁 Источник: {r['metadata'].get('source', 'N/A')}")


if __name__ == "__main__":
    main()
```

## 5. Chunking — нарезка документов

### Почему нужен chunking?

- Эмбеддинги работают лучше на коротких текстах (200-500 токенов)
- Длинный документ размывает семантику
- При поиске нужен конкретный релевантный фрагмент

### Стратегии chunking

```python
"""
Различные стратегии разбиения документов на чанки.
"""

import re
from typing import List, Dict


class TextChunker:
    """
    Класс для разбиения текста на чанки.
    """
    
    @staticmethod
    def chunk_by_chars(
        text: str, 
        chunk_size: int = 1000, 
        overlap: int = 200
    ) -> List[str]:
        """
        Разбивает по количеству символов с перекрытием.
        
        Args:
            text: Исходный текст
            chunk_size: Размер чанка в символах
            overlap: Перекрытие между чанками
            
        Returns:
            Список чанков
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Пытаемся закончить на границе предложения
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size // 2:
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return [c for c in chunks if c]
    
    @staticmethod
    def chunk_by_sentences(
        text: str, 
        sentences_per_chunk: int = 5,
        overlap_sentences: int = 1
    ) -> List[str]:
        """
        Разбивает по предложениям.
        
        Args:
            text: Исходный текст
            sentences_per_chunk: Предложений в чанке
            overlap_sentences: Перекрытие в предложениях
            
        Returns:
            Список чанков
        """
        # Простое разбиение по точкам (можно улучшить)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        start = 0
        
        while start < len(sentences):
            end = min(start + sentences_per_chunk, len(sentences))
            chunk = ' '.join(sentences[start:end])
            chunks.append(chunk)
            
            start = end - overlap_sentences
            if start < 0:
                start = end
        
        return chunks
    
    @staticmethod
    def chunk_by_paragraphs(text: str, min_chunk_size: int = 100) -> List[str]:
        """
        Разбивает по параграфам (двойной перенос строки).
        
        Args:
            text: Исходный текст
            min_chunk_size: Минимальный размер чанка
            
        Returns:
            Список чанков
        """
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Объединяем маленькие параграфы
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < min_chunk_size * 3:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    @staticmethod
    def chunk_with_metadata(
        text: str,
        source: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[Dict]:
        """
        Разбивает текст и добавляет метаданные к каждому чанку.
        
        Returns:
            Список словарей с текстом и метаданными
        """
        chunks = TextChunker.chunk_by_chars(text, chunk_size, overlap)
        
        return [
            {
                "text": chunk,
                "metadata": {
                    "source": source,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "char_start": sum(len(c) for c in chunks[:i]),
                }
            }
            for i, chunk in enumerate(chunks)
        ]


# ═══════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИЯ
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Пример длинного текста
    long_text = """
    Python — высокоуровневый язык программирования общего назначения с динамической 
    строгой типизацией и автоматическим управлением памятью. Он ориентирован на 
    повышение производительности разработчика, читаемости кода и его качества.

    Синтаксис ядра Python минималистичен. В то же время стандартная библиотека 
    включает большой объём полезных функций. Python поддерживает структурное, 
    объектно-ориентированное, функциональное, императивное и аспектно-ориентированное 
    программирование.

    Основные архитектурные черты — динамическая типизация, автоматическое управление 
    памятью, полная интроспекция, механизм обработки исключений, поддержка многопоточных 
    вычислений с глобальной блокировкой интерпретатора (GIL).

    Python активно развивается и используется в таких областях как веб-разработка, 
    машинное обучение, анализ данных, автоматизация и скриптинг, научные вычисления.
    """
    
    chunker = TextChunker()
    
    print("="*60)
    print("ДЕМОНСТРАЦИЯ CHUNKING")
    print("="*60)
    
    # По символам
    print("\n📊 Chunking по символам (500 символов, overlap 100):")
    chunks = chunker.chunk_by_chars(long_text, 500, 100)
    for i, chunk in enumerate(chunks):
        print(f"\n  Чанк {i+1} ({len(chunk)} символов):")
        print(f"    {chunk[:100]}...")
    
    # По предложениям
    print("\n📊 Chunking по предложениям (3 предложения):")
    chunks = chunker.chunk_by_sentences(long_text, 3, 1)
    for i, chunk in enumerate(chunks):
        print(f"\n  Чанк {i+1}:")
        print(f"    {chunk[:100]}...")
    
    # С метаданными
    print("\n📊 Chunking с метаданными:")
    chunks = chunker.chunk_with_metadata(long_text, "python_intro.txt", 400, 50)
    for chunk in chunks:
        print(f"\n  Чанк {chunk['metadata']['chunk_index']+1}:")
        print(f"    Источник: {chunk['metadata']['source']}")
        print(f"    Текст: {chunk['text'][:80]}...")
```

## Практические задания

### 🟢 Базовый уровень

**Задание 1: Исследование эмбеддингов**

1. Получите эмбеддинги для 10 слов из разных категорий:
   - 3 слова про еду
   - 3 слова про технологии
   - 4 случайных слова
2. Постройте матрицу сходства
3. Визуализируйте результаты (можно текстом)

**Задание 2: Первая векторная база**

1. Создайте SimpleVectorDB
2. Добавьте 20 документов из `sample_docs.txt`
3. Выполните 5 поисковых запросов
4. Оцените качество результатов

### 🟡 Средний уровень

**Задание 3: Сравнение стратегий chunking**

1. Возьмите длинный текст (1000+ слов)
2. Разбейте его тремя разными способами:
   - По символам (500 символов)
   - По предложениям (5 предложений)
   - По параграфам
3. Для каждого способа:
   - Проиндексируйте чанки
   - Выполните одинаковые запросы
   - Сравните качество результатов

**Задание 4: Семантический поиск с фильтрацией**

1. Создайте базу с документами и метаданными (категория, автор, дата)
2. Реализуйте поиск с фильтрацией по метаданным
3. Протестируйте на 10 запросах с разными фильтрами

### 🔴 Продвинутый уровень

**Задание 5: Поисковая система по PDF**

1. Загрузите PDF документ (учебник, статья)
2. Извлеките текст
3. Разбейте на чанки с overlap
4. Создайте поисковую систему с:
   - Семантическим поиском
   - Показом источника (страница)
   - Сохранением/загрузкой индекса

**Задание 6: Сравнение библиотек**

1. Реализуйте одинаковый поиск на:
   - FAISS
   - Chroma
2. Сравните по:
   - Скорости индексации
   - Скорости поиска
   - Качеству результатов
   - Удобству API
3. Оформите результаты в отчёт

## Контрольные вопросы

1. **Что такое эмбеддинг простыми словами?**
   <details>
   <summary>Ответ</summary>
   Эмбеддинг — это представление текста в виде вектора чисел, где похожие по смыслу тексты имеют близкие векторы. Как координаты на карте, только в многомерном пространстве смыслов.
   </details>

2. **Почему семантический поиск лучше поиска по ключевым словам?**
   <details>
   <summary>Ответ</summary>
   Семантический поиск понимает смысл, а не просто совпадение слов. "Купить телефон" и "приобрести смартфон" означают одно и то же, но имеют разные слова. Эмбеддинги улавливают это сходство.
   </details>

3. **Что такое chunking и зачем он нужен?**
   <details>
   <summary>Ответ</summary>
   Chunking — разбиение документа на небольшие фрагменты (200-500 токенов). Нужен потому что: 1) эмбеддинги лучше работают на коротких текстах, 2) при поиске нужен конкретный релевантный фрагмент, а не весь документ.
   </details>

4. **Чем отличается FAISS от Chroma?**
   <details>
   <summary>Ответ</summary>
   FAISS — низкоуровневая библиотека от Meta, очень быстрая, но требует ручного управления данными. Chroma — высокоуровневая библиотека с удобным API, встроенной персистентностью и метаданными, но медленнее.
   </details>

5. **Как измерить похожесть двух векторов?**
   <details>
   <summary>Ответ</summary>
   Обычно используют косинусное сходство — косинус угла между векторами. Значения от -1 (противоположные) до 1 (идентичные). Для нормализованных векторов это эквивалентно скалярному произведению.
   </details>

## Заключение урока

### Что мы изучили

- **Эмбеддинги**: как текст превращается в вектор чисел
- **Получение эмбеддингов**: через API с кэшированием
- **Векторные базы**: FAISS и Chroma для хранения и поиска
- **Семантический поиск**: нахождение похожих документов по смыслу
- **Chunking**: разбиение документов для эффективного поиска

### Связь с предыдущим уроком

В уроке 1 мы узнали, что RAG состоит из Retriever и Generator. Сегодня мы построили **Retriever** — систему, которая:
- Превращает документы в векторы
- Хранит их в специальной базе
- Находит релевантные по запросу

### Что дальше

В следующем уроке **"Интеграция поиска и генерации"** мы:
- Соединим Retriever с LLM (Generator)
- Создадим полноценный RAG pipeline
- Построим работающего QA-бота

### Ваш прогресс

🎯 **Отличная работа!** Теперь вы умеете:
- ✅ Получать и сравнивать эмбеддинги
- ✅ Создавать векторные базы данных
- ✅ Выполнять семантический поиск
- ✅ Разбивать документы на чанки

**Готовы собрать всё вместе?** Переходите к [Уроку 3: Интеграция поиска и генерации](lesson_3_rag_pipeline.md)!

---

## Дополнительные материалы

### Документация:
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Chroma Documentation](https://docs.trychroma.com/)

### Статьи:
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [Efficient and Robust Approximate Nearest Neighbor Search](https://arxiv.org/abs/1603.09320)

### Инструменты:
- [sentence-transformers](https://www.sbert.net/) — бесплатные модели эмбеддингов
- [Qdrant](https://qdrant.tech/) — production векторная база
- [Pinecone](https://www.pinecone.io/) — managed векторная база

