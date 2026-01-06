# Урок 4: Улучшение и расширение RAG

## Введение

Поздравляем! Вы уже создали работающий RAG pipeline. Но между MVP и production-системой — огромная пропасть. В этом финальном уроке модуля мы изучим продвинутые техники, которые превратят ваш RAG из "работает" в "работает отлично".

Мы рассмотрим:
- Гибридный поиск (семантика + ключевые слова)
- Продвинутые стратегии разбиения документов
- Техники улучшения запросов
- Production considerations: мониторинг, обновление базы, оптимизация стоимости

## Цели урока

После завершения урока вы сможете:

- ✅ Реализовать гибридный поиск для повышения точности
- ✅ Применять продвинутые стратегии chunking
- ✅ Использовать query transformation для улучшения поиска
- ✅ Настроить мониторинг и логирование RAG
- ✅ Оптимизировать стоимость и производительность

## Ключевые термины

- **Гибридный поиск** — комбинация семантического и keyword поиска
- **BM25** — алгоритм ранжирования на основе TF-IDF
- **HyDE** — Hypothetical Document Embeddings
- **Query expansion** — расширение запроса синонимами
- **Sentence window retrieval** — поиск с расширенным контекстом

## 1. Гибридный поиск

### Проблема чисто семантического поиска

Семантический поиск отлично находит похожие по смыслу тексты, но иногда:
- Пропускает точные совпадения (названия, коды, номера)
- Путает контекст при многозначных словах
- Не учитывает редкие термины

**Решение**: комбинировать с keyword-поиском (BM25).

### Реализация гибридного поиска

```python
"""
Гибридный поиск: семантика + BM25.
"""

import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import re


class HybridSearch:
    """
    Гибридный поиск, объединяющий векторный и keyword поиск.
    """
    
    def __init__(self, vector_db, alpha: float = 0.5):
        """
        Args:
            vector_db: Векторная база данных (наш SimpleVectorDB)
            alpha: Вес семантического поиска (0-1)
                   0 = только BM25
                   1 = только семантика
                   0.5 = равный вес
        """
        self.vector_db = vector_db
        self.alpha = alpha
        
        # BM25 индекс
        self.bm25 = None
        self.tokenized_docs = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Простая токенизация для BM25"""
        # Приводим к нижнему регистру
        text = text.lower()
        # Убираем пунктуацию и разбиваем на слова
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def build_bm25_index(self, documents: List[str]):
        """
        Строит BM25 индекс для keyword поиска.
        Должен вызываться после добавления документов в vector_db.
        """
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"✅ BM25 индекс построен для {len(documents)} документов")
    
    def search(
        self, 
        query: str, 
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Выполняет гибридный поиск.
        
        Args:
            query: Текстовый запрос
            query_embedding: Эмбеддинг запроса
            top_k: Количество результатов
            
        Returns:
            Список результатов с комбинированными scores
        """
        # 1. Семантический поиск
        semantic_results = self.vector_db.search(query_embedding, top_k * 2)
        
        # 2. BM25 поиск
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Нормализуем BM25 scores (0-1)
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        
        # 3. Объединяем результаты
        combined_scores = {}
        
        # Добавляем семантические результаты
        for r in semantic_results:
            idx = r["index"]
            combined_scores[idx] = {
                "text": r["text"],
                "metadata": r["metadata"],
                "semantic_score": r["score"],
                "bm25_score": float(bm25_scores[idx]),
                "index": idx
            }
        
        # Добавляем BM25 top результаты, которых нет в семантических
        bm25_top_indices = np.argsort(bm25_scores)[-top_k * 2:][::-1]
        for idx in bm25_top_indices:
            if idx not in combined_scores:
                combined_scores[idx] = {
                    "text": self.vector_db.documents[idx],
                    "metadata": self.vector_db.metadata[idx],
                    "semantic_score": 0.0,  # Не попал в семантический top
                    "bm25_score": float(bm25_scores[idx]),
                    "index": idx
                }
        
        # 4. Вычисляем комбинированный score
        results = []
        for idx, data in combined_scores.items():
            combined = (
                self.alpha * data["semantic_score"] +
                (1 - self.alpha) * data["bm25_score"]
            )
            results.append({
                **data,
                "combined_score": combined
            })
        
        # 5. Сортируем и возвращаем top_k
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:top_k]
    
    def search_with_auto_alpha(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Автоматически выбирает alpha на основе типа запроса.
        
        Эвристика:
        - Короткий запрос с точными терминами → больше BM25
        - Длинный вопрос → больше семантики
        """
        tokens = self._tokenize(query)
        
        # Эвристики
        is_short = len(tokens) <= 3
        has_code_like = any(re.match(r'^[A-Z0-9_-]+$', t) for t in query.split())
        has_question_words = any(w in query.lower() for w in ['как', 'что', 'почему', 'зачем', 'какой'])
        
        # Выбираем alpha
        if is_short and has_code_like:
            alpha = 0.3  # Больше BM25 для точных терминов
        elif has_question_words:
            alpha = 0.7  # Больше семантики для вопросов
        else:
            alpha = 0.5  # Баланс
        
        # Временно меняем alpha
        original_alpha = self.alpha
        self.alpha = alpha
        results = self.search(query, query_embedding, top_k)
        self.alpha = original_alpha
        
        return results


# ═══════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИЯ
# ═══════════════════════════════════════════════════════════════

def demo_hybrid_search():
    """Демонстрация преимуществ гибридного поиска"""
    
    # pip install rank-bm25
    
    documents = [
        "Python 3.12 добавил поддержку f-string в документации",
        "Новые возможности в Python 3.12 включают улучшенные сообщения об ошибках",
        "JavaScript ES2023 представил новые методы массивов",
        "Функция async/await в Python для асинхронного программирования",
        "PEP 701 описывает изменения в f-строках Python 3.12",
    ]
    
    print("="*70)
    print("СРАВНЕНИЕ: Семантический vs Гибридный поиск")
    print("="*70)
    
    # Запрос с точным термином
    query = "PEP 701"
    
    print(f"\n📌 Запрос: '{query}'")
    print("\nСемантический поиск может не найти точное совпадение 'PEP 701',")
    print("потому что это код, а не семантически значимый текст.")
    print("\nГибридный поиск найдёт его благодаря BM25 компоненту!")
    
    # ... (код демонстрации с реальными эмбеддингами)
```

## 2. Продвинутые стратегии Chunking

### Sentence Window Retrieval

Идея: ищем по маленьким чанкам (точность), но возвращаем расширенный контекст.

```python
"""
Sentence Window Retrieval - поиск с расширенным контекстом.
"""

from typing import List, Dict, Tuple


class SentenceWindowChunker:
    """
    Chunking с сохранением контекстного окна.
    
    Создаёт маленькие чанки для точного поиска,
    но хранит ссылки на соседние предложения для расширенного контекста.
    """
    
    def __init__(self, window_size: int = 2):
        """
        Args:
            window_size: Количество предложений до и после для контекста
        """
        self.window_size = window_size
    
    def chunk_with_windows(
        self, 
        text: str, 
        source: str = "unknown"
    ) -> List[Dict]:
        """
        Разбивает текст на предложения с контекстными окнами.
        
        Returns:
            Список чанков с полями:
            - text: само предложение (для эмбеддинга)
            - window_text: расширенный контекст (для LLM)
            - metadata: информация о позиции
        """
        # Разбиваем на предложения
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        
        for i, sentence in enumerate(sentences):
            # Определяем границы окна
            start = max(0, i - self.window_size)
            end = min(len(sentences), i + self.window_size + 1)
            
            # Формируем расширенный контекст
            window_sentences = sentences[start:end]
            window_text = ' '.join(window_sentences)
            
            chunks.append({
                "text": sentence,  # Для эмбеддинга
                "window_text": window_text,  # Для контекста LLM
                "metadata": {
                    "source": source,
                    "sentence_index": i,
                    "window_start": start,
                    "window_end": end,
                    "total_sentences": len(sentences)
                }
            })
        
        return chunks


class ParentChildChunker:
    """
    Parent-Child Chunking: маленькие чанки ссылаются на большие "родительские".
    """
    
    def __init__(
        self, 
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 400
    ):
        self.parent_size = parent_chunk_size
        self.child_size = child_chunk_size
    
    def chunk_hierarchical(
        self, 
        text: str, 
        source: str = "unknown"
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Создаёт иерархию чанков: родители и дети.
        
        Returns:
            (parent_chunks, child_chunks)
            
        При поиске:
        1. Ищем по child_chunks (точно)
        2. Возвращаем parent_chunks (контекст)
        """
        # Создаём родительские чанки
        parent_chunks = []
        start = 0
        parent_idx = 0
        
        while start < len(text):
            end = min(start + self.parent_size, len(text))
            
            # Пытаемся разделить на границе параграфа
            if end < len(text):
                last_para = text[start:end].rfind('\n\n')
                if last_para > self.parent_size // 2:
                    end = start + last_para
            
            parent_chunks.append({
                "text": text[start:end].strip(),
                "metadata": {
                    "source": source,
                    "parent_index": parent_idx,
                    "char_start": start,
                    "char_end": end
                }
            })
            
            parent_idx += 1
            start = end
        
        # Создаём дочерние чанки с ссылками на родителей
        child_chunks = []
        
        for parent in parent_chunks:
            parent_text = parent["text"]
            parent_idx = parent["metadata"]["parent_index"]
            
            # Разбиваем родителя на детей
            child_start = 0
            child_idx = 0
            
            while child_start < len(parent_text):
                child_end = min(child_start + self.child_size, len(parent_text))
                
                child_chunks.append({
                    "text": parent_text[child_start:child_end].strip(),
                    "metadata": {
                        "source": source,
                        "parent_index": parent_idx,
                        "child_index": child_idx,
                        "parent_text": parent_text  # Ссылка на полный родительский текст
                    }
                })
                
                child_idx += 1
                child_start = child_end
        
        return parent_chunks, child_chunks


# ═══════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИЯ
# ═══════════════════════════════════════════════════════════════

def demo_advanced_chunking():
    """Сравнение стратегий chunking"""
    
    sample_text = """
    Python — высокоуровневый язык программирования общего назначения. 
    Он был создан Гвидо ван Россумом в 1991 году. Python поддерживает 
    несколько парадигм программирования.
    
    Одной из ключевых особенностей Python является его читаемость. 
    Код на Python легко понять даже новичкам. Это делает его отличным 
    выбором для обучения программированию.
    
    Python широко используется в науке о данных и машинном обучении. 
    Библиотеки NumPy, Pandas и scikit-learn стали стандартом индустрии.
    TensorFlow и PyTorch используются для глубокого обучения.
    """
    
    print("="*70)
    print("СРАВНЕНИЕ СТРАТЕГИЙ CHUNKING")
    print("="*70)
    
    # Sentence Window
    print("\n📊 SENTENCE WINDOW RETRIEVAL:")
    sw_chunker = SentenceWindowChunker(window_size=1)
    sw_chunks = sw_chunker.chunk_with_windows(sample_text, "python.txt")
    
    for i, chunk in enumerate(sw_chunks[:3]):
        print(f"\n  Чанк {i+1}:")
        print(f"    Для поиска: '{chunk['text'][:50]}...'")
        print(f"    Контекст: '{chunk['window_text'][:80]}...'")
    
    # Parent-Child
    print("\n📊 PARENT-CHILD CHUNKING:")
    pc_chunker = ParentChildChunker(parent_chunk_size=500, child_chunk_size=100)
    parents, children = pc_chunker.chunk_hierarchical(sample_text, "python.txt")
    
    print(f"\n  Родительских чанков: {len(parents)}")
    print(f"  Дочерних чанков: {len(children)}")
    
    print("\n  Пример дочернего чанка:")
    child = children[0]
    print(f"    Для поиска: '{child['text'][:50]}...'")
    print(f"    Родитель: '{child['metadata']['parent_text'][:80]}...'")


if __name__ == "__main__":
    demo_advanced_chunking()
```

## 3. Query Transformation

### HyDE — Hypothetical Document Embeddings

Идея: вместо эмбеддинга вопроса, генерируем гипотетический ответ и ищем по нему.

```python
"""
Query Transformation техники для улучшения retrieval.
"""

import os
import requests
from typing import List, Dict


class QueryTransformer:
    """
    Техники трансформации запросов для улучшения поиска.
    """
    
    def __init__(self, llm_model: str = "openai/gpt-3.5-turbo"):
        self.llm_model = llm_model
        self.api_key = os.getenv("OPENROUTER_API_KEY")
    
    def _call_llm(self, prompt: str, max_tokens: int = 200) -> str:
        """Вызывает LLM"""
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
        )
        return response.json()["choices"][0]["message"]["content"]
    
    # ═══════════════════════════════════════════════════════════
    # HyDE - Hypothetical Document Embeddings
    # ═══════════════════════════════════════════════════════════
    
    def hyde(self, query: str) -> str:
        """
        HyDE: Генерирует гипотетический документ, который бы ответил на вопрос.
        
        Вместо поиска по вопросу "Как работает фотосинтез?",
        генерируем ответ и ищем по нему — так найдём более релевантные документы.
        """
        prompt = f"""Напиши короткий информативный параграф (3-4 предложения), 
который бы идеально отвечал на следующий вопрос.
Не начинай с "Ответ:" или подобного. Просто напиши содержательный текст.

Вопрос: {query}

Параграф:"""
        
        hypothetical_doc = self._call_llm(prompt, max_tokens=150)
        return hypothetical_doc.strip()
    
    # ═══════════════════════════════════════════════════════════
    # Query Expansion - расширение запроса
    # ═══════════════════════════════════════════════════════════
    
    def expand_query(self, query: str) -> str:
        """
        Расширяет запрос синонимами и связанными терминами.
        """
        prompt = f"""Расширь следующий поисковый запрос, добавив:
- Синонимы ключевых слов
- Связанные термины
- Альтернативные формулировки

Исходный запрос: {query}

Расширенный запрос (одной строкой, без нумерации):"""
        
        expanded = self._call_llm(prompt, max_tokens=100)
        return expanded.strip()
    
    # ═══════════════════════════════════════════════════════════
    # Multi-Query - множественные запросы
    # ═══════════════════════════════════════════════════════════
    
    def generate_multi_queries(self, query: str, n: int = 3) -> List[str]:
        """
        Генерирует несколько вариантов запроса для расширения поиска.
        
        Ищем по всем вариантам и объединяем результаты.
        """
        prompt = f"""Переформулируй следующий вопрос {n} разными способами.
Каждая формулировка должна искать ту же информацию, но другими словами.

Исходный вопрос: {query}

Варианты (по одному на строку):
1."""
        
        response = self._call_llm(prompt, max_tokens=200)
        
        # Парсим варианты
        lines = response.strip().split('\n')
        queries = [query]  # Оригинальный запрос тоже включаем
        
        for line in lines:
            # Убираем нумерацию
            clean = line.strip()
            if clean and clean[0].isdigit():
                clean = clean.split('.', 1)[-1].strip()
            if clean:
                queries.append(clean)
        
        return queries[:n + 1]  # Ограничиваем количество
    
    # ═══════════════════════════════════════════════════════════
    # Step-back Prompting - абстрагирование
    # ═══════════════════════════════════════════════════════════
    
    def stepback_query(self, query: str) -> str:
        """
        Step-back: Генерирует более общий вопрос для поиска background knowledge.
        
        "Почему небо голубое?" → "Как свет взаимодействует с атмосферой?"
        """
        prompt = f"""Для следующего конкретного вопроса сформулируй более общий, 
фундаментальный вопрос, ответ на который поможет ответить на исходный.

Пример:
Конкретный: "Почему Python медленнее C++?"
Общий: "Как работают интерпретируемые и компилируемые языки программирования?"

Конкретный вопрос: {query}
Общий вопрос:"""
        
        stepback = self._call_llm(prompt, max_tokens=100)
        return stepback.strip()


# ═══════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИЯ
# ═══════════════════════════════════════════════════════════════

def demo_query_transformation():
    """Демонстрация техник трансформации запросов"""
    
    transformer = QueryTransformer()
    
    query = "Как работает garbage collection в Python?"
    
    print("="*70)
    print("QUERY TRANSFORMATION")
    print("="*70)
    
    print(f"\n📌 Исходный запрос: '{query}'")
    
    # HyDE
    print("\n" + "-"*70)
    print("🔮 HyDE (Hypothetical Document):")
    print("-"*70)
    hyde_doc = transformer.hyde(query)
    print(hyde_doc)
    print("\n💡 Этот текст используется для поиска вместо вопроса!")
    
    # Query Expansion
    print("\n" + "-"*70)
    print("📈 Query Expansion:")
    print("-"*70)
    expanded = transformer.expand_query(query)
    print(expanded)
    
    # Multi-Query
    print("\n" + "-"*70)
    print("🔄 Multi-Query:")
    print("-"*70)
    multi = transformer.generate_multi_queries(query, n=3)
    for i, q in enumerate(multi, 1):
        print(f"  {i}. {q}")
    
    # Step-back
    print("\n" + "-"*70)
    print("⬆️ Step-back Query:")
    print("-"*70)
    stepback = transformer.stepback_query(query)
    print(stepback)
    print("\n💡 Сначала ищем по общему вопросу, потом по конкретному!")


if __name__ == "__main__":
    demo_query_transformation()
```

## 4. Обновление базы знаний

### Production-ready система обновления

```python
"""
Система управления базой знаний в production.
"""

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path


class KnowledgeBaseManager:
    """
    Менеджер базы знаний с поддержкой:
    - Инкрементального добавления
    - Обновления документов
    - Удаления устаревших
    - Версионирования
    """
    
    def __init__(self, rag_pipeline, storage_path: str = "./kb_storage"):
        self.rag = rag_pipeline
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Метаданные о документах
        self.document_registry: Dict[str, Dict] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Загружает реестр документов"""
        registry_path = self.storage_path / "registry.json"
        if registry_path.exists():
            with open(registry_path, 'r', encoding='utf-8') as f:
                self.document_registry = json.load(f)
    
    def _save_registry(self):
        """Сохраняет реестр документов"""
        registry_path = self.storage_path / "registry.json"
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.document_registry, f, ensure_ascii=False, indent=2)
    
    def _compute_hash(self, content: str) -> str:
        """Вычисляет хеш контента"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def add_document(
        self, 
        doc_id: str, 
        content: str, 
        metadata: Dict = None
    ) -> bool:
        """
        Добавляет новый документ или обновляет существующий.
        
        Returns:
            True если документ был добавлен/обновлён, False если без изменений
        """
        content_hash = self._compute_hash(content)
        
        # Проверяем, изменился ли документ
        if doc_id in self.document_registry:
            if self.document_registry[doc_id]["hash"] == content_hash:
                print(f"⏭️ Документ {doc_id} не изменился, пропускаем")
                return False
            
            # Документ изменился — нужно обновить
            print(f"🔄 Обновляем документ {doc_id}")
            self._remove_document_chunks(doc_id)
        else:
            print(f"➕ Добавляем новый документ {doc_id}")
        
        # Добавляем в RAG
        self.rag.add_documents([content], [doc_id])
        
        # Обновляем реестр
        self.document_registry[doc_id] = {
            "hash": content_hash,
            "added_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "chunk_count": len([d for d in self.rag.documents if doc_id in self.rag.metadata[self.rag.documents.index(d)].get("source", "")])
        }
        
        self._save_registry()
        return True
    
    def _remove_document_chunks(self, doc_id: str):
        """
        Удаляет чанки документа из индекса.
        
        Примечание: FAISS не поддерживает удаление напрямую.
        В production используйте Qdrant/Milvus или переиндексируйте.
        """
        # Для демонстрации просто помечаем как удалённые
        # В реальности нужна полная переиндексация или БД с поддержкой удаления
        print(f"⚠️ Удаление чанков {doc_id} (требует переиндексации)")
    
    def remove_document(self, doc_id: str):
        """Удаляет документ из базы"""
        if doc_id in self.document_registry:
            self._remove_document_chunks(doc_id)
            del self.document_registry[doc_id]
            self._save_registry()
            print(f"🗑️ Документ {doc_id} удалён")
        else:
            print(f"❌ Документ {doc_id} не найден")
    
    def remove_outdated(self, max_age_days: int = 30):
        """
        Удаляет документы старше указанного возраста.
        """
        cutoff = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        
        to_remove = []
        for doc_id, info in self.document_registry.items():
            updated_at = datetime.fromisoformat(info["updated_at"]).timestamp()
            if updated_at < cutoff:
                to_remove.append(doc_id)
        
        for doc_id in to_remove:
            self.remove_document(doc_id)
        
        print(f"🧹 Удалено {len(to_remove)} устаревших документов")
    
    def rebuild_index(self):
        """
        Полная переиндексация базы.
        Используйте при критических обновлениях.
        """
        print("🔄 Начинаем полную переиндексацию...")
        
        # Очищаем текущий индекс
        self.rag.index.reset()
        self.rag.documents = []
        self.rag.metadata = []
        
        # Переиндексируем все документы
        for doc_id, info in self.document_registry.items():
            # Загружаем контент документа
            # (в реальности нужно хранить контент или путь к файлу)
            print(f"  Переиндексация {doc_id}...")
        
        print("✅ Переиндексация завершена")
    
    def get_stats(self) -> Dict:
        """Статистика базы знаний"""
        return {
            "total_documents": len(self.document_registry),
            "total_chunks": len(self.rag.documents),
            "oldest_document": min(
                (info["added_at"] for info in self.document_registry.values()),
                default="N/A"
            ),
            "newest_document": max(
                (info["updated_at"] for info in self.document_registry.values()),
                default="N/A"
            )
        }
```

## 5. Мониторинг и логирование

```python
"""
Мониторинг и логирование RAG-системы.
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class QueryLog:
    """Лог одного запроса"""
    timestamp: str
    query: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    num_retrieved: int
    avg_relevance_score: float
    tokens_used: int
    answer_length: int
    confidence: str


class RAGMonitor:
    """
    Система мониторинга RAG pipeline.
    """
    
    def __init__(self, log_file: str = "rag_monitoring.log"):
        self.log_file = log_file
        self.query_logs: List[QueryLog] = []
        
        # Агрегированные метрики
        self.metrics = defaultdict(list)
        
        # Настройка логирования
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("RAGMonitor")
    
    def log_query(
        self,
        query: str,
        retrieval_time: float,
        generation_time: float,
        num_retrieved: int,
        avg_score: float,
        tokens_used: int,
        answer: str,
        confidence: str
    ):
        """Логирует запрос и его метрики"""
        
        total_time = retrieval_time + generation_time
        
        log_entry = QueryLog(
            timestamp=datetime.now().isoformat(),
            query=query,
            retrieval_time_ms=retrieval_time * 1000,
            generation_time_ms=generation_time * 1000,
            total_time_ms=total_time * 1000,
            num_retrieved=num_retrieved,
            avg_relevance_score=avg_score,
            tokens_used=tokens_used,
            answer_length=len(answer),
            confidence=confidence
        )
        
        self.query_logs.append(log_entry)
        
        # Обновляем метрики
        self.metrics["latency"].append(total_time * 1000)
        self.metrics["retrieval_scores"].append(avg_score)
        self.metrics["tokens"].append(tokens_used)
        
        # Логируем в файл
        self.logger.info(
            f"Query: '{query[:50]}...' | "
            f"Time: {total_time*1000:.0f}ms | "
            f"Docs: {num_retrieved} | "
            f"Score: {avg_score:.2f} | "
            f"Confidence: {confidence}"
        )
        
        # Алерты
        if total_time > 5:  # > 5 секунд
            self.logger.warning(f"Slow query detected: {total_time:.1f}s")
        
        if avg_score < 0.3:
            self.logger.warning(f"Low relevance query: {query[:50]}...")
    
    def get_dashboard_metrics(self) -> Dict:
        """Возвращает метрики для дашборда"""
        
        if not self.query_logs:
            return {"status": "no data"}
        
        latencies = self.metrics["latency"]
        scores = self.metrics["retrieval_scores"]
        tokens = self.metrics["tokens"]
        
        return {
            "total_queries": len(self.query_logs),
            "latency": {
                "avg_ms": sum(latencies) / len(latencies),
                "p50_ms": sorted(latencies)[len(latencies) // 2],
                "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 20 else max(latencies),
                "max_ms": max(latencies)
            },
            "relevance": {
                "avg_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "low_relevance_rate": sum(1 for s in scores if s < 0.5) / len(scores)
            },
            "tokens": {
                "total": sum(tokens),
                "avg_per_query": sum(tokens) / len(tokens)
            },
            "confidence_distribution": {
                "ВЫСОКАЯ": sum(1 for log in self.query_logs if log.confidence == "ВЫСОКАЯ"),
                "СРЕДНЯЯ": sum(1 for log in self.query_logs if log.confidence == "СРЕДНЯЯ"),
                "НИЗКАЯ": sum(1 for log in self.query_logs if log.confidence == "НИЗКАЯ"),
                "НЕТ ДАННЫХ": sum(1 for log in self.query_logs if log.confidence == "НЕТ ДАННЫХ")
            }
        }
    
    def print_report(self):
        """Выводит отчёт о работе системы"""
        
        metrics = self.get_dashboard_metrics()
        
        if metrics.get("status") == "no data":
            print("📊 Нет данных для отчёта")
            return
        
        print("\n" + "="*70)
        print("📊 ОТЧЁТ О РАБОТЕ RAG-СИСТЕМЫ")
        print("="*70)
        
        print(f"\n📈 ОБЩАЯ СТАТИСТИКА:")
        print(f"   Всего запросов: {metrics['total_queries']}")
        print(f"   Всего токенов: {metrics['tokens']['total']:,}")
        
        print(f"\n⏱️ ЛАТЕНТНОСТЬ:")
        print(f"   Средняя: {metrics['latency']['avg_ms']:.0f} ms")
        print(f"   P50: {metrics['latency']['p50_ms']:.0f} ms")
        print(f"   P95: {metrics['latency']['p95_ms']:.0f} ms")
        print(f"   Максимум: {metrics['latency']['max_ms']:.0f} ms")
        
        print(f"\n🎯 КАЧЕСТВО RETRIEVAL:")
        print(f"   Средний score: {metrics['relevance']['avg_score']:.2f}")
        print(f"   Минимальный: {metrics['relevance']['min_score']:.2f}")
        print(f"   % низкой релевантности: {metrics['relevance']['low_relevance_rate']:.1%}")
        
        print(f"\n📊 РАСПРЕДЕЛЕНИЕ УВЕРЕННОСТИ:")
        for conf, count in metrics['confidence_distribution'].items():
            bar = "█" * (count * 2)
            print(f"   {conf}: {bar} ({count})")
    
    def export_logs(self, filepath: str):
        """Экспортирует логи в JSON"""
        
        logs_data = [
            {
                "timestamp": log.timestamp,
                "query": log.query,
                "retrieval_time_ms": log.retrieval_time_ms,
                "generation_time_ms": log.generation_time_ms,
                "total_time_ms": log.total_time_ms,
                "num_retrieved": log.num_retrieved,
                "avg_relevance_score": log.avg_relevance_score,
                "tokens_used": log.tokens_used,
                "answer_length": log.answer_length,
                "confidence": log.confidence
            }
            for log in self.query_logs
        ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(logs_data, f, ensure_ascii=False, indent=2)
        
        print(f"📁 Логи экспортированы в {filepath}")
```

## 6. Оптимизация стоимости

```python
"""
Стратегии оптимизации стоимости RAG-системы.
"""

from typing import Dict
import os


class CostOptimizer:
    """
    Калькулятор и оптимизатор стоимости RAG.
    """
    
    # Примерные цены ($/1M токенов) — актуализируйте!
    PRICING = {
        "embeddings": {
            "text-embedding-3-small": 0.02,
            "text-embedding-3-large": 0.13,
            "text-embedding-ada-002": 0.10,
        },
        "llm": {
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
            "claude-3-sonnet": {"input": 3.0, "output": 15.0},
        }
    }
    
    @staticmethod
    def estimate_indexing_cost(
        num_documents: int,
        avg_doc_length: int,  # в символах
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 500
    ) -> Dict:
        """
        Оценивает стоимость индексации документов.
        """
        # Примерно 4 символа = 1 токен
        tokens_per_doc = avg_doc_length / 4
        total_tokens = num_documents * tokens_per_doc
        
        # Количество чанков
        num_chunks = int(total_tokens / (chunk_size / 4))
        
        # Стоимость эмбеддингов
        price_per_million = CostOptimizer.PRICING["embeddings"].get(
            embedding_model, 0.02
        )
        embedding_cost = (total_tokens / 1_000_000) * price_per_million
        
        return {
            "total_tokens": int(total_tokens),
            "num_chunks": num_chunks,
            "embedding_cost_usd": round(embedding_cost, 4),
            "embedding_model": embedding_model
        }
    
    @staticmethod
    def estimate_query_cost(
        num_queries: int,
        avg_context_tokens: int = 2000,
        avg_response_tokens: int = 300,
        llm_model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-3-small"
    ) -> Dict:
        """
        Оценивает стоимость запросов.
        """
        # Эмбеддинг запроса
        query_embedding_tokens = num_queries * 20  # ~20 токенов на запрос
        embedding_price = CostOptimizer.PRICING["embeddings"].get(
            embedding_model, 0.02
        )
        embedding_cost = (query_embedding_tokens / 1_000_000) * embedding_price
        
        # LLM генерация
        llm_prices = CostOptimizer.PRICING["llm"].get(
            llm_model, {"input": 0.50, "output": 1.50}
        )
        
        total_input_tokens = num_queries * avg_context_tokens
        total_output_tokens = num_queries * avg_response_tokens
        
        llm_input_cost = (total_input_tokens / 1_000_000) * llm_prices["input"]
        llm_output_cost = (total_output_tokens / 1_000_000) * llm_prices["output"]
        
        total_cost = embedding_cost + llm_input_cost + llm_output_cost
        
        return {
            "num_queries": num_queries,
            "embedding_cost_usd": round(embedding_cost, 4),
            "llm_input_cost_usd": round(llm_input_cost, 4),
            "llm_output_cost_usd": round(llm_output_cost, 4),
            "total_cost_usd": round(total_cost, 4),
            "cost_per_query_usd": round(total_cost / num_queries, 6)
        }
    
    @staticmethod
    def suggest_optimizations(current_config: Dict) -> list:
        """
        Предлагает оптимизации на основе текущей конфигурации.
        """
        suggestions = []
        
        # Проверяем модель эмбеддингов
        if current_config.get("embedding_model") == "text-embedding-3-large":
            suggestions.append({
                "type": "embedding_model",
                "suggestion": "Используйте text-embedding-3-small",
                "potential_savings": "85% на эмбеддинги",
                "tradeoff": "Небольшое снижение качества retrieval"
            })
        
        # Проверяем LLM модель
        if current_config.get("llm_model") in ["gpt-4", "claude-3-opus"]:
            suggestions.append({
                "type": "llm_model",
                "suggestion": "Используйте gpt-4o-mini или claude-3-haiku для простых запросов",
                "potential_savings": "90%+ на генерацию",
                "tradeoff": "Роутинг по сложности запроса"
            })
        
        # Кэширование
        if not current_config.get("caching_enabled"):
            suggestions.append({
                "type": "caching",
                "suggestion": "Включите кэширование эмбеддингов",
                "potential_savings": "До 50% на повторных запросах",
                "tradeoff": "Дополнительное хранилище"
            })
        
        # Размер контекста
        if current_config.get("avg_context_size", 0) > 3000:
            suggestions.append({
                "type": "context_size",
                "suggestion": "Уменьшите количество документов в контексте",
                "potential_savings": "30-50% на LLM токенах",
                "tradeoff": "Возможно снижение полноты ответов"
            })
        
        return suggestions


# ═══════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИЯ
# ═══════════════════════════════════════════════════════════════

def demo_cost_optimization():
    """Демонстрация расчёта и оптимизации стоимости"""
    
    print("="*70)
    print("💰 АНАЛИЗ СТОИМОСТИ RAG-СИСТЕМЫ")
    print("="*70)
    
    # Сценарий: корпоративная база знаний
    print("\n📊 СЦЕНАРИЙ: Корпоративная база знаний")
    print("-"*70)
    
    # Индексация
    indexing = CostOptimizer.estimate_indexing_cost(
        num_documents=1000,
        avg_doc_length=5000,
        embedding_model="text-embedding-3-small"
    )
    
    print(f"\n📄 ИНДЕКСАЦИЯ ({indexing['num_chunks']} чанков):")
    print(f"   Токенов: {indexing['total_tokens']:,}")
    print(f"   Стоимость: ${indexing['embedding_cost_usd']}")
    
    # Запросы
    query_cost = CostOptimizer.estimate_query_cost(
        num_queries=10000,
        avg_context_tokens=2000,
        avg_response_tokens=300,
        llm_model="gpt-3.5-turbo"
    )
    
    print(f"\n❓ ЗАПРОСЫ (10,000/месяц):")
    print(f"   Эмбеддинги: ${query_cost['embedding_cost_usd']}")
    print(f"   LLM input: ${query_cost['llm_input_cost_usd']}")
    print(f"   LLM output: ${query_cost['llm_output_cost_usd']}")
    print(f"   ИТОГО: ${query_cost['total_cost_usd']}")
    print(f"   За запрос: ${query_cost['cost_per_query_usd']}")
    
    # Оптимизации
    print("\n💡 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ:")
    print("-"*70)
    
    suggestions = CostOptimizer.suggest_optimizations({
        "embedding_model": "text-embedding-3-small",
        "llm_model": "gpt-3.5-turbo",
        "caching_enabled": False,
        "avg_context_size": 2000
    })
    
    for i, s in enumerate(suggestions, 1):
        print(f"\n{i}. {s['suggestion']}")
        print(f"   Экономия: {s['potential_savings']}")
        print(f"   Компромисс: {s['tradeoff']}")


if __name__ == "__main__":
    demo_cost_optimization()
```

## Практические задания

### 🟢 Базовый уровень

**Задание 1: Гибридный поиск**

1. Установите `rank-bm25`
2. Реализуйте `HybridSearch` на своих документах
3. Сравните результаты с чистым семантическим поиском
4. Найдите запрос, где гибридный поиск лучше

**Задание 2: Query Expansion**

1. Реализуйте `expand_query` из `QueryTransformer`
2. Протестируйте на 5 запросах
3. Сравните retrieval до и после расширения

### 🟡 Средний уровень

**Задание 3: HyDE implementation**

1. Реализуйте полный HyDE pipeline:
   - Генерация гипотетического документа
   - Эмбеддинг гипотетического документа
   - Поиск по нему
2. Сравните с обычным поиском на 10 вопросах
3. Измерьте улучшение метрик

**Задание 4: Мониторинг**

1. Интегрируйте `RAGMonitor` в ваш pipeline
2. Запустите 20+ запросов
3. Проанализируйте отчёт
4. Идентифицируйте проблемные запросы

### 🔴 Продвинутый уровень

**Задание 5: Production-ready RAG**

Создайте полноценную систему с:
1. Гибридным поиском
2. Query transformation (выбор лучшей техники)
3. Мониторингом и алертами
4. API эндпоинтами (FastAPI)
5. Документацией

**Задание 6: Финальный проект модуля**

Создайте QA-бота по документации выбранного фреймворка:

1. Соберите документацию (scraping/API)
2. Реализуйте оптимизированный RAG pipeline
3. Добавьте веб-интерфейс
4. Проведите оценку качества
5. Напишите отчёт с метриками

## Контрольные вопросы

1. **Что такое гибридный поиск и зачем он нужен?**
   <details>
   <summary>Ответ</summary>
   Гибридный поиск — комбинация семантического (по смыслу) и keyword (BM25) поиска. Нужен потому что: 1) семантический поиск может пропустить точные совпадения (коды, названия), 2) BM25 не понимает синонимы. Комбинация даёт лучшие результаты.
   </details>

2. **Как работает HyDE?**
   <details>
   <summary>Ответ</summary>
   HyDE (Hypothetical Document Embeddings) генерирует с помощью LLM гипотетический документ, который бы отвечал на вопрос, затем ищет по эмбеддингу этого документа вместо вопроса. Это помогает найти релевантные документы, когда вопрос и ответ формулируются по-разному.
   </details>

3. **Какие метрики важны для мониторинга RAG?**
   <details>
   <summary>Ответ</summary>
   1) Латентность (время ответа), 2) Качество retrieval (avg score), 3) Использование токенов, 4) Распределение уверенности, 5) Доля запросов с низкой релевантностью. Важны алерты на аномалии.
   </details>

4. **Как оптимизировать стоимость RAG?**
   <details>
   <summary>Ответ</summary>
   1) Использовать дешёвые модели эмбеддингов (text-embedding-3-small), 2) Роутинг по сложности (простые запросы → дешёвые LLM), 3) Кэширование эмбеддингов, 4) Оптимизация размера контекста, 5) Батчинг запросов.
   </details>

5. **Что такое Sentence Window Retrieval?**
   <details>
   <summary>Ответ</summary>
   Техника chunking, где для поиска используются маленькие чанки (точность), но при генерации ответа подаётся расширенный контекст (окно из соседних предложений). Сочетает точность поиска с полнотой контекста.
   </details>

## Заключение модуля

### Что мы изучили в Модуле 4

1. **Урок 1**: Зачем нужен RAG — проблемы LLM и как RAG их решает
2. **Урок 2**: Эмбеддинги и векторные базы — как искать по смыслу
3. **Урок 3**: RAG Pipeline — полный цикл от вопроса до ответа
4. **Урок 4**: Продвинутые техники — гибридный поиск, HyDE, мониторинг

### Что вы создали

- ✅ Систему получения и сравнения эмбеддингов
- ✅ Векторную базу данных с семантическим поиском
- ✅ Полноценный RAG pipeline
- ✅ QA-бота с мониторингом и оптимизациями

### Связь с предыдущими модулями

- **Модуль 2**: LLM-as-a-judge используется для оценки RAG
- **Модуль 3**: SchoolBot можно расширить RAG для работы с учебниками!

### Что дальше

В **Модуле 5: Агенты** мы научим LLM:
- Использовать инструменты (API, поиск, калькулятор)
- Планировать и выполнять многошаговые задачи
- Автономно решать сложные проблемы

RAG станет одним из инструментов агента!

### Ваш прогресс

🎯 **Отличная работа!** Вы освоили:
- ✅ Концепцию и архитектуру RAG
- ✅ Работу с эмбеддингами и векторными базами
- ✅ Создание production-ready RAG систем
- ✅ Продвинутые техники оптимизации

**Готовы к агентам?** Переходите к [Модулю 5: Создание автономных агентных систем](../module_5/README.md)!

---

## Дополнительные материалы

### Статьи:
- [HyDE: Hypothetical Document Embeddings](https://arxiv.org/abs/2212.10496)
- [From Local to Global: A Hybrid RAG Approach](https://arxiv.org/abs/2404.16130)
- [Benchmarking Large Language Models in Retrieval-Augmented Generation](https://arxiv.org/abs/2309.01431)

### Инструменты:
- [LangChain Advanced RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [LlamaIndex](https://www.llamaindex.ai/)
- [Instructor](https://github.com/jxnl/instructor) — structured outputs
- [Phoenix by Arize](https://phoenix.arize.com/) — LLM observability

### Курсы:
- [Building Production RAG Applications](https://www.deeplearning.ai/short-courses/)
- [Advanced Retrieval for AI](https://www.deeplearning.ai/short-courses/)

