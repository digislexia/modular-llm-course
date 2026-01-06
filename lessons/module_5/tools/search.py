"""
Инструменты поиска для LLM-агента.

Включает:
- Wikipedia поиск (русская и английская)
- DuckDuckGo поиск (веб-поиск)

Пример:
    >>> wikipedia_search("Python programming")
    "Python — высокоуровневый язык программирования..."
"""

import requests
from typing import Optional
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Tool:
    """Базовый класс для инструмента агента"""
    name: str
    description: str
    func: Callable
    parameters: dict = field(default_factory=dict)
    
    def to_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys())
                }
            }
        }
    
    def execute(self, **kwargs) -> str:
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            return f"Ошибка: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# WIKIPEDIA ПОИСК
# ═══════════════════════════════════════════════════════════════════════════════

def wikipedia_search(query: str, language: str = "ru", sentences: int = 3) -> str:
    """
    Ищет информацию в Wikipedia.
    
    Args:
        query: Поисковый запрос
        language: Язык ('ru' или 'en')
        sentences: Количество предложений в ответе (1-10)
        
    Returns:
        Краткое описание из Wikipedia или сообщение об ошибке
        
    Examples:
        >>> wikipedia_search("Python")
        "Python — высокоуровневый язык программирования..."
    """
    sentences = max(1, min(10, sentences))
    
    # Формируем URL
    base_url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/"
    url = base_url + query.replace(" ", "_")
    
    headers = {
        "User-Agent": "LLMAgent/1.0 (Educational project)"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        # Если страница не найдена — пробуем поиск
        if response.status_code == 404:
            return _wikipedia_search_fallback(query, language, sentences)
        
        if response.status_code != 200:
            return f"Ошибка Wikipedia API: {response.status_code}"
        
        data = response.json()
        
        # Получаем краткое описание
        extract = data.get("extract", "")
        
        if not extract:
            return f"Статья '{query}' не содержит описания."
        
        # Ограничиваем количество предложений
        sentences_list = extract.split('. ')
        result = '. '.join(sentences_list[:sentences])
        
        if not result.endswith('.'):
            result += '.'
        
        # Добавляем заголовок
        title = data.get("title", query)
        
        return f"[{title}] {result}"
        
    except requests.exceptions.Timeout:
        return "Ошибка: превышено время ожидания Wikipedia"
    except requests.exceptions.RequestException as e:
        return f"Ошибка сети: {str(e)}"
    except Exception as e:
        return f"Ошибка поиска в Wikipedia: {str(e)}"


def _wikipedia_search_fallback(query: str, language: str, sentences: int) -> str:
    """
    Резервный поиск через Wikipedia Search API.
    
    Используется когда прямой URL не работает.
    """
    search_url = (
        f"https://{language}.wikipedia.org/w/api.php?"
        f"action=query&list=search&srsearch={query}&format=json"
    )
    
    try:
        response = requests.get(search_url, timeout=10)
        data = response.json()
        
        search_results = data.get("query", {}).get("search", [])
        
        if not search_results:
            return f"Ничего не найдено по запросу: {query}"
        
        # Берём первый результат и повторяем поиск
        first_title = search_results[0]["title"]
        return wikipedia_search(first_title, language, sentences)
        
    except Exception as e:
        return f"Ошибка поиска: {str(e)}"


# Создаём инструмент
wikipedia_tool = Tool(
    name="wikipedia",
    description=(
        "Ищет информацию в Wikipedia (русская или английская). "
        "Используй для поиска фактов, определений, биографий, "
        "исторических событий, научных концепций."
    ),
    func=wikipedia_search,
    parameters={
        "query": {
            "type": "string",
            "description": "Поисковый запрос на русском или английском"
        }
    }
)


# ═══════════════════════════════════════════════════════════════════════════════
# DUCKDUCKGO ПОИСК (Instant Answers API)
# ═══════════════════════════════════════════════════════════════════════════════

def duckduckgo_search(query: str) -> str:
    """
    Ищет информацию через DuckDuckGo Instant Answers API.
    
    Бесплатный API без ключа. Возвращает краткие ответы.
    
    Args:
        query: Поисковый запрос
        
    Returns:
        Результат поиска или сообщение об ошибке
        
    Note:
        Этот API возвращает только "instant answers", а не полноценный веб-поиск.
    """
    url = "https://api.duckduckgo.com/"
    
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1
    }
    
    headers = {
        "User-Agent": "LLMAgent/1.0"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return f"Ошибка DuckDuckGo API: {response.status_code}"
        
        data = response.json()
        
        # Пробуем разные поля ответа
        
        # Abstract (основной результат)
        if data.get("Abstract"):
            source = data.get("AbstractSource", "DuckDuckGo")
            return f"[{source}] {data['Abstract']}"
        
        # Answer (прямой ответ)
        if data.get("Answer"):
            return f"[Answer] {data['Answer']}"
        
        # Definition
        if data.get("Definition"):
            source = data.get("DefinitionSource", "Dictionary")
            return f"[{source}] {data['Definition']}"
        
        # Related topics
        related = data.get("RelatedTopics", [])
        if related and isinstance(related[0], dict):
            texts = []
            for topic in related[:3]:
                if "Text" in topic:
                    texts.append(topic["Text"])
            if texts:
                return "Связанные темы:\n• " + "\n• ".join(texts)
        
        return f"Не найдено прямого ответа на: {query}"
        
    except requests.exceptions.Timeout:
        return "Ошибка: превышено время ожидания"
    except Exception as e:
        return f"Ошибка поиска: {str(e)}"


duckduckgo_tool = Tool(
    name="web_search",
    description=(
        "Выполняет веб-поиск через DuckDuckGo. "
        "Используй для получения кратких ответов на общие вопросы. "
        "Для детальной информации лучше использовать wikipedia."
    ),
    func=duckduckgo_search,
    parameters={
        "query": {
            "type": "string",
            "description": "Поисковый запрос на английском (лучше работает)"
        }
    }
)


# ═══════════════════════════════════════════════════════════════════════════════
# ТЕСТЫ
# ═══════════════════════════════════════════════════════════════════════════════

def test_wikipedia():
    """Тестирование Wikipedia поиска"""
    
    # Тест на русском
    result = wikipedia_search("Python", "ru")
    assert len(result) > 50
    assert "Ошибка" not in result
    print(f"✅ Wikipedia (ru): {result[:100]}...")
    
    # Тест на английском
    result = wikipedia_search("Python", "en")
    assert len(result) > 50
    assert "Ошибка" not in result
    print(f"✅ Wikipedia (en): {result[:100]}...")
    
    # Тест несуществующей страницы
    result = wikipedia_search("asdfghjklzxcvbnm12345", "ru")
    # Должен сработать fallback или вернуть "не найдено"
    print(f"✅ Wikipedia (not found): {result[:100]}...")
    
    print("\n✅ Все тесты Wikipedia пройдены!")


def test_duckduckgo():
    """Тестирование DuckDuckGo поиска"""
    
    result = duckduckgo_search("Python programming language")
    print(f"✅ DuckDuckGo: {result[:200]}...")
    
    print("\n✅ Тест DuckDuckGo пройден!")


if __name__ == "__main__":
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ПОИСКОВЫХ ИНСТРУМЕНТОВ")
    print("=" * 60)
    
    test_wikipedia()
    print()
    test_duckduckgo()

