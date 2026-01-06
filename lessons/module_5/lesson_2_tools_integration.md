# Урок 2: Интеграция инструментов (Function Calling)

## Введение

LLM — мощный инструмент для рассуждений и генерации текста. Но у него есть ограничения:
- Не может выполнять точные вычисления
- Не имеет доступа к актуальной информации
- Не может взаимодействовать с внешним миром

**Решение:** Function Calling — механизм, позволяющий модели вызывать внешние функции.

В этом уроке мы создадим набор инструментов и научим агента их использовать.

## Цели урока

После завершения урока вы сможете:

- ✅ Понимать концепцию инструментов (tools) для LLM
- ✅ Использовать Function Calling в OpenAI/OpenRouter API
- ✅ Создавать собственные инструменты
- ✅ Реализовать агента с набором инструментов

## Ключевые термины

- **Инструмент (Tool)** — внешняя функция, которую может вызвать LLM
- **Function Calling** — механизм API для вызова функций из LLM
- **Tool Schema** — JSON-описание функции для модели
- **ToolKit** — набор инструментов агента

## 1. Концепция инструментов

### Зачем LLM нужны инструменты?

```
┌────────────────────────────────────────────────────────────────┐
│                    БЕЗ ИНСТРУМЕНТОВ                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Вопрос: "Сколько будет 23847 × 9182?"                        │
│                                                                │
│  LLM: "Примерно 219 миллионов..."  ← НЕТОЧНО! (218,961,354)   │
│                                                                │
│  Вопрос: "Какая сейчас погода в Москве?"                      │
│                                                                │
│  LLM: "Обычно в это время года..."  ← НЕ ЗНАЕТ АКТУАЛЬНО!     │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                    С ИНСТРУМЕНТАМИ                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Вопрос: "Сколько будет 23847 × 9182?"                        │
│                                                                │
│  LLM: [вызывает calculator("23847 * 9182")]                   │
│       "Результат: 218,961,354"  ← ТОЧНО!                      │
│                                                                │
│  Вопрос: "Какая сейчас погода в Москве?"                      │
│                                                                │
│  LLM: [вызывает weather_api("Moscow")]                        │
│       "Сейчас в Москве +15°C, облачно"  ← АКТУАЛЬНО!          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Примеры инструментов

| Категория | Инструмент | Что делает |
|-----------|------------|------------|
| **Вычисления** | calculator | Математические операции |
| **Информация** | wikipedia | Поиск в энциклопедии |
| **Время** | datetime | Текущая дата/время |
| **Погода** | weather | Прогноз погоды |
| **Файлы** | file_read/write | Работа с файлами |
| **Код** | python_exec | Выполнение Python |
| **API** | http_request | HTTP-запросы |

## 2. Function Calling в API

### Как это работает

```
┌─────────────────────────────────────────────────────────────────┐
│                       FUNCTION CALLING                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Пользователь: "Сколько будет 15% от 850?"                  │
│                          │                                      │
│                          ▼                                      │
│  2. Приложение → API:                                          │
│     - Сообщение пользователя                                   │
│     - Список доступных функций (tools)                         │
│                          │                                      │
│                          ▼                                      │
│  3. LLM решает: "Нужен калькулятор!"                          │
│     Возвращает: tool_call = {                                  │
│       "function": "calculator",                                │
│       "arguments": {"expression": "850 * 0.15"}               │
│     }                                                          │
│                          │                                      │
│                          ▼                                      │
│  4. Приложение выполняет функцию:                              │
│     result = calculator("850 * 0.15") → 127.5                  │
│                          │                                      │
│                          ▼                                      │
│  5. Приложение → API:                                          │
│     - Результат функции: 127.5                                 │
│                          │                                      │
│                          ▼                                      │
│  6. LLM формирует ответ:                                       │
│     "15% от 850 равно 127.5"                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Формат описания функции (Tool Schema)

```python
"""
Пример описания инструмента для OpenAI API.
"""

# Описание функции в формате JSON Schema
tool_schema = {
    "type": "function",
    "function": {
        "name": "calculator",                          # Имя функции
        "description": "Выполняет математические вычисления. "
                       "Используй для любых расчётов.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Математическое выражение. "
                                   "Примеры: '2 + 2', '15 * 3.14', 'sqrt(16)'"
                }
            },
            "required": ["expression"]                 # Обязательные параметры
        }
    }
}
```

### Полный пример с OpenRouter

```python
"""
Пример использования Function Calling с OpenRouter API.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()


def call_openrouter_with_tools(
    messages: list,
    tools: list,
    model: str = "openai/gpt-4-turbo-preview"
) -> dict:
    """
    Вызов OpenRouter API с поддержкой function calling.
    
    Args:
        messages: История сообщений
        tools: Список доступных инструментов
        model: Модель для использования
        
    Returns:
        Ответ API
    """
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto"  # Модель сама решает, вызывать ли функцию
        }
    )
    
    return response.json()


# ═══════════════════════════════════════════════════════════════
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ═══════════════════════════════════════════════════════════════

# Описание инструмента
calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Выполняет математические вычисления",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Математическое выражение, например '2 + 2 * 3'"
                }
            },
            "required": ["expression"]
        }
    }
}

# Запрос пользователя
messages = [
    {"role": "system", "content": "Ты — полезный ассистент с калькулятором."},
    {"role": "user", "content": "Сколько будет 23 * 17?"}
]

# Вызов API
response = call_openrouter_with_tools(messages, [calculator_tool])

# Обработка ответа
message = response["choices"][0]["message"]

if "tool_calls" in message:
    # Модель хочет вызвать функцию
    tool_call = message["tool_calls"][0]
    function_name = tool_call["function"]["name"]
    arguments = json.loads(tool_call["function"]["arguments"])
    
    print(f"Модель вызывает: {function_name}({arguments})")
    
    # Выполняем функцию
    if function_name == "calculator":
        result = eval(arguments["expression"])  # В реальности — безопасный калькулятор!
        print(f"Результат: {result}")
else:
    # Модель отвечает напрямую
    print(f"Ответ: {message['content']}")
```

## 3. Создание набора инструментов

### Базовый класс Tool

```python
"""
Базовые классы для инструментов агента.
"""

from dataclasses import dataclass, field
from typing import Callable, Any
import json


@dataclass
class Tool:
    """
    Инструмент для LLM-агента.
    
    Attributes:
        name: Уникальное имя инструмента
        description: Описание для LLM (когда использовать)
        func: Python-функция для выполнения
        parameters: Описание параметров (JSON Schema)
    """
    name: str
    description: str
    func: Callable
    parameters: dict = field(default_factory=dict)
    
    def to_schema(self) -> dict:
        """
        Преобразует инструмент в формат для API.
        """
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
        """
        Выполняет инструмент с заданными аргументами.
        
        Returns:
            Результат в виде строки
        """
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            return f"Ошибка: {str(e)}"
    
    def __repr__(self):
        return f"Tool(name='{self.name}')"


class ToolKit:
    """
    Набор инструментов для агента.
    
    Позволяет регистрировать, получать и выполнять инструменты.
    """
    
    def __init__(self):
        self.tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """
        Регистрирует инструмент.
        
        Args:
            tool: Инструмент для регистрации
        """
        self.tools[tool.name] = tool
        print(f"✅ Зарегистрирован инструмент: {tool.name}")
    
    def get(self, name: str) -> Tool:
        """
        Получает инструмент по имени.
        """
        if name not in self.tools:
            raise KeyError(f"Инструмент '{name}' не найден. "
                           f"Доступны: {list(self.tools.keys())}")
        return self.tools[name]
    
    def get_schemas(self) -> list:
        """
        Возвращает схемы всех инструментов для API.
        """
        return [tool.to_schema() for tool in self.tools.values()]
    
    def execute(self, name: str, **kwargs) -> str:
        """
        Выполняет инструмент по имени.
        """
        tool = self.get(name)
        return tool.execute(**kwargs)
    
    def list_tools(self) -> list:
        """
        Возвращает список доступных инструментов.
        """
        return list(self.tools.keys())
    
    def describe(self) -> str:
        """
        Возвращает описание всех инструментов.
        """
        descriptions = []
        for tool in self.tools.values():
            params = ", ".join(tool.parameters.keys())
            descriptions.append(f"• {tool.name}({params}): {tool.description}")
        return "\n".join(descriptions)
```

### Реализация инструментов

#### 1. Безопасный калькулятор

```python
"""
Безопасный калькулятор для агента.
Не использует eval() напрямую!
"""

import ast
import operator
import math


def safe_calculator(expression: str) -> float:
    """
    Безопасно вычисляет математическое выражение.
    
    Поддерживает: +, -, *, /, **, sqrt, sin, cos, tan, log, abs
    
    Args:
        expression: Математическое выражение
        
    Returns:
        Результат вычисления
        
    Examples:
        >>> safe_calculator("2 + 2")
        4.0
        >>> safe_calculator("sqrt(16)")
        4.0
        >>> safe_calculator("3.14 * 2 ** 2")
        12.56
    """
    # Разрешённые операторы
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # Разрешённые функции
    FUNCTIONS = {
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'abs': abs,
        'round': round,
        'floor': math.floor,
        'ceil': math.ceil,
    }
    
    # Константы
    CONSTANTS = {
        'pi': math.pi,
        'e': math.e,
    }
    
    def _eval(node):
        """Рекурсивно вычисляет AST-узел"""
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in CONSTANTS:
                return CONSTANTS[node.id]
            raise ValueError(f"Неизвестная константа: {node.id}")
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op = OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Неподдерживаемый оператор: {type(node.op).__name__}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op = OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Неподдерживаемый оператор: {type(node.op).__name__}")
            return op(operand)
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name not in FUNCTIONS:
                raise ValueError(f"Неизвестная функция: {func_name}")
            args = [_eval(arg) for arg in node.args]
            return FUNCTIONS[func_name](*args)
        else:
            raise ValueError(f"Неподдерживаемый тип узла: {type(node).__name__}")
    
    # Парсим выражение
    try:
        tree = ast.parse(expression, mode='eval')
        result = _eval(tree.body)
        return float(result)
    except Exception as e:
        raise ValueError(f"Ошибка вычисления '{expression}': {str(e)}")


# Создаём инструмент
calculator_tool = Tool(
    name="calculator",
    description="Выполняет математические вычисления. "
                "Поддерживает: +, -, *, /, **, sqrt, sin, cos, log, abs, pi, e. "
                "Используй для любых числовых расчётов.",
    func=safe_calculator,
    parameters={
        "expression": {
            "type": "string",
            "description": "Математическое выражение. Примеры: '2 + 2', 'sqrt(16)', 'pi * 2'"
        }
    }
)
```

#### 2. Поиск в Wikipedia

```python
"""
Инструмент поиска в Wikipedia.
"""

import requests
from typing import Optional


def wikipedia_search(query: str, sentences: int = 3) -> str:
    """
    Ищет информацию в Wikipedia.
    
    Args:
        query: Поисковый запрос
        sentences: Количество предложений в ответе (1-10)
        
    Returns:
        Краткое описание из Wikipedia или сообщение об ошибке
        
    Examples:
        >>> wikipedia_search("Python programming")
        "Python is a high-level programming language..."
    """
    sentences = max(1, min(10, sentences))
    
    # Используем Wikipedia API
    url = "https://ru.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_")
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 404:
            # Пробуем поиск
            search_url = f"https://ru.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
            search_response = requests.get(search_url, timeout=10)
            search_data = search_response.json()
            
            if search_data.get("query", {}).get("search"):
                first_result = search_data["query"]["search"][0]["title"]
                # Рекурсивный вызов с найденным заголовком
                return wikipedia_search(first_result, sentences)
            else:
                return f"Ничего не найдено по запросу: {query}"
        
        data = response.json()
        
        # Получаем extract (краткое описание)
        extract = data.get("extract", "")
        
        if not extract:
            return f"Статья '{query}' не содержит описания."
        
        # Ограничиваем количество предложений
        sentences_list = extract.split('. ')
        result = '. '.join(sentences_list[:sentences])
        
        if not result.endswith('.'):
            result += '.'
        
        return result
        
    except requests.exceptions.Timeout:
        return "Ошибка: превышено время ожидания Wikipedia"
    except Exception as e:
        return f"Ошибка поиска в Wikipedia: {str(e)}"


# Альтернативная версия с использованием библиотеки wikipedia
def wikipedia_search_v2(query: str, sentences: int = 3) -> str:
    """
    Ищет информацию в Wikipedia (версия с библиотекой).
    
    Требует: pip install wikipedia-api
    """
    try:
        import wikipediaapi
        
        wiki = wikipediaapi.Wikipedia(
            language='ru',
            user_agent='SchoolBot/1.0 (educational project)'
        )
        
        page = wiki.page(query)
        
        if not page.exists():
            return f"Страница '{query}' не найдена в Wikipedia"
        
        # Берём первые N предложений из summary
        summary = page.summary
        sentences_list = summary.split('. ')
        result = '. '.join(sentences_list[:sentences])
        
        if not result.endswith('.'):
            result += '.'
        
        return result
        
    except ImportError:
        return "Библиотека wikipedia-api не установлена. Запустите: pip install wikipedia-api"
    except Exception as e:
        return f"Ошибка: {str(e)}"


# Создаём инструмент
wikipedia_tool = Tool(
    name="wikipedia",
    description="Ищет информацию в Wikipedia. "
                "Используй для поиска фактов, определений, биографий, исторических событий.",
    func=wikipedia_search,
    parameters={
        "query": {
            "type": "string",
            "description": "Поисковый запрос на русском или английском"
        }
    }
)
```

#### 3. Текущая дата и время

```python
"""
Инструмент для работы с датой и временем.
"""

from datetime import datetime, timedelta
import pytz


def get_current_datetime(timezone: str = "Europe/Moscow") -> str:
    """
    Возвращает текущую дату и время.
    
    Args:
        timezone: Часовой пояс (по умолчанию Москва)
        
    Returns:
        Форматированная строка с датой и временем
    """
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        
        # Форматируем красиво
        weekdays = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
        months = ["января", "февраля", "марта", "апреля", "мая", "июня",
                  "июля", "августа", "сентября", "октября", "ноября", "декабря"]
        
        weekday = weekdays[now.weekday()]
        month = months[now.month - 1]
        
        return (f"{now.day} {month} {now.year} года, {weekday}, "
                f"{now.hour:02d}:{now.minute:02d} ({timezone})")
    
    except Exception as e:
        return f"Ошибка получения времени: {str(e)}"


def calculate_date(operation: str) -> str:
    """
    Вычисляет даты (через N дней, разница между датами).
    
    Args:
        operation: Операция с датой. Примеры:
            - "через 7 дней"
            - "30 дней назад"
            - "дней до 2024-12-31"
    
    Returns:
        Результат вычисления
    """
    now = datetime.now()
    
    operation_lower = operation.lower()
    
    try:
        # Через N дней
        if "через" in operation_lower and "дн" in operation_lower:
            import re
            match = re.search(r'через\s*(\d+)\s*дн', operation_lower)
            if match:
                days = int(match.group(1))
                future_date = now + timedelta(days=days)
                return future_date.strftime("%d.%m.%Y")
        
        # N дней назад
        if "назад" in operation_lower and "дн" in operation_lower:
            import re
            match = re.search(r'(\d+)\s*дн\w*\s*назад', operation_lower)
            if match:
                days = int(match.group(1))
                past_date = now - timedelta(days=days)
                return past_date.strftime("%d.%m.%Y")
        
        # Дней до даты
        if "дней до" in operation_lower:
            import re
            match = re.search(r'(\d{4})-(\d{2})-(\d{2})', operation)
            if match:
                target = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                diff = (target - now).days
                return f"{diff} дней"
        
        return "Не удалось распознать операцию. Примеры: 'через 7 дней', '30 дней назад', 'дней до 2024-12-31'"
        
    except Exception as e:
        return f"Ошибка: {str(e)}"


# Создаём инструменты
datetime_tool = Tool(
    name="current_datetime",
    description="Возвращает текущую дату и время. "
                "Используй, когда нужно узнать сегодняшнюю дату или время.",
    func=get_current_datetime,
    parameters={
        "timezone": {
            "type": "string",
            "description": "Часовой пояс (по умолчанию Europe/Moscow)"
        }
    }
)

date_calculator_tool = Tool(
    name="date_calculator",
    description="Вычисляет даты: через N дней, N дней назад, дней до определённой даты.",
    func=calculate_date,
    parameters={
        "operation": {
            "type": "string",
            "description": "Операция: 'через 7 дней', '30 дней назад', 'дней до 2024-12-31'"
        }
    }
)
```

#### 4. HTTP-запросы (универсальный инструмент)

```python
"""
Универсальный инструмент для HTTP-запросов.
Осторожно: может использоваться для доступа к внешним API.
"""

import requests
from typing import Optional


def http_get(url: str, headers: Optional[dict] = None) -> str:
    """
    Выполняет GET-запрос.
    
    Args:
        url: URL для запроса
        headers: Дополнительные заголовки
        
    Returns:
        Текст ответа или сообщение об ошибке
    """
    # Ограничение на домены (безопасность)
    ALLOWED_DOMAINS = [
        "api.openweathermap.org",
        "api.exchangerate-api.com",
        "ru.wikipedia.org",
        "en.wikipedia.org",
    ]
    
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    
    if domain not in ALLOWED_DOMAINS:
        return f"Домен '{domain}' не разрешён. Разрешены: {ALLOWED_DOMAINS}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # Ограничиваем размер ответа
            text = response.text[:5000]
            if len(response.text) > 5000:
                text += "\n... (ответ обрезан)"
            return text
        else:
            return f"Ошибка HTTP {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "Ошибка: превышено время ожидания"
    except Exception as e:
        return f"Ошибка запроса: {str(e)}"


http_tool = Tool(
    name="http_get",
    description="Выполняет HTTP GET-запрос к разрешённым API. "
                "Используй для получения данных из внешних сервисов.",
    func=http_get,
    parameters={
        "url": {
            "type": "string",
            "description": "URL для запроса"
        }
    }
)
```

## 4. Агент с инструментами

### Полная реализация

```python
"""
Простой агент с инструментами.
Демонстрирует интеграцию Function Calling.
"""

import os
import json
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class SimpleToolAgent:
    """
    Агент с поддержкой инструментов.
    
    Использует OpenRouter API для function calling.
    """
    
    def __init__(
        self,
        toolkit: ToolKit,
        model: str = "openai/gpt-4-turbo-preview",
        system_prompt: Optional[str] = None
    ):
        """
        Инициализация агента.
        
        Args:
            toolkit: Набор инструментов
            model: Модель для использования
            system_prompt: Системный промпт (опционально)
        """
        self.toolkit = toolkit
        self.model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError("Не найден OPENROUTER_API_KEY в переменных окружения")
        
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.conversation_history = []
    
    def _default_system_prompt(self) -> str:
        """Системный промпт по умолчанию"""
        tools_desc = self.toolkit.describe()
        
        return f"""Ты — умный ассистент с доступом к инструментам.

Доступные инструменты:
{tools_desc}

Правила:
1. Используй инструменты для точных вычислений и поиска информации
2. Не выдумывай данные — всегда проверяй через инструменты
3. Если инструмент вернул ошибку, попробуй другой подход
4. Отвечай на русском языке
"""
    
    def _call_api(self, messages: list, tools: list) -> dict:
        """Вызов OpenRouter API"""
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto"
            },
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def run(self, user_message: str, max_tool_calls: int = 5) -> str:
        """
        Обрабатывает запрос пользователя.
        
        Args:
            user_message: Сообщение пользователя
            max_tool_calls: Максимум вызовов инструментов
            
        Returns:
            Ответ агента
        """
        print(f"\n{'='*60}")
        print(f"👤 Пользователь: {user_message}")
        print('='*60)
        
        # Добавляем сообщение пользователя
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history,
            {"role": "user", "content": user_message}
        ]
        
        tool_calls_count = 0
        
        while tool_calls_count < max_tool_calls:
            # Запрос к API
            response = self._call_api(messages, self.toolkit.get_schemas())
            message = response["choices"][0]["message"]
            
            # Проверяем, есть ли вызовы инструментов
            if "tool_calls" not in message or not message["tool_calls"]:
                # Финальный ответ
                final_answer = message.get("content", "")
                
                # Сохраняем в историю
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": final_answer})
                
                print(f"\n🤖 Агент: {final_answer}")
                return final_answer
            
            # Обрабатываем вызовы инструментов
            tool_calls_count += 1
            messages.append(message)  # Добавляем сообщение с tool_calls
            
            for tool_call in message["tool_calls"]:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])
                
                print(f"\n🔧 Вызов: {function_name}({function_args})")
                
                # Выполняем инструмент
                try:
                    result = self.toolkit.execute(function_name, **function_args)
                    print(f"   → Результат: {result[:100]}...")
                except Exception as e:
                    result = f"Ошибка: {str(e)}"
                    print(f"   → ❌ {result}")
                
                # Добавляем результат
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result
                })
        
        return "Превышено максимальное количество вызовов инструментов"
    
    def clear_history(self):
        """Очищает историю разговора"""
        self.conversation_history = []
        print("🗑️ История очищена")


# ═══════════════════════════════════════════════════════════════
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ═══════════════════════════════════════════════════════════════

def main():
    """Демонстрация работы агента"""
    
    # Создаём набор инструментов
    toolkit = ToolKit()
    toolkit.register(calculator_tool)
    toolkit.register(wikipedia_tool)
    toolkit.register(datetime_tool)
    
    # Создаём агента
    agent = SimpleToolAgent(toolkit)
    
    # Тестовые запросы
    queries = [
        "Сколько будет 234 * 567?",
        "Кто такой Альберт Эйнштейн?",
        "Какой сегодня день недели?",
        "Вычисли площадь круга с радиусом 5",
    ]
    
    for query in queries:
        agent.run(query)
        print()


if __name__ == "__main__":
    main()
```

### Тестирование агента

```python
"""
Тесты для агента с инструментами.
"""

def test_calculator():
    """Тест калькулятора"""
    toolkit = ToolKit()
    toolkit.register(calculator_tool)
    
    # Базовые операции
    assert toolkit.execute("calculator", expression="2 + 2") == "4.0"
    assert toolkit.execute("calculator", expression="10 / 3") == "3.3333333333333335"
    assert toolkit.execute("calculator", expression="sqrt(16)") == "4.0"
    assert toolkit.execute("calculator", expression="pi * 2") == str(3.141592653589793 * 2)
    
    print("✅ Тесты калькулятора пройдены")


def test_wikipedia():
    """Тест поиска Wikipedia"""
    toolkit = ToolKit()
    toolkit.register(wikipedia_tool)
    
    result = toolkit.execute("wikipedia", query="Python")
    assert len(result) > 0
    assert "Ошибка" not in result
    
    print("✅ Тесты Wikipedia пройдены")


def test_datetime():
    """Тест даты/времени"""
    toolkit = ToolKit()
    toolkit.register(datetime_tool)
    
    result = toolkit.execute("current_datetime", timezone="Europe/Moscow")
    assert "года" in result
    assert ":" in result  # Время
    
    print("✅ Тесты даты/времени пройдены")


if __name__ == "__main__":
    test_calculator()
    test_wikipedia()
    test_datetime()
    print("\n✅ Все тесты пройдены!")
```

## Практические задания

### 🟢 Базовый уровень

**Задание 1: Генератор случайных чисел**

Создайте инструмент `random_number`, который генерирует случайное число в заданном диапазоне.

```python
# Шаблон
def random_number(min_value: int, max_value: int) -> int:
    """
    Генерирует случайное целое число в диапазоне [min_value, max_value].
    """
    # Ваш код здесь
    pass

random_tool = Tool(
    name="random_number",
    description="...",  # Заполните
    func=random_number,
    parameters={
        # Заполните
    }
)
```

**Задание 2: Тестирование**

Протестируйте агента с запросами:
- "Брось кубик" (1-6)
- "Выбери случайное число от 1 до 100"
- "Сгенерируй PIN-код из 4 цифр"

### 🟡 Средний уровень

**Задание 3: Конвертер валют**

Создайте инструмент для конвертации валют с использованием бесплатного API.

```python
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Конвертирует валюту.
    
    Args:
        amount: Сумма
        from_currency: Исходная валюта (USD, EUR, RUB)
        to_currency: Целевая валюта
    """
    # API: https://api.exchangerate-api.com/v4/latest/{from_currency}
    # Ваш код здесь
    pass
```

**Задание 4: Поиск файлов**

Создайте инструмент `file_search`, который ищет файлы по имени в заданной директории.

```python
def file_search(pattern: str, directory: str = ".") -> str:
    """
    Ищет файлы по паттерну в директории.
    """
    # Используйте glob или os.walk
    pass
```

### 🔴 Продвинутый уровень

**Задание 5: Выполнение Python-кода**

Создайте безопасный инструмент для выполнения Python-кода.

Требования:
- Ограничение времени выполнения (таймаут)
- Запрет опасных операций (import os, __import__, exec, eval)
- Ограничение на вывод (не более 1000 символов)

**Задание 6: Система разрешений**

Реализуйте систему разрешений для ToolKit:
- Разные пользователи имеют доступ к разным инструментам
- Логирование всех вызовов
- Ограничение частоты вызовов (rate limiting)

## Контрольные вопросы

1. **Что такое Function Calling?**
   <details>
   <summary>Ответ</summary>
   Function Calling — механизм API, позволяющий LLM вызывать внешние функции. Модель получает описания доступных функций и решает, когда и с какими аргументами их вызвать.
   </details>

2. **Какие компоненты включает Tool Schema?**
   <details>
   <summary>Ответ</summary>
   - name: имя функции
   - description: описание (когда использовать)
   - parameters: JSON Schema параметров
   - required: список обязательных параметров
   </details>

3. **Почему калькулятор на eval() небезопасен?**
   <details>
   <summary>Ответ</summary>
   eval() выполняет любой Python-код, что позволяет атакующему выполнить вредоносный код: `eval("__import__('os').system('rm -rf /')")`. Нужно использовать AST-парсинг с белым списком операций.
   </details>

4. **Как ограничить доступ к внешним API?**
   <details>
   <summary>Ответ</summary>
   1. Белый список разрешённых доменов
   2. Валидация URL перед запросом
   3. Таймауты на запросы
   4. Ограничение размера ответа
   5. Логирование всех запросов
   </details>

5. **Что делать, если инструмент вернул ошибку?**
   <details>
   <summary>Ответ</summary>
   1. Вернуть понятное сообщение об ошибке модели
   2. Модель может попробовать другой инструмент или подход
   3. Логировать ошибки для отладки
   4. Установить максимум повторных попыток
   </details>

## Заключение урока

### Что мы изучили

- **Концепция инструментов**: внешние функции для LLM
- **Function Calling**: механизм API для вызова функций
- **Реализация инструментов**: калькулятор, Wikipedia, время
- **Безопасность**: валидация входных данных, белые списки

### Созданные инструменты

| Инструмент | Назначение | Особенности |
|------------|------------|-------------|
| calculator | Вычисления | Безопасный AST-парсинг |
| wikipedia | Поиск информации | REST API Wikipedia |
| current_datetime | Дата/время | Поддержка часовых поясов |
| http_get | HTTP-запросы | Белый список доменов |

### Что дальше

В следующем уроке **"Агентный цикл ReAct"** мы:
- Реализуем паттерн ReAct (Reasoning + Acting)
- Создадим цикл THOUGHT → ACTION → OBSERVATION
- Построим Research Agent для многошаговых задач

**Готовы к ReAct?** Переходите к [Уроку 3: Агентный цикл ReAct](lesson_3_react_cycle.md)! 🔄

---

## Дополнительные материалы

### Документация:
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [OpenRouter API](https://openrouter.ai/docs)

### Библиотеки:
- [LangChain Tools](https://python.langchain.com/docs/modules/tools/)
- [Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page)

### Безопасность:
- [OWASP LLM Security](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

