# Урок 2: Реализация ядра системы

## Введение

В предыдущем уроке мы спроектировали архитектуру SchoolBot и создали базовую структуру проекта. Теперь пришло время оживить нашего помощника! В этом уроке мы реализуем ключевые компоненты: роутер моделей, менеджер контекста и систему промптов.

К концу урока SchoolBot сможет реально объяснять школьные темы и создавать конспекты, выбирая подходящую модель для каждой задачи.

## Цели урока

После завершения урока вы сможете:

- ✅ Реализовать роутер для интеллектуального выбора модели
- ✅ Создать менеджер контекста для длинных учебных сессий
- ✅ Интегрировать все компоненты в работающую систему
- ✅ Использовать промпты, адаптированные под школьную программу

## Ключевые термины

- **Роутер моделей** — компонент, выбирающий оптимальную модель для задачи
- **Контекст диалога** — история сообщений, сохраняемая между запросами
- **Стратегия управления контекстом** — алгоритм сжатия/обрезки истории
- **Класс-фасад** — единая точка входа, скрывающая сложность системы

## 1. Реализация роутера моделей

### Напоминание из Модуля 2

В модуле 2 мы изучали многомодельный роутинг. Теперь применим эти знания:

- Простые вопросы → быстрая дешёвая модель
- Сложные объяснения → мощная модель
- При ошибках → fallback на резервную модель

### Реализация ModelRouter (`router.py`)

```python
"""
Роутер моделей для SchoolBot.
Выбирает оптимальную модель в зависимости от типа задачи.
"""

import os
import time
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class ModelRouter:
    """
    Интеллектуальный роутер для выбора модели под задачу.
    Поддерживает fallback при недоступности основной модели.
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("❌ API ключ не найден! Проверьте файл .env")
        
        # Модели по категориям
        self.models = {
            "premium": [
                "anthropic/claude-3.5-sonnet",
                "openai/gpt-4-turbo",
            ],
            "balanced": [
                "meta-llama/llama-3-70b-instruct",
                "mistralai/mixtral-8x22b-instruct",
            ],
            "fast": [
                "meta-llama/llama-3-8b-instruct",
                "mistralai/mistral-7b-instruct",
            ]
        }
        
        # Маппинг типов задач на категории моделей
        self.task_model_mapping = {
            "explain_complex": "premium",      # Сложные объяснения
            "explain_simple": "balanced",      # Простые объяснения
            "generate_exam": "balanced",       # Генерация тестов
            "check_answer": "fast",            # Проверка ответов
            "summarize": "balanced",           # Конспекты
            "tutor": "balanced",               # Режим репетитора
            "quality_check": "fast",           # Проверка качества
        }
        
        # Статистика использования
        self.usage_stats = {
            "requests": 0,
            "tokens_used": 0,
            "errors": 0
        }
    
    def select_model(self, task_type: str, complexity: str = "medium") -> str:
        """
        Выбирает модель на основе типа задачи и сложности.
        
        Args:
            task_type: Тип задачи (explain_complex, generate_exam, и т.д.)
            complexity: Сложность (easy, medium, hard)
            
        Returns:
            Название модели
        """
        # Определяем категорию модели
        category = self.task_model_mapping.get(task_type, "balanced")
        
        # Корректируем на основе сложности
        if complexity == "hard" and category == "balanced":
            category = "premium"
        elif complexity == "easy" and category == "balanced":
            category = "fast"
        
        # Возвращаем первую модель из категории
        models_list = self.models.get(category, self.models["balanced"])
        return models_list[0]
    
    def send_request(self, 
                     messages: List[Dict], 
                     task_type: str = "explain_simple",
                     complexity: str = "medium",
                     max_tokens: int = 1500,
                     temperature: float = 0.7) -> Dict:
        """
        Отправляет запрос к модели с автоматическим выбором и fallback.
        
        Args:
            messages: Список сообщений диалога
            task_type: Тип задачи для выбора модели
            complexity: Сложность запроса
            max_tokens: Максимум токенов в ответе
            temperature: Температура генерации
            
        Returns:
            Dict с результатом или ошибкой
        """
        # Выбираем основную модель
        primary_model = self.select_model(task_type, complexity)
        
        # Определяем список fallback моделей
        category = self.task_model_mapping.get(task_type, "balanced")
        fallback_models = self.models.get(category, [])[1:] + self.models["fast"]
        
        # Пробуем основную модель и fallback
        all_models = [primary_model] + fallback_models
        
        for model in all_models[:3]:  # Максимум 3 попытки
            result = self._try_model(model, messages, max_tokens, temperature)
            
            if result["success"]:
                return result
            
            print(f"⚠️ Модель {model} недоступна, пробуем следующую...")
            time.sleep(1)  # Пауза между попытками
        
        self.usage_stats["errors"] += 1
        return {
            "success": False,
            "error": "Все модели недоступны. Попробуйте позже."
        }
    
    def _try_model(self, model: str, messages: List[Dict],
                   max_tokens: int, temperature: float) -> Dict:
        """Пробует отправить запрос к конкретной модели"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/schoolbot",
            "X-Title": "SchoolBot"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "usage": {"include": True}
        }
        
        try:
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Обновляем статистику
            self.usage_stats["requests"] += 1
            usage = result.get("usage", {})
            self.usage_stats["tokens_used"] += usage.get("total_tokens", 0)
            
            return {
                "success": True,
                "content": result["choices"][0]["message"]["content"],
                "model": model,
                "usage": usage
            }
            
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Таймаут запроса"}
        except requests.exceptions.HTTPError as e:
            return {"success": False, "error": f"HTTP ошибка: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_stats(self) -> Dict:
        """Возвращает статистику использования"""
        return self.usage_stats.copy()


# Тестирование роутера
if __name__ == "__main__":
    router = ModelRouter()
    
    print("🔄 Тестирование роутера моделей")
    print("="*50)
    
    # Тест выбора моделей
    test_cases = [
        ("explain_complex", "hard"),
        ("explain_simple", "easy"),
        ("check_answer", "medium"),
        ("generate_exam", "hard"),
    ]
    
    for task, complexity in test_cases:
        model = router.select_model(task, complexity)
        print(f"Задача: {task}, Сложность: {complexity} → Модель: {model}")
    
    print("\n" + "="*50)
    print("📤 Тестовый запрос к API")
    
    test_messages = [
        {"role": "system", "content": "Ты — помощник для школьников."},
        {"role": "user", "content": "Что такое квадратное уравнение? Ответь кратко."}
    ]
    
    result = router.send_request(test_messages, "explain_simple", "easy")
    
    if result["success"]:
        print(f"✅ Успешно! Модель: {result['model']}")
        print(f"📝 Ответ: {result['content'][:200]}...")
    else:
        print(f"❌ Ошибка: {result['error']}")
```

### 🔍 Проверьте себя

Прежде чем продолжить, ответьте:
- Какую модель выберет роутер для задачи `explain_complex` со сложностью `hard`?
- Что произойдёт, если основная модель вернёт ошибку?

<details>
<summary>Ответы</summary>

1. Для `explain_complex` + `hard` будет выбрана модель категории "premium" (например, Claude 3.5 Sonnet)
2. Роутер автоматически попробует fallback модели из той же категории, затем из категории "fast"
</details>

## 2. Реализация менеджера контекста

### Напоминание из Модуля 2

В уроке 4 модуля 2 мы изучали стратегии управления контекстом:
- **Скользящее окно** — сохраняем последние N сообщений
- **Резюмирование** — сжимаем старые сообщения в краткое описание
- **Гибридный подход** — комбинируем оба метода

### Реализация ContextManager (`context.py`)

```python
"""
Менеджер контекста для SchoolBot.
Управляет историей диалога и предотвращает переполнение контекста.
"""

from typing import Dict, List, Optional
from datetime import datetime


class ContextManager:
    """
    Управляет историей диалога с поддержкой разных стратегий.
    """
    
    def __init__(self, 
                 max_messages: int = 20,
                 summary_threshold: int = 15,
                 keep_recent: int = 5):
        """
        Args:
            max_messages: Максимум сообщений в контексте
            summary_threshold: Когда начинать резюмирование
            keep_recent: Сколько последних сообщений сохранять при резюме
        """
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        self.keep_recent = keep_recent
        
        # История сообщений
        self.messages: List[Dict] = []
        
        # Системный промпт (всегда в начале)
        self.system_prompt: Optional[str] = None
        
        # Резюме предыдущего контекста
        self.context_summary: Optional[str] = None
        
        # Метаданные сессии
        self.session_start = datetime.now()
        self.topics_discussed: List[str] = []
    
    def set_system_prompt(self, prompt: str):
        """Устанавливает системный промпт"""
        self.system_prompt = prompt
    
    def add_message(self, role: str, content: str, topic: str = None):
        """
        Добавляет сообщение в историю.
        
        Args:
            role: 'user' или 'assistant'
            content: Текст сообщения
            topic: Тема сообщения (для отслеживания)
        """
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        if topic and topic not in self.topics_discussed:
            self.topics_discussed.append(topic)
        
        # Проверяем, не пора ли управлять контекстом
        if len(self.messages) >= self.summary_threshold:
            self._manage_context()
    
    def get_messages_for_api(self) -> List[Dict]:
        """
        Возвращает сообщения в формате для API.
        
        Returns:
            Список сообщений с системным промптом и контекстом
        """
        result = []
        
        # 1. Системный промпт
        if self.system_prompt:
            result.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        # 2. Резюме предыдущего контекста (если есть)
        if self.context_summary:
            result.append({
                "role": "system",
                "content": f"Краткое содержание предыдущей части разговора:\n{self.context_summary}"
            })
        
        # 3. Актуальные сообщения
        for msg in self.messages[-self.max_messages:]:
            result.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return result
    
    def _manage_context(self):
        """Управляет размером контекста при достижении порога"""
        
        if len(self.messages) < self.summary_threshold:
            return
        
        # Сообщения для резюмирования (старые)
        old_messages = self.messages[:-self.keep_recent]
        
        # Создаём текстовое резюме
        summary_parts = []
        for msg in old_messages:
            role = "Ученик" if msg["role"] == "user" else "Помощник"
            # Берём первые 100 символов каждого сообщения
            content_preview = msg["content"][:100]
            if len(msg["content"]) > 100:
                content_preview += "..."
            summary_parts.append(f"- {role}: {content_preview}")
        
        # Обновляем резюме
        new_summary = "\n".join(summary_parts[-10:])  # Последние 10 пунктов резюме
        
        if self.context_summary:
            self.context_summary = f"{self.context_summary}\n\nПродолжение:\n{new_summary}"
        else:
            self.context_summary = new_summary
        
        # Оставляем только последние сообщения
        self.messages = self.messages[-self.keep_recent:]
        
        print(f"📝 Контекст оптимизирован. Сохранено {len(self.messages)} сообщений.")
    
    def get_topics(self) -> List[str]:
        """Возвращает список обсуждённых тем"""
        return self.topics_discussed.copy()
    
    def get_session_info(self) -> Dict:
        """Возвращает информацию о сессии"""
        return {
            "start_time": self.session_start.isoformat(),
            "messages_count": len(self.messages),
            "topics": self.topics_discussed,
            "has_summary": self.context_summary is not None
        }
    
    def clear(self):
        """Очищает историю, сохраняя системный промпт"""
        self.messages = []
        self.context_summary = None
        self.topics_discussed = []
        self.session_start = datetime.now()
        print("🗑️ История очищена")
    
    def export_session(self) -> Dict:
        """Экспортирует сессию для сохранения"""
        return {
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "context_summary": self.context_summary,
            "topics": self.topics_discussed,
            "session_start": self.session_start.isoformat()
        }
    
    def import_session(self, data: Dict):
        """Импортирует сохранённую сессию"""
        self.system_prompt = data.get("system_prompt")
        self.messages = data.get("messages", [])
        self.context_summary = data.get("context_summary")
        self.topics_discussed = data.get("topics", [])
        self.session_start = datetime.fromisoformat(
            data.get("session_start", datetime.now().isoformat())
        )
        print("📂 Сессия загружена")


# Тестирование менеджера контекста
if __name__ == "__main__":
    print("🧪 Тестирование ContextManager")
    print("="*50)
    
    # Создаём менеджер с маленькими лимитами для теста
    ctx = ContextManager(max_messages=10, summary_threshold=8, keep_recent=3)
    
    # Устанавливаем системный промпт
    ctx.set_system_prompt("Ты — помощник для школьников.")
    
    # Добавляем сообщения
    test_messages = [
        ("user", "Что такое производная?", "производная"),
        ("assistant", "Производная — это скорость изменения функции..."),
        ("user", "Приведи пример", "производная"),
        ("assistant", "Например, если y = x², то y' = 2x..."),
        ("user", "А что такое интеграл?", "интеграл"),
        ("assistant", "Интеграл — это операция, обратная дифференцированию..."),
        ("user", "Как они связаны?", "связь"),
        ("assistant", "Производная и интеграл связаны через теорему Ньютона-Лейбница..."),
        ("user", "Расскажи подробнее про теорему", "теорема"),
        ("assistant", "Теорема Ньютона-Лейбница гласит, что..."),
    ]
    
    for role, content, *topic in test_messages:
        topic = topic[0] if topic else None
        ctx.add_message(role, content, topic)
        print(f"Добавлено: {role[:4]}... | Всего сообщений: {len(ctx.messages)}")
    
    print("\n" + "="*50)
    print("📊 Информация о сессии:")
    print(ctx.get_session_info())
    
    print("\n📝 Сообщения для API:")
    api_messages = ctx.get_messages_for_api()
    for msg in api_messages:
        print(f"  [{msg['role']}]: {msg['content'][:50]}...")
```

## 3. Интеграция компонентов в SchoolAssistant

Теперь объединим всё в главном классе. Обновим файл `assistant.py`:

```python
"""
Главный класс SchoolAssistant — интегрированное ядро SchoolBot.
"""

import os
from typing import Dict, List, Optional
from config import (
    SUBJECTS, DIFFICULTY_LEVELS, DEFAULT_MODEL,
    CONTEXT_CONFIG, QUALITY_CONFIG
)
from prompts import PromptLibrary
from router import ModelRouter
from context import ContextManager


class SchoolAssistant:
    """
    Умный помощник для школьников старших классов.
    Интегрирует роутер моделей, менеджер контекста и промпты.
    """
    
    def __init__(self):
        """Инициализация помощника со всеми компонентами"""
        # Текущие настройки
        self.current_subject = "математика"
        self.current_grade = 11
        self.current_difficulty = "medium"
        self.current_exam_type = "ege"  # или "oge"
        
        # Инициализация компонентов
        print("🚀 Инициализация SchoolBot...")
        
        try:
            self.router = ModelRouter()
            print("   ✅ Роутер моделей")
        except Exception as e:
            print(f"   ❌ Ошибка роутера: {e}")
            self.router = None
        
        self.context = ContextManager(
            max_messages=CONTEXT_CONFIG["max_messages"],
            summary_threshold=CONTEXT_CONFIG["summary_threshold"],
            keep_recent=CONTEXT_CONFIG["keep_recent"]
        )
        print("   ✅ Менеджер контекста")
        
        self.prompts = PromptLibrary
        print("   ✅ Библиотека промптов")
        
        # Устанавливаем системный промпт
        self._update_system_prompt()
        
        print("\n🎓 SchoolBot готов к работе!")
        self._print_current_settings()
    
    def _update_system_prompt(self):
        """Обновляет системный промпт при изменении настроек"""
        system_prompt = self.prompts.get_system_prompt(
            subject=self.current_subject,
            grade=self.current_grade,
            difficulty=self.current_difficulty
        )
        self.context.set_system_prompt(system_prompt)
    
    def _print_current_settings(self):
        """Выводит текущие настройки"""
        print(f"   Предмет: {self.current_subject}")
        print(f"   Класс: {self.current_grade}")
        print(f"   Сложность: {self.current_difficulty}")
        print(f"   Экзамен: {self.current_exam_type.upper()}")
    
    # ==================== ОСНОВНЫЕ МЕТОДЫ ====================
    
    def explain_topic(self, topic: str) -> str:
        """
        Объясняет тему простым языком с примерами.
        
        Args:
            topic: Тема для объяснения
            
        Returns:
            Понятное объяснение темы
        """
        if not self.router:
            return "❌ Ошибка: роутер моделей не инициализирован"
        
        # Формируем промпт
        user_prompt = self.prompts.get_explain_prompt(
            topic=topic,
            grade=self.current_grade,
            difficulty=self.current_difficulty
        )
        
        # Добавляем в контекст
        self.context.add_message("user", f"Объясни тему: {topic}", topic)
        
        # Получаем сообщения для API
        messages = self.context.get_messages_for_api()
        messages.append({"role": "user", "content": user_prompt})
        
        # Определяем сложность задачи для роутера
        task_complexity = "hard" if self.current_difficulty == "hard" else "medium"
        task_type = "explain_complex" if task_complexity == "hard" else "explain_simple"
        
        # Отправляем запрос
        print(f"🔄 Генерирую объяснение темы '{topic}'...")
        result = self.router.send_request(
            messages=messages,
            task_type=task_type,
            complexity=task_complexity,
            max_tokens=2000,
            temperature=0.7
        )
        
        if result["success"]:
            explanation = result["content"]
            # Сохраняем ответ в контекст
            self.context.add_message("assistant", explanation)
            return explanation
        else:
            return f"❌ Ошибка при генерации: {result['error']}"
    
    def create_summary(self, topic: str) -> str:
        """
        Создаёт краткий конспект по теме.
        
        Args:
            topic: Тема для конспекта
            
        Returns:
            Структурированный конспект
        """
        if not self.router:
            return "❌ Ошибка: роутер моделей не инициализирован"
        
        # Формируем промпт
        user_prompt = self.prompts.get_summary_prompt(
            topic=topic,
            exam_type=self.current_exam_type
        )
        
        # Добавляем в контекст
        self.context.add_message("user", f"Создай конспект: {topic}", topic)
        
        # Получаем сообщения
        messages = self.context.get_messages_for_api()
        messages.append({"role": "user", "content": user_prompt})
        
        # Отправляем запрос
        print(f"📋 Создаю конспект по теме '{topic}'...")
        result = self.router.send_request(
            messages=messages,
            task_type="summarize",
            complexity=self.current_difficulty,
            max_tokens=1500,
            temperature=0.5
        )
        
        if result["success"]:
            summary = result["content"]
            self.context.add_message("assistant", summary)
            return summary
        else:
            return f"❌ Ошибка при создании конспекта: {result['error']}"
    
    def tutor_message(self, user_message: str, topic: str = None) -> str:
        """
        Обрабатывает сообщение в режиме репетитора.
        
        Args:
            user_message: Сообщение ученика
            topic: Текущая тема (опционально)
            
        Returns:
            Ответ репетитора
        """
        if not self.router:
            return "❌ Ошибка: роутер моделей не инициализирован"
        
        # Добавляем сообщение пользователя в контекст
        self.context.add_message("user", user_message, topic)
        
        # Формируем промпт для репетитора
        tutor_prompt = self.prompts.get_tutor_prompt(
            subject=self.current_subject,
            topic=topic or "общая тема",
            grade=self.current_grade,
            user_message=user_message
        )
        
        # Получаем контекст с историей
        messages = self.context.get_messages_for_api()
        
        # Добавляем инструкцию репетитора в последнее системное сообщение
        # (не как отдельное сообщение, чтобы не загромождать)
        
        # Отправляем запрос
        result = self.router.send_request(
            messages=messages,
            task_type="tutor",
            complexity=self.current_difficulty,
            max_tokens=1000,
            temperature=0.7
        )
        
        if result["success"]:
            response = result["content"]
            self.context.add_message("assistant", response)
            return response
        else:
            return f"❌ Ошибка: {result['error']}"
    
    def generate_exam(self, topic: str, exam_type: str = None, 
                      num_questions: int = 3) -> str:
        """
        Генерирует тестовые задания в формате ОГЭ/ЕГЭ.
        (Полная реализация в Уроке 3)
        """
        if not self.router:
            return "❌ Ошибка: роутер моделей не инициализирован"
        
        exam = exam_type or self.current_exam_type
        
        # Формируем промпт
        user_prompt = self.prompts.get_exam_prompt(
            topic=topic,
            num_questions=num_questions,
            exam_type=exam,
            grade=self.current_grade,
            difficulty=self.current_difficulty
        )
        
        messages = self.context.get_messages_for_api()
        messages.append({"role": "user", "content": user_prompt})
        
        print(f"📝 Генерирую {num_questions} заданий {exam.upper()} по теме '{topic}'...")
        result = self.router.send_request(
            messages=messages,
            task_type="generate_exam",
            complexity=self.current_difficulty,
            max_tokens=2500,
            temperature=0.8
        )
        
        if result["success"]:
            return result["content"]
        else:
            return f"❌ Ошибка: {result['error']}"
    
    def check_answer(self, question: str, student_answer: str) -> str:
        """
        Проверяет ответ ученика.
        (Полная реализация в Уроке 3)
        """
        # Заглушка — будет реализовано в Уроке 3
        return f"[Проверка ответа будет реализована в Уроке 3]"
    
    # ==================== НАСТРОЙКИ ====================
    
    def set_subject(self, subject: str) -> bool:
        """Устанавливает текущий предмет"""
        subject_lower = subject.lower()
        
        for subj_name, subj_info in SUBJECTS.items():
            if subject_lower == subj_name or subject_lower in subj_info["aliases"]:
                self.current_subject = subj_name
                self._update_system_prompt()
                print(f"✅ Предмет: {subj_name} — {subj_info['description']}")
                return True
        
        print(f"❌ Предмет не найден: {subject}")
        print(f"   Доступные: {', '.join(SUBJECTS.keys())}")
        return False
    
    def set_grade(self, grade: int) -> bool:
        """Устанавливает класс ученика"""
        if grade in [9, 10, 11]:
            self.current_grade = grade
            # Автоматически подбираем тип экзамена
            self.current_exam_type = "oge" if grade == 9 else "ege"
            self._update_system_prompt()
            print(f"✅ Класс: {grade} ({self.current_exam_type.upper()})")
            return True
        
        print(f"❌ Некорректный класс: {grade}. Допустимо: 9, 10, 11")
        return False
    
    def set_difficulty(self, difficulty: str) -> bool:
        """Устанавливает уровень сложности"""
        if difficulty in DIFFICULTY_LEVELS:
            self.current_difficulty = difficulty
            self._update_system_prompt()
            info = DIFFICULTY_LEVELS[difficulty]
            print(f"✅ Сложность: {difficulty} — {info['description']}")
            return True
        
        print(f"❌ Некорректный уровень: {difficulty}")
        print(f"   Допустимо: {', '.join(DIFFICULTY_LEVELS.keys())}")
        return False
    
    def get_settings(self) -> Dict:
        """Возвращает текущие настройки"""
        return {
            "subject": self.current_subject,
            "grade": self.current_grade,
            "difficulty": self.current_difficulty,
            "exam_type": self.current_exam_type
        }
    
    # ==================== СЕССИИ ====================
    
    def get_history(self) -> List[Dict]:
        """Возвращает историю сессии"""
        return self.context.messages.copy()
    
    def get_topics(self) -> List[str]:
        """Возвращает список обсуждённых тем"""
        return self.context.get_topics()
    
    def clear_history(self):
        """Очищает историю"""
        self.context.clear()
    
    def get_stats(self) -> Dict:
        """Возвращает статистику использования"""
        router_stats = self.router.get_stats() if self.router else {}
        session_info = self.context.get_session_info()
        
        return {
            "router": router_stats,
            "session": session_info
        }


# Тестирование интегрированного класса
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 ТЕСТИРОВАНИЕ SCHOOLASSISTANT")
    print("="*60)
    
    # Создаём помощника
    bot = SchoolAssistant()
    
    print("\n" + "-"*40)
    print("Тест 1: Изменение настроек")
    print("-"*40)
    bot.set_subject("физика")
    bot.set_grade(10)
    bot.set_difficulty("medium")
    
    print("\n" + "-"*40)
    print("Тест 2: Объяснение темы")
    print("-"*40)
    explanation = bot.explain_topic("закон Ома")
    print(f"\n📚 Объяснение:\n{explanation[:500]}...")
    
    print("\n" + "-"*40)
    print("Тест 3: Создание конспекта")
    print("-"*40)
    summary = bot.create_summary("законы Ньютона")
    print(f"\n📋 Конспект:\n{summary[:500]}...")
    
    print("\n" + "-"*40)
    print("Тест 4: Режим репетитора")
    print("-"*40)
    response = bot.tutor_message("Я не понимаю, почему тела падают с одинаковой скоростью в вакууме?")
    print(f"\n👨‍🏫 Ответ репетитора:\n{response[:500]}...")
    
    print("\n" + "-"*40)
    print("Тест 5: Статистика")
    print("-"*40)
    stats = bot.get_stats()
    print(f"📊 Статистика: {stats}")
```

## 4. Обновление CLI

Теперь обновим `main.py`, чтобы использовать новый функционал:

```python
"""
Точка входа SchoolBot — обновлённый CLI с реальным функционалом.
"""

import sys
from assistant import SchoolAssistant


def print_help():
    """Выводит справку по командам"""
    help_text = """
╔════════════════════════════════════════════════════════════════╗
║                    🎓 SchoolBot — Справка                       ║
╠════════════════════════════════════════════════════════════════╣
║  ОБУЧЕНИЕ:                                                      ║
║  /explain [тема]     — объяснить тему простым языком           ║
║  /exam [тема] [n]    — создать n заданий ОГЭ/ЕГЭ (по умолч. 3) ║
║  /summarize [тема]   — создать краткий конспект                ║
║  [любой текст]       — режим репетитора (диалог)               ║
║                                                                 ║
║  НАСТРОЙКИ:                                                     ║
║  /subject [предмет]  — математика, физика, информатика...      ║
║  /grade [9/10/11]    — класс (влияет на ОГЭ/ЕГЭ)               ║
║  /level [easy/medium/hard] — сложность объяснений              ║
║  /settings           — показать текущие настройки              ║
║                                                                 ║
║  СЕССИЯ:                                                        ║
║  /history            — обсуждённые темы                        ║
║  /stats              — статистика использования                ║
║  /clear              — очистить историю                        ║
║                                                                 ║
║  /help — справка  |  /exit — выход                             ║
╚════════════════════════════════════════════════════════════════╝
"""
    print(help_text)


def print_welcome():
    """Приветственное сообщение"""
    welcome = """
╔════════════════════════════════════════════════════════════════╗
║                                                                 ║
║     🎓 Добро пожаловать в SchoolBot!                           ║
║                                                                 ║
║     Умный помощник для подготовки к ОГЭ и ЕГЭ                  ║
║                                                                 ║
║     • Объясню любую тему простым языком                        ║
║     • Создам тестовые задания для практики                     ║
║     • Помогу разобраться в сложных местах                      ║
║                                                                 ║
║     Введите /help для списка команд                            ║
║                                                                 ║
╚════════════════════════════════════════════════════════════════╝
"""
    print(welcome)


def parse_command(user_input: str) -> tuple:
    """Разбирает команду пользователя"""
    parts = user_input.strip().split(maxsplit=1)
    command = parts[0].lower() if parts else ""
    args = parts[1] if len(parts) > 1 else ""
    return command, args


def main():
    """Главный цикл программы"""
    print_welcome()
    
    # Создаём помощника
    try:
        assistant = SchoolAssistant()
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        print("Проверьте файл .env с API ключом")
        return
    
    print("\n💡 Совет: просто напишите вопрос для режима репетитора")
    print("   или используйте команды /explain, /exam, /summarize\n")
    
    # Главный цикл
    while True:
        try:
            user_input = input("\n📝 Вы: ").strip()
            
            if not user_input:
                continue
            
            command, args = parse_command(user_input)
            
            # === ВЫХОД ===
            if command in ["/exit", "/quit", "/q"]:
                stats = assistant.get_stats()
                print(f"\n📊 За сессию: {stats['router'].get('requests', 0)} запросов")
                print("👋 До встречи! Удачи на экзаменах!")
                break
            
            # === СПРАВКА ===
            elif command == "/help":
                print_help()
            
            # === НАСТРОЙКИ ===
            elif command == "/settings":
                settings = assistant.get_settings()
                print(f"\n⚙️ Текущие настройки:")
                print(f"   📚 Предмет: {settings['subject']}")
                print(f"   🎓 Класс: {settings['grade']}")
                print(f"   📊 Сложность: {settings['difficulty']}")
                print(f"   📝 Экзамен: {settings['exam_type'].upper()}")
            
            elif command == "/subject":
                if args:
                    assistant.set_subject(args)
                else:
                    from config import SUBJECTS
                    print("📚 Доступные предметы:")
                    for name, info in SUBJECTS.items():
                        print(f"   • {name} — {info['description']}")
            
            elif command == "/grade":
                if args:
                    try:
                        assistant.set_grade(int(args))
                    except ValueError:
                        print("❌ Укажите номер класса: /grade 11")
                else:
                    print("❌ Укажите класс: /grade 9, /grade 10 или /grade 11")
            
            elif command == "/level":
                if args:
                    assistant.set_difficulty(args)
                else:
                    from config import DIFFICULTY_LEVELS
                    print("📊 Уровни сложности:")
                    for level, info in DIFFICULTY_LEVELS.items():
                        print(f"   • {level} — {info['description']}")
            
            # === ОБУЧЕНИЕ ===
            elif command == "/explain":
                if args:
                    result = assistant.explain_topic(args)
                    print(f"\n📚 ОБЪЯСНЕНИЕ:\n{'─'*40}\n{result}\n{'─'*40}")
                else:
                    print("❌ Укажите тему: /explain квадратные уравнения")
            
            elif command == "/exam":
                if args:
                    # Парсим: /exam тема [количество]
                    parts = args.rsplit(maxsplit=1)
                    topic = parts[0]
                    num = 3
                    if len(parts) > 1 and parts[1].isdigit():
                        num = int(parts[1])
                        topic = parts[0]
                    
                    result = assistant.generate_exam(topic, num_questions=num)
                    print(f"\n📝 ЗАДАНИЯ {assistant.current_exam_type.upper()}:\n{'─'*40}\n{result}\n{'─'*40}")
                else:
                    print("❌ Укажите тему: /exam производные 5")
            
            elif command == "/summarize":
                if args:
                    result = assistant.create_summary(args)
                    print(f"\n📋 КОНСПЕКТ:\n{'─'*40}\n{result}\n{'─'*40}")
                else:
                    print("❌ Укажите тему: /summarize законы термодинамики")
            
            # === СЕССИЯ ===
            elif command == "/history":
                topics = assistant.get_topics()
                if topics:
                    print(f"\n📜 Обсуждённые темы: {', '.join(topics)}")
                else:
                    print("📜 Темы ещё не обсуждались")
            
            elif command == "/stats":
                stats = assistant.get_stats()
                print(f"\n📊 Статистика:")
                print(f"   Запросов: {stats['router'].get('requests', 0)}")
                print(f"   Токенов: {stats['router'].get('tokens_used', 0)}")
                print(f"   Сообщений в сессии: {stats['session'].get('messages_count', 0)}")
            
            elif command == "/clear":
                assistant.clear_history()
            
            # === РЕЖИМ РЕПЕТИТОРА ===
            else:
                if user_input.startswith("/"):
                    print(f"❌ Неизвестная команда: {command}")
                    print("   Введите /help для списка команд")
                else:
                    # Обрабатываем как вопрос репетитору
                    print("\n🤔 Думаю...")
                    result = assistant.tutor_message(user_input)
                    print(f"\n👨‍🏫 Репетитор:\n{'─'*40}\n{result}\n{'─'*40}")
        
        except KeyboardInterrupt:
            print("\n\n👋 До встречи!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
```

## Практические задания

### 🟢 Базовый уровень

**Задание 1: Тестирование роутера**
Запустите `python router.py` и убедитесь, что:
- Роутер выбирает разные модели для разных задач
- Тестовый запрос успешно выполняется

**Задание 2: Тестирование контекста**
Запустите `python context.py` и проверьте:
- Сообщения добавляются корректно
- При достижении порога срабатывает оптимизация

### 🟡 Средний уровень

**Задание 3: Улучшение промптов**
Создайте специализированные промпты для одного предмета. Например, для математики добавьте:
- Явное требование пошагового решения
- Примеры с числами
- Визуализацию (описание графиков словами)

**Задание 4: Добавление статистики по предметам**
Расширьте `ContextManager`, чтобы он отслеживал:
- Сколько вопросов задано по каждому предмету
- Какие темы вызвали больше всего вопросов

### 🔴 Продвинутый уровень

**Задание 5: Умный роутинг по содержанию**
Добавьте в `ModelRouter` анализ текста запроса:
- Если в вопросе есть формулы/числа → математическая модель
- Если нужен творческий ответ → модель для генерации
- Используйте ключевые слова для определения

**Задание 6: Резюмирование через LLM**
Измените `ContextManager._manage_context()` так, чтобы резюме создавалось с помощью LLM, а не простым обрезанием текста.

## Контрольные вопросы

1. **Как роутер выбирает модель для задачи?**
   <details>
   <summary>Ответ</summary>
   Роутер использует маппинг task_model_mapping, который связывает типы задач с категориями моделей. Затем учитывается сложность: для hard задач используется premium категория, для easy — fast. Первая модель из выбранной категории становится основной.
   </details>

2. **Что происходит при переполнении контекста?**
   <details>
   <summary>Ответ</summary>
   При достижении summary_threshold срабатывает _manage_context(): старые сообщения преобразуются в текстовое резюме, которое добавляется как системное сообщение, а в истории остаются только keep_recent последних сообщений.
   </details>

3. **Зачем нужен системный промпт и когда он обновляется?**
   <details>
   <summary>Ответ</summary>
   Системный промпт задаёт роль и контекст для модели (предмет, класс, сложность). Он обновляется при изменении любой из этих настроек через метод _update_system_prompt(), чтобы модель адаптировала свои ответы.
   </details>

4. **Как работает fallback в роутере?**
   <details>
   <summary>Ответ</summary>
   Если основная модель возвращает ошибку, роутер пробует следующие модели из той же категории, затем модели из категории "fast". Максимум 3 попытки с паузой 1 секунду между ними.
   </details>

5. **Почему мы используем разную температуру для разных задач?**
   <details>
   <summary>Ответ</summary>
   Температура влияет на креативность ответов. Для объяснений (0.7) нужна некоторая вариативность. Для конспектов (0.5) важнее точность. Для генерации заданий (0.8) нужно разнообразие. Для проверки ответов нужна минимальная вариативность.
   </details>

## Заключение урока

### Что мы изучили

В этом уроке мы создали работающее ядро SchoolBot:

- **ModelRouter** — интеллектуальный выбор модели с fallback
- **ContextManager** — управление историей диалога с оптимизацией
- **Интеграция** — объединение компонентов в SchoolAssistant
- **Обновлённый CLI** — полноценный интерфейс с реальным функционалом

### Связь с Модулем 2

Мы применили знания из предыдущего модуля:
- Урок 2: Многомодельный роутинг → наш ModelRouter
- Урок 4: Управление контекстом → наш ContextManager
- Урок 3: Концепция LLM-as-a-judge → добавим в Уроке 3

### Что нас ждёт дальше

В следующем уроке **"Добавление интеллектуальных функций"** мы:
- Интегрируем LLM-as-a-judge для проверки качества объяснений
- Создадим генератор тестовых заданий в формате ОГЭ/ЕГЭ
- Добавим адаптивную сложность и обработку ошибок

### Ваш прогресс

🚀 **Отличная работа!** SchoolBot теперь может:
- ✅ Объяснять темы из школьной программы
- ✅ Создавать конспекты для повторения
- ✅ Вести диалог в режиме репетитора
- ✅ Выбирать оптимальную модель для задачи
- ✅ Управлять контекстом длинных сессий

**Готовы добавить интеллект?** Переходите к [Уроку 3: Интеллектуальные функции](lesson_3_smart_features.md)!

---

## Дополнительные материалы

### Паттерны проектирования:
- [Facade Pattern](https://refactoring.guru/design-patterns/facade)
- [Strategy Pattern](https://refactoring.guru/design-patterns/strategy)

### Работа с API:
- [Requests Library](https://docs.python-requests.org/)
- [OpenRouter Documentation](https://openrouter.ai/docs)

### Управление состоянием:
- [State Management Patterns](https://martinfowler.com/eaaDev/EventSourcing.html)

