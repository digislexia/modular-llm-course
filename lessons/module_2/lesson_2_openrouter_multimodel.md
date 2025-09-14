# Урок 2: Многомодельность и роутинг в OpenRouter

## Введение

В мире LLM нет универсальной модели, которая была бы лучшей во всём. GPT-4 отлично справляется с рассуждениями, Claude — с анализом документов, Mixtral — быстр и экономичен для простых задач. **OpenRouter** открывает доступ к десяткам моделей через единый API, а вы научитесь выбирать подходящую модель для каждой задачи.

В этом уроке вы освоите стратегическое мышление разработчика: когда использовать дорогую, но мощную модель, а когда достаточно быстрой и дешёвой. Вы создадите систему автоматического выбора моделей и научитесь строить надёжные приложения с резервными сценариями.

## Цели урока

После завершения урока вы сможете:

- ✅ Сравнивать различные модели через OpenRouter API
- ✅ Анализировать модели по качеству, скорости и стоимости  
- ✅ Использовать AutoRouter для автоматического выбора модели
- ✅ Реализовывать fallback стратегии для надёжности
- ✅ Создавать системы интеллектуального роутинга запросов

## Ключевые термины

- **Роутинг моделей** — автоматический выбор подходящей модели на основе характеристик задачи
- **AutoRouter** — функция OpenRouter для автоматического выбора оптимальной модели
- **Fallback** — резервный сценарий при недоступности основной модели
- **Многомодельная архитектура** — система, использующая разные модели для разных типов задач
- **Стоимость за токен** — цена использования модели, измеряемая в долларах за 1000 токенов

## 1. Обзор доступных моделей в OpenRouter

### Категории моделей

OpenRouter предоставляет доступ к моделям различных классов:

**Топовые модели (премиум)**:
- `openai/gpt-4-turbo` — лучшие рассуждения, высокая стоимость
- `anthropic/claude-3.5-sonnet` — отличный анализ текста, большой контекст
- `google/gemini-pro-1.5` — мультимодальность, большое контекстное окно

**Сбалансированные модели**:
- `microsoft/wizardlm-2-8x22b` — хорошее качество, средняя цена  
- `meta-llama/llama-3-70b-instruct` — открытая модель, быстрая
- `mistralai/mixtral-8x22b-instruct` — специализация на коде

**Быстрые и экономичные**:
- `microsoft/wizardlm-2-7b` — базовые задачи, низкая стоимость
- `google/gemma-7b-it` — простые запросы, высокая скорость

### Получение списка доступных моделей

```python
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def get_available_models():
    """Получает список всех доступных моделей в OpenRouter"""
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    url = "https://openrouter.ai/api/v1/models"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        models_data = response.json()
        return models_data["data"]
        
    except Exception as e:
        print(f"Ошибка получения списка моделей: {e}")
        return []

def analyze_models():
    """Анализирует характеристики доступных моделей"""
    
    models = get_available_models()
    
    if not models:
        print("❌ Не удалось получить список моделей")
        return
    
    print("📊 АНАЛИЗ ДОСТУПНЫХ МОДЕЛЕЙ")
    print("=" * 60)
    
    # Сортируем по стоимости
    sorted_models = sorted(models, key=lambda x: x.get("pricing", {}).get("prompt", 0))
    
    print("🏷️ Топ-10 самых дешёвых моделей:")
    for i, model in enumerate(sorted_models[:10], 1):
        name = model["id"]
        context_length = model.get("context_length", "N/A")
        prompt_price = model.get("pricing", {}).get("prompt", 0)
        
        print(f"{i:2d}. {name}")
        print(f"    Контекст: {context_length:,} токенов")
        print(f"    Цена: ${prompt_price:.6f} за 1K токенов")
        print()

if __name__ == "__main__":
    analyze_models()
```

### 🔍 Проверьте себя

Запустите анализ моделей и ответьте:
- Какие модели самые дешёвые для экспериментов?
- У каких моделей самое большое контекстное окно?
- Как соотносятся цена и качество у разных провайдеров?

## 2. Сравнение моделей на практических задачах

### Класс для сравнения моделей

```python
import time
from datetime import datetime
import json

class ModelComparison:
    """Сравнивает работу разных моделей на одинаковых задачах"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("API ключ не найден!")
    
    def send_request_to_model(self, prompt, model_name, max_tokens=1000, temperature=0.7):
        """Отправляет запрос к конкретной модели"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-username/llm-course",
            "X-Title": "Model Comparison Tool"
        }
        
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "usage": {"include": True}
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(self.url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            end_time = time.time()
            result = response.json()
            
            return {
                "success": True,
                "model": model_name,
                "response": result["choices"][0]["message"]["content"],
                "usage": result.get("usage", {}),
                "response_time": end_time - start_time,
                "finish_reason": result["choices"][0].get("finish_reason")
            }
            
        except Exception as e:
            return {
                "success": False,
                "model": model_name,
                "error": str(e),
                "response_time": time.time() - start_time
            }
    
    def compare_models_on_task(self, prompt, models_list, task_name=""):
        """Сравнивает список моделей на одной задаче"""
        
        print(f"🔄 Сравниваем модели на задаче: {task_name}")
        print("=" * 50)
        print(f"📝 Промпт: {prompt[:100]}..." if len(prompt) > 100 else f"📝 Промпт: {prompt}")
        print()
        
        results = []
        
        for i, model in enumerate(models_list, 1):
            print(f"[{i}/{len(models_list)}] Тестируем {model}...")
            
            result = self.send_request_to_model(prompt, model)
            results.append(result)
            
            if result["success"]:
                print(f"✅ Успешно за {result['response_time']:.2f}с")
            else:
                print(f"❌ Ошибка: {result['error']}")
            
            time.sleep(1)  # Пауза между запросами
        
        return results
    
    def analyze_comparison_results(self, results, task_name=""):
        """Анализирует результаты сравнения"""
        
        print(f"\n📊 РЕЗУЛЬТАТЫ СРАВНЕНИЯ: {task_name}")
        print("=" * 60)
        
        successful_results = [r for r in results if r["success"]]
        
        if not successful_results:
            print("❌ Все запросы завершились ошибкой")
            return
        
        # Анализ по различным критериям
        print("\n🚀 Скорость ответа:")
        speed_sorted = sorted(successful_results, key=lambda x: x["response_time"])
        for i, result in enumerate(speed_sorted[:3], 1):
            print(f"{i}. {result['model']}: {result['response_time']:.2f}с")
        
        print("\n💰 Эффективность по токенам:")
        for result in successful_results:
            usage = result["usage"]
            if usage:
                total_tokens = usage.get("total_tokens", 0)
                print(f"   {result['model']}: {total_tokens} токенов")
        
        print("\n📝 Качество ответов (длина):")
        length_sorted = sorted(successful_results, key=lambda x: len(x["response"]), reverse=True)
        for i, result in enumerate(length_sorted[:3], 1):
            response_length = len(result["response"])
            print(f"{i}. {result['model']}: {response_length} символов")
        
        print("\n📄 Ответы моделей:")
        print("-" * 40)
        
        for result in successful_results:
            print(f"\n🤖 {result['model']}:")
            print(f"Время: {result['response_time']:.2f}с")
            if result["usage"]:
                print(f"Токены: {result['usage'].get('total_tokens', 'N/A')}")
            print(f"Ответ: {result['response'][:200]}...")
            print("-" * 40)
    
    def save_comparison_results(self, results, task_name, filename=None):
        """Сохраняет результаты сравнения в файл"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_comparison_{timestamp}.json"
        
        comparison_data = {
            "timestamp": datetime.now().isoformat(),
            "task_name": task_name,
            "results": results
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(comparison_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Результаты сохранены в {filename}")

# Демонстрационные задачи для сравнения
def demo_model_comparison():
    """Демонстрирует сравнение моделей на разных типах задач"""
    
    comparator = ModelComparison()
    
    # Модели для сравнения (выберите доступные в вашем OpenRouter аккаунте)
    test_models = [
        "microsoft/wizardlm-2-8x22b",
        "meta-llama/llama-3-70b-instruct", 
        "mistralai/mixtral-8x22b-instruct"
    ]
    
    # Задача 1: Креативное письмо
    creative_task = """
Напиши короткий рассказ (150-200 слов) на тему "Робот, который научился мечтать".
Используй яркие образы и неожиданный поворот сюжета.
"""
    
    results_creative = comparator.compare_models_on_task(
        creative_task, 
        test_models, 
        "Креативное письмо"
    )
    
    comparator.analyze_comparison_results(results_creative, "Креативное письмо")
    
    input("\nНажмите Enter для следующего теста...")
    
    # Задача 2: Аналитическое мышление  
    analytical_task = """
Проанализируй плюсы и минусы удалённой работы для IT-компаний.
Структурируй ответ по пунктам, приведи конкретные примеры.
Объём: до 300 слов.
"""
    
    results_analytical = comparator.compare_models_on_task(
        analytical_task,
        test_models,
        "Аналитическое мышление"
    )
    
    comparator.analyze_comparison_results(results_analytical, "Аналитическое мышление")
    
    # Сохранение результатов
    save_choice = input("\n💾 Сохранить результаты? (y/n): ").lower()
    if save_choice in ['y', 'yes', 'д', 'да']:
        comparator.save_comparison_results(results_creative, "Креативное письмо")
        comparator.save_comparison_results(results_analytical, "Аналитическое мышление")

if __name__ == "__main__":
    demo_model_comparison()
```

### 🔍 Проверьте себя

После запуска сравнения проанализируйте:
- Какая модель быстрее отвечает на креативные задачи?
- Где важнее скорость, а где качество?
- Как соотносятся стоимость и качество ответов?

## 3. AutoRouter: автоматический выбор модели

### Что такое AutoRouter

**AutoRouter** — функция OpenRouter, которая автоматически выбирает оптимальную модель на основе:
- Характера запроса
- Требований к качеству
- Бюджетных ограничений
- Скорости ответа

### Использование AutoRouter

```python
def demonstrate_autorouter():
    """Демонстрирует работу AutoRouter"""
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Используем специальную модель для AutoRouter
    autorouter_model = "openrouter/auto"
    
    test_prompts = [
        {
            "name": "Простой вопрос", 
            "prompt": "Столица Франции?",
            "expected_model_type": "быстрая и дешёвая"
        },
        {
            "name": "Сложный анализ",
            "prompt": """
            Проведи глубокий анализ влияния искусственного интеллекта 
            на рынок труда в ближайшие 10 лет. Рассмотри разные сценарии
            развития событий и их последствия для различных отраслей.
            """,
            "expected_model_type": "мощная и дорогая"
        },
        {
            "name": "Творческая задача",
            "prompt": """
            Напиши стихотворение в стиле Есенина о современных технологиях.
            Используй его характерные образы и ритм.
            """,
            "expected_model_type": "специализированная на творчестве"
        }
    ]
    
    print("🤖 ДЕМОНСТРАЦИЯ AUTOROUTER")
    print("=" * 50)
    
    for i, test_case in enumerate(test_prompts, 1):
        print(f"\n📝 Тест {i}: {test_case['name']}")
        print(f"Ожидаемый тип модели: {test_case['expected_model_type']}")
        print("-" * 30)
        
        data = {
            "model": autorouter_model,
            "messages": [{"role": "user", "content": test_case["prompt"]}],
            "max_tokens": 1000,
            "temperature": 0.7,
            # Дополнительные параметры для AutoRouter
            "usage": {"include": True},
            "route": "fallback"  # Включаем fallback в случае ошибки
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            
            # Получаем информацию о выбранной модели
            chosen_model = result.get("model", "Неизвестно")
            usage_info = result.get("usage", {})
            response_text = result["choices"][0]["message"]["content"]
            
            print(f"✅ AutoRouter выбрал: {chosen_model}")
            print(f"💰 Использовано токенов: {usage_info.get('total_tokens', 'N/A')}")
            print(f"📄 Ответ: {response_text[:150]}...")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        
        input("\nНажмите Enter для следующего теста...")

if __name__ == "__main__":
    demonstrate_autorouter()
```

## 4. Fallback стратегии для надёжности

### Зачем нужен Fallback

Проблемы, которые решает fallback:
- Модель временно недоступна
- Превышен лимит запросов  
- Таймаут соединения
- Ошибка API

### Реализация системы Fallback

```python
import random
import time
from typing import List, Dict, Optional

class ReliableModelRouter:
    """Надёжная система роутинга с fallback стратегиями"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Определяем приоритеты моделей по категориям
        self.model_categories = {
            "premium": [
                "openai/gpt-4-turbo",
                "anthropic/claude-3.5-sonnet",
                "google/gemini-pro-1.5"
            ],
            "balanced": [
                "microsoft/wizardlm-2-8x22b",
                "meta-llama/llama-3-70b-instruct",
                "mistralai/mixtral-8x22b-instruct"
            ],
            "budget": [
                "microsoft/wizardlm-2-7b",
                "google/gemma-7b-it",
                "meta-llama/llama-3-8b-instruct"
            ]
        }
        
    def send_request_with_fallback(self, prompt: str, 
                                   preferred_models: List[str],
                                   max_retries: int = 3,
                                   timeout: int = 30) -> Dict:
        """
        Отправляет запрос с fallback через список моделей
        
        Args:
            prompt: Текст запроса
            preferred_models: Список моделей в порядке приоритета
            max_retries: Максимальное количество попыток
            timeout: Таймаут для каждого запроса
        """
        
        last_error = None
        
        for attempt, model in enumerate(preferred_models, 1):
            print(f"🔄 Попытка {attempt}: используем модель {model}")
            
            try:
                result = self._send_single_request(prompt, model, timeout)
                
                if result["success"]:
                    print(f"✅ Успешно получен ответ от {model}")
                    return result
                else:
                    last_error = result["error"]
                    print(f"❌ Ошибка с моделью {model}: {last_error}")
                    
            except Exception as e:
                last_error = str(e)
                print(f"❌ Исключение с моделью {model}: {e}")
            
            # Пауза между попытками (exponential backoff)
            if attempt < len(preferred_models):
                wait_time = 2 ** (attempt - 1)  # 1, 2, 4, 8 секунд
                print(f"⏱️ Ожидание {wait_time}с перед следующей попыткой...")
                time.sleep(wait_time)
        
        return {
            "success": False,
            "error": f"Все модели недоступны. Последняя ошибка: {last_error}",
            "attempted_models": preferred_models
        }
    
    def _send_single_request(self, prompt: str, model: str, timeout: int) -> Dict:
        """Отправляет запрос к одной модели"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.7,
            "usage": {"include": True}
        }
        
        start_time = time.time()
        
        response = requests.post(self.url, headers=headers, json=data, timeout=timeout)
        response.raise_for_status()
        
        end_time = time.time()
        result = response.json()
        
        return {
            "success": True,
            "model": model,
            "response": result["choices"][0]["message"]["content"],
            "usage": result.get("usage", {}),
            "response_time": end_time - start_time
        }
    
    def smart_model_selection(self, prompt: str, task_type: str = "general") -> List[str]:
        """
        Интеллектуально выбирает модели на основе типа задачи
        
        Args:
            prompt: Текст запроса
            task_type: Тип задачи ("creative", "analytical", "coding", "general")
        """
        
        # Анализируем сложность запроса
        prompt_complexity = self._analyze_prompt_complexity(prompt)
        
        if task_type == "creative" or "стих" in prompt.lower() or "рассказ" in prompt.lower():
            # Для творческих задач предпочитаем модели с хорошей генерацией
            return self.model_categories["premium"] + self.model_categories["balanced"]
            
        elif task_type == "analytical" or len(prompt) > 500:
            # Для сложного анализа используем мощные модели
            return self.model_categories["premium"] + self.model_categories["balanced"]
            
        elif task_type == "coding" or "код" in prompt.lower() or "программ" in prompt.lower():
            # Для программирования подходят специализированные модели
            return [
                "mistralai/mixtral-8x22b-instruct",  # Хорош в коде
                "microsoft/wizardlm-2-8x22b",
                "meta-llama/llama-3-70b-instruct"
            ]
            
        elif prompt_complexity == "simple":
            # Для простых вопросов используем быстрые модели
            return self.model_categories["budget"] + self.model_categories["balanced"]
            
        else:
            # Общий случай: сбалансированные, затем премиум
            return self.model_categories["balanced"] + self.model_categories["premium"]
    
    def _analyze_prompt_complexity(self, prompt: str) -> str:
        """Анализирует сложность запроса"""
        
        # Простые критерии сложности
        simple_indicators = ["что такое", "столица", "когда", "где", "кто"]
        complex_indicators = ["анализ", "сравни", "разработай", "стратегия"]
        
        prompt_lower = prompt.lower()
        
        if any(indicator in prompt_lower for indicator in simple_indicators):
            return "simple"
        elif any(indicator in prompt_lower for indicator in complex_indicators):
            return "complex"
        elif len(prompt) > 300:
            return "complex"
        else:
            return "medium"

def demo_fallback_system():
    """Демонстрирует работу fallback системы"""
    
    router = ReliableModelRouter()
    
    test_cases = [
        {
            "prompt": "Что такое квантовое программирование?",
            "task_type": "general"
        },
        {
            "prompt": """
            Напиши функцию на Python для сортировки списка словарей 
            по нескольким ключам с возможностью указания порядка сортировки.
            """,
            "task_type": "coding"
        },
        {
            "prompt": """
            Создай стратегический план развития стартапа в области EdTech.
            Учти конкурентную среду, целевую аудиторию и финансовые показатели.
            """,
            "task_type": "analytical"
        }
    ]
    
    print("🛡️ ДЕМОНСТРАЦИЯ FALLBACK СИСТЕМЫ")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Тест {i}: {test_case['task_type']} задача")
        print(f"Запрос: {test_case['prompt'][:100]}...")
        
        # Получаем рекомендованные модели
        recommended_models = router.smart_model_selection(
            test_case["prompt"], 
            test_case["task_type"]
        )
        
        print(f"🎯 Рекомендованные модели: {recommended_models[:3]}")
        
        # Отправляем запрос с fallback
        result = router.send_request_with_fallback(
            test_case["prompt"],
            recommended_models[:3]  # Используем топ-3 модели
        )
        
        if result["success"]:
            print(f"✅ Получен ответ от модели: {result['model']}")
            print(f"📄 Ответ: {result['response'][:200]}...")
            
            usage = result["usage"]
            if usage:
                print(f"💰 Использовано токенов: {usage.get('total_tokens', 'N/A')}")
        else:
            print(f"❌ Все попытки неудачны: {result['error']}")
        
        input("\nНажмите Enter для следующего теста...")

if __name__ == "__main__":
    demo_fallback_system()
```

### 🔍 Проверьте себя

Протестируйте fallback систему:
1. Попробуйте с недоступными моделями в списке
2. Сымитируйте медленное соединение
3. Проверьте, как система выбирает модели для разных типов задач

## 5. Создание мини-фреймворка выбора модели

Объединим всё в интеллектуальную систему роутинга:

```python
class IntelligentModelRouter:
    """Интеллектуальная система выбора моделей"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Определяем профили использования
        self.usage_profiles = {
            "speed_first": {
                "description": "Приоритет скорости",
                "models": [
                    "google/gemma-7b-it",
                    "microsoft/wizardlm-2-7b", 
                    "meta-llama/llama-3-8b-instruct"
                ],
                "max_tokens": 500,
                "temperature": 0.3
            },
            "quality_first": {
                "description": "Приоритет качества",
                "models": [
                    "openai/gpt-4-turbo",
                    "anthropic/claude-3.5-sonnet",
                    "google/gemini-pro-1.5"
                ],
                "max_tokens": 2000,
                "temperature": 0.7
            },
            "balanced": {
                "description": "Баланс скорости и качества",
                "models": [
                    "microsoft/wizardlm-2-8x22b",
                    "meta-llama/llama-3-70b-instruct",
                    "mistralai/mixtral-8x22b-instruct"
                ],
                "max_tokens": 1000,
                "temperature": 0.5
            },
            "cost_effective": {
                "description": "Минимальная стоимость",
                "models": [
                    "google/gemma-7b-it",
                    "microsoft/wizardlm-2-7b",
                    "meta-llama/llama-3-8b-instruct"
                ],
                "max_tokens": 300,
                "temperature": 0.3
            }
        }
        
        # Специализированные настройки по типам задач
        self.task_specific_configs = {
            "creative_writing": {
                "profile": "quality_first",
                "temperature_boost": 0.2,
                "preferred_models": ["openai/gpt-4-turbo", "anthropic/claude-3.5-sonnet"]
            },
            "code_generation": {
                "profile": "balanced", 
                "temperature_boost": -0.2,
                "preferred_models": ["mistralai/mixtral-8x22b-instruct"]
            },
            "data_analysis": {
                "profile": "quality_first",
                "temperature_boost": -0.1,
                "preferred_models": ["anthropic/claude-3.5-sonnet", "openai/gpt-4-turbo"]
            },
            "simple_qa": {
                "profile": "speed_first",
                "temperature_boost": -0.2,
                "preferred_models": ["google/gemma-7b-it"]
            },
            "translation": {
                "profile": "balanced",
                "temperature_boost": -0.3,
                "preferred_models": ["google/gemini-pro-1.5"]
            }
        }
    
    def detect_task_type(self, prompt: str) -> str:
        """Автоматически определяет тип задачи по промпту"""
        
        prompt_lower = prompt.lower()
        
        # Паттерны для разных типов задач
        patterns = {
            "creative_writing": [
                "напиши рассказ", "сочини стихотворение", "создай историю",
                "придумай сюжет", "напиши эссе", "креативно"
            ],
            "code_generation": [
                "напиши код", "функцию", "программу", "скрипт", 
                "алгоритм", "python", "javascript", "класс"
            ],
            "data_analysis": [
                "проанализируй", "сравни", "исследуй", "оцени",
                "статистика", "тренды", "выводы", "рекомендации"
            ],
            "simple_qa": [
                "что такое", "как называется", "столица", "когда",
                "где находится", "кто", "сколько", "простой вопрос"
            ],
            "translation": [
                "переведи", "перевод", "translate", "на английский",
                "на русский", "с языка", "значение слова"
            ]
        }
        
        # Подсчитываем совпадения для каждого типа
        scores = {}
        for task_type, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            if score > 0:
                scores[task_type] = score
        
        # Возвращаем тип с максимальным скором
        if scores:
            return max(scores, key=scores.get)
        else:
            return "simple_qa"  # По умолчанию
    
    def get_optimal_config(self, prompt: str, user_preference: str = "auto") -> Dict:
        """Получает оптимальную конфигурацию для запроса"""
        
        if user_preference == "auto":
            task_type = self.detect_task_type(prompt)
            config = self.task_specific_configs.get(task_type, {})
        else:
            config = {"profile": user_preference}
        
        profile_name = config.get("profile", "balanced")
        profile = self.usage_profiles[profile_name].copy()
        
        # Применяем модификации температуры
        temperature_boost = config.get("temperature_boost", 0)
        profile["temperature"] = max(0.0, min(1.0, profile["temperature"] + temperature_boost))
        
        # Используем предпочтительные модели, если указаны
        if "preferred_models" in config:
            profile["models"] = config["preferred_models"] + profile["models"]
        
        return {
            "detected_task": task_type if user_preference == "auto" else user_preference,
            "profile_name": profile_name,
            "config": profile
        }
    
    def send_smart_request(self, prompt: str, preference: str = "auto") -> Dict:
        """Отправляет умный запрос с автоматическим выбором модели"""
        
        # Получаем оптимальную конфигурацию
        optimal_config = self.get_optimal_config(prompt, preference)
        config = optimal_config["config"]
        
        print(f"🎯 Обнаружен тип задачи: {optimal_config['detected_task']}")
        print(f"📊 Используемый профиль: {optimal_config['profile_name']}")
        print(f"🤖 Рекомендованные модели: {config['models'][:3]}")
        
        # Создаём fallback роутер для отправки
        router = ReliableModelRouter()
        
        # Настраиваем параметры запроса
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        for model in config["models"][:3]:  # Попробуем топ-3 модели
            try:
                data = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": config["max_tokens"],
                    "temperature": config["temperature"],
                    "usage": {"include": True}
                }
                
                print(f"🔄 Пробуем модель: {model}")
                
                response = requests.post(self.url, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                
                return {
                    "success": True,
                    "model": model,
                    "task_type": optimal_config["detected_task"],
                    "profile": optimal_config["profile_name"],
                    "response": result["choices"][0]["message"]["content"],
                    "usage": result.get("usage", {}),
                    "config_used": config
                }
                
            except Exception as e:
                print(f"❌ Ошибка с моделью {model}: {e}")
                continue
        
        return {
            "success": False,
            "error": "Все рекомендованные модели недоступны"
        }

def interactive_smart_router():
    """Интерактивная демонстрация умного роутера"""
    
    router = IntelligentModelRouter()
    
    print("🧠 ИНТЕЛЛЕКТУАЛЬНЫЙ РОУТЕР МОДЕЛЕЙ")
    print("=" * 50)
    print("Введите 'exit' для выхода")
    print("Доступные режимы: auto, speed_first, quality_first, balanced, cost_effective")
    print()
    
    while True:
        prompt = input("📝 Введите ваш запрос: ").strip()
        
        if prompt.lower() in ['exit', 'выход']:
            print("👋 До встречи!")
            break
        
        if not prompt:
            continue
        
        preference = input("⚙️ Режим работы (Enter=auto): ").strip() or "auto"
        
        print("\n" + "="*50)
        result = router.send_smart_request(prompt, preference)
        
        if result["success"]:
            print(f"\n✅ Успешно обработано моделью: {result['model']}")
            print(f"🎯 Тип задачи: {result['task_type']}")
            print(f"📊 Профиль: {result['profile']}")
            
            usage = result["usage"]
            if usage:
                print(f"💰 Использовано токенов: {usage.get('total_tokens', 'N/A')}")
            
            print(f"\n📄 ОТВЕТ:")
            print("-" * 30)
            print(result["response"])
            print("-" * 30)
            
        else:
            print(f"❌ Ошибка: {result['error']}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    interactive_smart_router()
```

## Практические задания

### 🟢 Базовый уровень

**Задание 1: Сравнение моделей**
Выберите 3 модели и сравните их на задачах:
- Перевод текста с русского на английский
- Решение простой математической задачи
- Краткое изложение новости

**Задание 2: Простой роутер**
Создайте функцию, которая выбирает модель на основе длины промпта:
- Короткие запросы (< 50 слов) → быстрая модель
- Длинные запросы (> 200 слов) → мощная модель

### 🟡 Средний уровень

**Задание 3: Fallback с логированием**
Расширьте fallback систему возможностью:
- Логирования всех попыток и ошибок
- Статистики успешности каждой модели
- Автоматической корректировки приоритетов

**Задание 4: Анализ стоимости**
Создайте инструмент для:
- Подсчёта стоимости запросов к разным моделям
- Сравнения цена/качество
- Рекомендаций по оптимизации расходов

**Задание 5: Кастомные профили**
Добавьте возможность создавать пользовательские профили:
- Сохранение в файл конфигурации
- Загрузка и применение профилей
- Интерфейс редактирования профилей

### 🔴 Продвинутый уровень

**Задание 6: Адаптивная система**
Создайте систему, которая:
- Учится на результатах предыдущих запросов
- Корректирует выбор моделей на основе обратной связи
- Адаптируется к изменениям в доступности моделей

**Задание 7: Мониторинг и аналитика**
Разработайте систему мониторинга:
- Дашборд использования моделей
- Анализ трендов и паттернов
- Предсказание нагрузки и стоимости

**Задание 8: Распределённая система**
Создайте систему для команды:
- Общий пул моделей и лимитов
- Балансировка нагрузки между пользователями
- Приоритизация важных запросов

## Контрольные вопросы

1. **Зачем может понадобиться fallback система?**
   <details>
   <summary>Ответ</summary>
   Fallback обеспечивает надёжность приложения при временной недоступности модели, превышении лимитов API, сбоях сети или ошибках сервера. Это позволяет пользователям получить ответ даже при проблемах с основной моделью.
   </details>

2. **В чём преимущество AutoRouter?**
   <details>
   <summary>Ответ</summary>
   AutoRouter автоматически выбирает оптимальную модель на основе характеристик запроса, что экономит время разработчика и обеспечивает оптимальное соотношение качества, скорости и стоимости без необходимости ручной настройки.
   </details>

3. **Как определить, какая модель лучше подходит под задачу?**
   <details>
   <summary>Ответ</summary>
   Нужно учитывать: тип задачи (творчество, анализ, код), требования к качеству, ограничения по времени и бюджету, размер контекста, специализацию модели. Лучший способ — тестирование на реальных примерах.
   </details>

4. **Что такое exponential backoff и зачем он нужен?**
   <details>
   <summary>Ответ</summary>
   Exponential backoff — это стратегия увеличения интервала между повторными попытками (1с, 2с, 4с, 8с...). Это предотвращает перегрузку сервера множественными запросами и повышает вероятность успешного подключения.
   </details>

5. **Как оптимизировать стоимость использования множественных моделей?**
   <details>
   <summary>Ответ</summary>
   Использовать дешёвые модели для простых задач, кэшировать частые запросы, группировать похожие задачи, мониторить использование токенов, настроить автоматическое переключение на более дешёвые модели при превышении бюджета.
   </details>

## Заключение урока

### Что мы изучили

В этом уроке вы освоили стратегическое использование множественных моделей:

- **Анализ и сравнение**: научились оценивать модели по скорости, качеству и стоимости
- **AutoRouter**: изучили автоматический выбор оптимальной модели
- **Fallback стратегии**: создали надёжные системы с резервными сценариями
- **Интеллектуальный роутинг**: построили фреймворк для умного выбора модели под задачу

### Связь с предыдущим уроком

Продвинутые техники промпт-инжиниринга из урока 1 теперь дополнены умным выбором модели:
- **Сложные промпты** требуют мощных моделей — теперь вы умеете их выбирать
- **ReAct и ToT** работают лучше с моделями, специализированными на рассуждениях
- **Композиция техник** может потребовать разных моделей для разных этапов

### Что нас ждёт дальше

В финальном уроке модуля **"Контроль качества и оптимизация"** мы изучим:
- LLM-as-a-judge для автоматической оценки качества
- Создание пайплайнов валидации ответов
- Оптимизацию токенов и снижение стоимости
- Мониторинг и улучшение AI-систем в продакшене

### Ваши достижения

🚀 **Превосходный результат!** Вы стали архитектором AI-систем:
- ✅ Умеете сравнивать и выбирать модели для разных задач
- ✅ Создаёте надёжные системы с автоматическим fallback
- ✅ Используете AutoRouter для оптимизации ресурсов
- ✅ Проектируете интеллектуальные системы роутинга

**Готовы изучать контроль качества AI-систем?** Переходите к [финальному уроку](lesson_3_quality_optimization.md)!

---

## Дополнительные материалы

### Документация OpenRouter:
- [OpenRouter Models](https://openrouter.ai/models) — полный список доступных моделей
- [AutoRouter Guide](https://openrouter.ai/docs/quick-start#auto-routing) — руководство по автоматическому роутингу
- [Pricing Calculator](https://openrouter.ai/models) — калькулятор стоимости запросов

### Инструменты мониторинга:
- [OpenRouter Dashboard](https://openrouter.ai/activity) — статистика использования API
- [Model Benchmarks](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) — сравнения моделей

### Продвинутые концепции:
- [Load Balancing Strategies](https://openrouter.ai/docs/limits) — стратегии балансировки нагрузки
- [Cost Optimization Patterns](https://openrouter.ai/docs/costs) — паттерны оптимизации стоимости
