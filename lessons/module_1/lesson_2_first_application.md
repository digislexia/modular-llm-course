# Урок 2: Подключение к модели и первое приложение

## Введение

Теперь, когда вы понимаете принципы работы LLM, пришло время перейти к практике! В этом уроке мы превратим теоретические знания в работающие приложения. Вы настроите окружение разработчика, подключитесь к реальной языковой модели через API и создадите ваше первое приложение с использованием ИИ.

К концу урока у вас будет работающий чат-бот, который может поддерживать диалог и запоминать контекст разговора.

## Цели урока

После завершения урока вы сможете:

- ✅ Настроить рабочее окружение для разработки с LLM API
- ✅ Выполнить первый запрос к языковой модели и получить ответ
- ✅ Создать полноценное приложение (чат-бот или суммаризатор)
- ✅ Понимать структуру запросов и ответов API
- ✅ Работать с токенами и учитывать ограничения модели

## Ключевые термины

- **API (Application Programming Interface)** — набор правил и протоколов для взаимодействия между программами
- **Промпт (prompt)** — текст, который мы отправляем модели как входные данные
- **Completion** — текст, который генерирует модель в ответ на промпт
- **Токен** — единица текста для модели (примерно ¾ слова для русского языка)
- **Температура (temperature)** — параметр, контролирующий "креативность" модели (0.0-1.0)

## 1. Подготовка окружения разработчика

### Установка Python и необходимых библиотек

Убедитесь, что у вас установлен Python 3.8 или новее:

```bash
python --version
```

Создайте изолированное окружение для проекта:

```bash
# Создание виртуального окружения
python -m venv llm_course_env

# Активация (Windows)
llm_course_env\Scripts\activate

# Активация (macOS/Linux)  
source llm_course_env/bin/activate
```

Установите необходимые зависимости:

```bash
pip install requests python-dotenv
```

Создайте файл `requirements.txt` для управления зависимостями:

```text
requests==2.31.0
python-dotenv==1.0.0
```

### Выбор API провайдера: OpenRouter

Для этого курса мы используем **OpenRouter** — агрегатор различных языковых моделей с единым API.

**Преимущества OpenRouter:**
- Единый интерфейс для доступа к разным моделям
- Конкурентные цены и прозрачная тарификация  
- Совместимость с OpenAI API (легко переключаться)
- Не требует ожидания доступа

### Получение API ключа

1. Перейдите на сайт [openrouter.ai](https://openrouter.ai/)
2. Нажмите "Sign Up" и создайте аккаунт
3. Перейдите в раздел "API Keys" в личном кабинете
4. Создайте новый ключ с описанием "LLM Course"
5. **Важно**: скопируйте ключ сразу — он показывается только один раз!

### 🔍 Проверьте себя

Прежде чем продолжить, убедитесь, что выполнили все шаги:
- [ ] Python 3.8+ установлен
- [ ] Виртуальное окружение создано и активировано  
- [ ] Библиотеки установлены
- [ ] API ключ от OpenRouter получен

Если что-то не работает — не переходите дальше, решите проблему сначала.

## 2. Создание первого приложения

### Структура проекта

Создайте следующую структуру папок:

```
my_first_llm_app/
├── .env                 # Переменные окружения (НЕ коммитить!)
├── .gitignore          # Исключения для Git
├── requirements.txt    # Зависимости проекта
├── main.py            # Основной файл программы
└── utils/             # Вспомогательные модули
    └── __init__.py
```

### Настройка переменных окружения

Создайте файл `.env` в корне проекта:

```env
# API ключ OpenRouter
OPENROUTER_API_KEY=your_api_key_here

# Настройки по умолчанию
DEFAULT_MODEL=microsoft/wizardlm-2-8x22b
MAX_TOKENS=1000
TEMPERATURE=0.7
```

**🚨 Безопасность**: файл `.env` содержит секретную информацию!

Создайте `.gitignore`:

```gitignore
# Переменные окружения
.env

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
```

### Первый запрос к модели

Создайте файл `main.py`:

```python
import os
import requests
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

def send_request_to_llm(prompt, model=None, max_tokens=None, temperature=None):
    """
    Отправляет запрос к языковой модели через OpenRouter API
    
    Args:
        prompt (str): Текст запроса к модели
        model (str): Название модели (по умолчанию из .env)
        max_tokens (int): Максимальное количество токенов в ответе
        temperature (float): Температура генерации (0.0-1.0)
    
    Returns:
        dict: Словарь с ответом модели и метаинформацией
    """
    
    # Получаем настройки из переменных окружения
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError(
            "API ключ не найден! Проверьте файл .env и переменную OPENROUTER_API_KEY"
        )
    
    # Параметры запроса с значениями по умолчанию
    model = model or os.getenv("DEFAULT_MODEL", "microsoft/wizardlm-2-8x22b")
    max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "1000"))
    temperature = temperature or float(os.getenv("TEMPERATURE", "0.7"))
    
    # URL и заголовки для API
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-username/llm-course",  # Опционально
        "X-Title": "LLM Course App"  # Опционально
    }
    
    # Тело запроса в формате OpenAI API
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Включаем информацию об использовании токенов
        "usage": {
            "include": True
        }
    }
    
    try:
        print(f"🔄 Отправляем запрос к модели {model}...")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        # Проверяем статус ответа
        response.raise_for_status()
        
        result = response.json()
        
        # Извлекаем данные из ответа
        message_content = result["choices"][0]["message"]["content"]
        usage_info = result.get("usage", {})
        
        return {
            "response": message_content,
            "model": model,
            "usage": usage_info,
            "success": True
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "response": None,
            "error": f"Ошибка сети: {e}",
            "success": False
        }
    except KeyError as e:
        return {
            "response": None,
            "error": f"Неожиданный формат ответа API: отсутствует ключ {e}",
            "success": False
        }
    except Exception as e:
        return {
            "response": None,
            "error": f"Неизвестная ошибка: {e}",
            "success": False
        }

def main():
    """Основная функция для тестирования API"""
    
    print("🚀 Тестируем подключение к LLM API")
    print("=" * 50)
    
    # Тестовый промпт
    test_prompt = """Объясни простыми словами и с примером, что такое машинное обучение. 
    Ответ должен быть понятен человеку без технического образования."""
    
    print(f"📝 Промпт: {test_prompt}")
    print("-" * 50)
    
    # Отправляем запрос
    result = send_request_to_llm(test_prompt)
    
    if result["success"]:
        print("✅ Запрос выполнен успешно!")
        print(f"🤖 Ответ модели ({result['model']}):")
        print(result["response"])
        
        # Показываем информацию об использовании
        usage = result["usage"]
        if usage:
            print("\n📊 Статистика использования:")
            print(f"   Токенов в запросе: {usage.get('prompt_tokens', 'N/A')}")
            print(f"   Токенов в ответе: {usage.get('completion_tokens', 'N/A')}")
            print(f"   Всего токенов: {usage.get('total_tokens', 'N/A')}")
    else:
        print("❌ Ошибка при выполнении запроса:")
        print(f"   {result['error']}")

if __name__ == "__main__":
    main()
```

### Первый запуск

Убедитесь, что ваш API ключ правильно добавлен в `.env`, затем запустите:

```bash
python main.py
```

**🎉 Поздравляем!** Если вы видите ответ от модели — вы только что выполнили свой первый запрос к LLM!

### 🔍 Проверьте себя

Проанализируйте полученный ответ:
- Соответствует ли он вашему промпту?
- Какой стиль использует модель?  
- Сколько токенов потратилось на запрос и ответ?
- Что произойдёт, если изменить температуру на 0.1 или 0.9?

Попробуйте изменить параметры и сравните результаты.

## 3. Анализ структуры API запросов и ответов

### Структура запроса

```json
{
  "model": "microsoft/wizardlm-2-8x22b",
  "messages": [
    {
      "role": "user", 
      "content": "Ваш промпт здесь"
    }
  ],
  "max_tokens": 1000,
  "temperature": 0.7
}
```

**Ключевые параметры:**

- **`model`**: выбранная языковая модель
- **`messages`**: массив сообщений (поддерживает историю диалога)
- **`max_tokens`**: максимальное количество токенов в ответе  
- **`temperature`**: контролирует случайность (0.0 = детерминированно, 1.0 = очень креативно)

### Структура ответа

```json
{
  "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
  "object": "chat.completion",
  "created": 1677664795,
  "model": "microsoft/wizardlm-2-8x22b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Ответ модели здесь..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 120,
    "total_tokens": 135
  }
}
```

**Важные поля:**

- **`choices[0].message.content`**: сгенерированный текст
- **`usage`**: информация о потреблённых токенах (для подсчёта стоимости)
- **`finish_reason`**: причина окончания генерации ("stop", "length", "content_filter")

### Понимание токенов

**Что такое токен?**

Для русского языка токенизация работает примерно так:

```python
# Примеры токенизации
"Привет" → ["Прив", "ет"] = 2 токена
"программирование" → ["программ", "иров", "ание"] = 3 токена  
"!" → ["!"] = 1 токен
"123" → ["123"] = 1 токен
```

**Практические соотношения для русского:**
- 1 токен ≈ 0.75 слова
- 100 слов ≈ 133 токена
- 1000 символов ≈ 200-250 токенов

**Почему это важно:**
- API взимает плату за токены
- Модели имеют лимит токенов
- Нужно планировать размер промптов

## 4. Создание полноценного чат-бота

Теперь создадим чат-бота, который может поддерживать диалог:

```python
import os
import json
from datetime import datetime
import requests
from dotenv import load_dotenv

load_dotenv()

class LLMChatbot:
    """Класс для создания чат-бота на основе LLM API"""
    
    def __init__(self, model=None, system_prompt=None):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("DEFAULT_MODEL", "microsoft/wizardlm-2-8x22b")
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.conversation_history = []
        self.total_tokens_used = 0
        
        if not self.api_key:
            raise ValueError("API ключ не найден в переменных окружения!")
        
        # Устанавливаем системный промпт, если предоставлен
        if system_prompt:
            self.set_system_prompt(system_prompt)
    
    def set_system_prompt(self, prompt):
        """Устанавливает системный промпт для определения поведения бота"""
        system_message = {
            "role": "system",
            "content": prompt
        }
        
        # Если уже есть системный промпт, заменяем его
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history[0] = system_message
        else:
            self.conversation_history.insert(0, system_message)
    
    def send_message(self, user_message):
        """
        Отправляет сообщение боту и получает ответ
        
        Args:
            user_message (str): Сообщение пользователя
            
        Returns:
            dict: Результат с ответом бота и метаинформацией
        """
        
        # Добавляем сообщение пользователя в историю
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": self.conversation_history,
            "max_tokens": int(os.getenv("MAX_TOKENS", "1000")),
            "temperature": float(os.getenv("TEMPERATURE", "0.7")),
            # Запрашиваем информацию об использовании токенов
            "usage": {
                "include": True
            }
        }
        
        try:
            response = requests.post(self.url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            bot_response = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            
            # Добавляем ответ бота в историю
            self.conversation_history.append({
                "role": "assistant", 
                "content": bot_response
            })
            
            # Обновляем счётчик токенов
            self.total_tokens_used += usage.get("total_tokens", 0)
            
            return {
                "response": bot_response,
                "usage": usage,
                "success": True,
                "total_tokens_session": self.total_tokens_used
            }
            
        except Exception as e:
            return {
                "response": None,
                "error": str(e),
                "success": False
            }
    
    def reset_conversation(self):
        """Очищает историю диалога (кроме системного промпта)"""
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            # Сохраняем только системный промпт
            self.conversation_history = [self.conversation_history[0]]
        else:
            self.conversation_history = []
        
        self.total_tokens_used = 0
    
    def get_conversation_summary(self):
        """Возвращает краткую сводку о текущем диалоге"""
        messages_count = len([m for m in self.conversation_history if m["role"] in ["user", "assistant"]])
        system_prompt_set = bool(self.conversation_history and self.conversation_history[0]["role"] == "system")
        
        return {
            "messages_count": messages_count,
            "has_system_prompt": system_prompt_set,
            "total_tokens_used": self.total_tokens_used,
            "model": self.model
        }
    
    def save_conversation(self, filename):
        """Сохраняет историю диалога в JSON файл"""
        conversation_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "total_tokens": self.total_tokens_used,
            "messages": self.conversation_history
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)

def create_interactive_chat():
    """Создаёт интерактивный чат-интерфейс"""
    
    print("🤖 Добро пожаловать в интерактивный LLM чат!")
    print("Введите 'exit', 'quit' или 'выход' для завершения")
    print("Введите 'reset' для сброса истории диалога")
    print("Введите 'stats' для просмотра статистики")
    print("=" * 60)
    
    # Создаём бота с системным промптом
    system_prompt = """
    Ты дружелюбный и умный помощник по имени Алекс. 
    Отвечай на русском языке, будь полезным и терпеливым.
    Если не знаешь точного ответа, честно об этом скажи.
    Используй эмодзи, чтобы сделать общение более живым.
    """
    
    chatbot = LLMChatbot(system_prompt=system_prompt.strip())
    
    print("🎯 Бот настроен и готов к общению!")
    
    while True:
        try:
            user_input = input("\n🧑 Вы: ").strip()
            
            if not user_input:
                continue
            
            # Команды управления
            if user_input.lower() in ['exit', 'quit', 'выход']:
                # Предложить сохранить диалог
                save = input("💾 Сохранить диалог? (y/n): ").lower()
                if save in ['y', 'yes', 'д', 'да']:
                    filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    chatbot.save_conversation(filename)
                    print(f"✅ Диалог сохранён в файл {filename}")
                
                print("👋 До свидания!")
                break
            
            elif user_input.lower() == 'reset':
                chatbot.reset_conversation()
                print("🔄 История диалога очищена")
                continue
                
            elif user_input.lower() == 'stats':
                stats = chatbot.get_conversation_summary()
                print(f"📊 Статистика сессии:")
                print(f"   Сообщений в диалоге: {stats['messages_count']}")
                print(f"   Использовано токенов: {stats['total_tokens_used']}")
                print(f"   Модель: {stats['model']}")
                print(f"   Системный промпт: {'✅' if stats['has_system_prompt'] else '❌'}")
                continue
            
            # Отправляем сообщение боту
            print("🤖 Алекс: ", end="", flush=True)
            result = chatbot.send_message(user_input)
            
            if result["success"]:
                print(result["response"])
                
                # Показываем токены (опционально)
                usage = result["usage"]
                if usage and usage.get("total_tokens"):
                    print(f"   💡 Использовано токенов: {usage['total_tokens']} "
                          f"(всего за сессию: {result['total_tokens_session']})")
            else:
                print(f"❌ Ошибка: {result['error']}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Чат завершён пользователем")
            break
        except Exception as e:
            print(f"\n❌ Неожиданная ошибка: {e}")

# Демонстрационная функция для тестирования
def demo_chatbot_capabilities():
    """Демонстрирует возможности чат-бота"""
    
    print("🧪 ДЕМОНСТРАЦИЯ ВОЗМОЖНОСТЕЙ ЧАТ-БОТА")
    print("=" * 50)
    
    # Создаём бота-эксперта
    expert_prompt = """
    Ты эксперт по программированию с 10-летним опытом.
    Объясняй код детально, давай практические советы.
    Всегда приводи примеры кода при объяснениях.
    """
    
    expert_bot = LLMChatbot(system_prompt=expert_prompt.strip())
    
    # Серия тестовых вопросов
    test_questions = [
        "Что такое рекурсия в программировании?",
        "Можешь показать пример рекурсивной функции?",
        "Какие есть альтернативы рекурсии?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Вопрос {i}: {question}")
        print("-" * 30)
        
        result = expert_bot.send_message(question)
        
        if result["success"]:
            print(f"🤖 Эксперт: {result['response']}")
            print(f"   💡 Токенов использовано: {result['usage'].get('total_tokens', 'N/A')}")
        else:
            print(f"❌ Ошибка: {result['error']}")
    
    # Показываем итоговую статистику
    stats = expert_bot.get_conversation_summary()
    print(f"\n📊 Итоговая статистика демонстрации:")
    print(f"   Всего сообщений: {stats['messages_count']}")
    print(f"   Потрачено токенов: {stats['total_tokens_used']}")

if __name__ == "__main__":
    print("Выберите режим:")
    print("1. Интерактивный чат")
    print("2. Демонстрация возможностей")
    
    choice = input("Введите номер (1 или 2): ").strip()
    
    if choice == "1":
        create_interactive_chat()
    elif choice == "2":
        demo_chatbot_capabilities()
    else:
        print("❌ Неверный выбор. Запустите программу снова.")
```

### 🔍 Проверьте себя

Запустите чат-бота и протестируйте:

1. **Память диалога**: задайте вопрос, а потом спросите "А что ты думаешь о моём предыдущем вопросе?"
2. **Системный промпт**: заметьте, как бот ведёт себя согласно заданной роли
3. **Статистика**: используйте команду `stats` чтобы увидеть потребление токенов

Ответьте себе: как системный промпт влияет на поведение бота?

## 5. Создание суммаризатора документов

Давайте создадим ещё одно полезное приложение — суммаризатор текста:

```python
def create_text_summarizer():
    """Создаёт интеллектуальный суммаризатор текста"""
    
    class TextSummarizer:
        def __init__(self):
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            self.model = os.getenv("DEFAULT_MODEL", "microsoft/wizardlm-2-8x22b")
            
        def summarize(self, text, summary_length="medium", style="professional"):
            """
            Создаёт суммарий текста
            
            Args:
                text (str): Исходный текст для суммаризации
                summary_length (str): "short", "medium", "detailed" 
                style (str): "professional", "casual", "academic"
            """
            
            # Настройки для разных типов суммариев
            length_settings = {
                "short": "1-2 предложения, только самое важное",
                "medium": "1 абзац (3-4 предложения), основные идеи", 
                "detailed": "2-3 абзаца, подробный обзор с ключевыми деталями"
            }
            
            style_settings = {
                "professional": "деловой стиль, нейтральный тон",
                "casual": "простой разговорный язык, как для друзей",
                "academic": "научный стиль, с терминологией и структурой"
            }
            
            prompt = f"""
Создай суммарий следующего текста.

ТРЕБОВАНИЯ К СУММАРИЮ:
- Длина: {length_settings.get(summary_length, length_settings["medium"])}
- Стиль: {style_settings.get(style, style_settings["professional"])}
- Сохрани все ключевые идеи и важные факты
- Используй русский язык
- Структурируй информацию логично

ИСХОДНЫЙ ТЕКСТ:
{text}

СУММАРИЙ:
"""
            
            result = send_request_to_llm(prompt, model=self.model)
            return result
            
        def compare_summaries(self, text):
            """Создаёт суммарии разной длины для сравнения"""
            
            print("🔄 Создаём суммарии разной детализации...\n")
            
            lengths = ["short", "medium", "detailed"]
            results = {}
            
            for length in lengths:
                print(f"📝 Создаём {length} суммарий...")
                result = self.summarize(text, length)
                
                if result["success"]:
                    results[length] = {
                        "summary": result["response"],
                        "tokens": result["usage"].get("total_tokens", "N/A")
                    }
                else:
                    results[length] = {"error": result["error"]}
            
            return results
    
    # Демонстрация работы
    summarizer = TextSummarizer()
    
    # Пример текста для суммаризации
    sample_text = """
    Искусственный интеллект (ИИ) стремительно развивается и меняет мир вокруг нас. 
    Современные системы ИИ могут выполнять задачи, которые ещё недавно казались 
    исключительно человеческими: понимать естественный язык, создавать изображения, 
    писать код и даже сочинять музыку.
    
    Одним из наиболее значительных достижений стали большие языковые модели (LLM), 
    такие как GPT-4, Claude и другие. Эти модели обучаются на огромных объёмах текста 
    и могут генерировать связные, осмысленные ответы на самые разные вопросы. 
    Они находят применение в образовании, бизнесе, науке и творчестве.
    
    Однако развитие ИИ также вызывает обеспокоенность. Эксперты предупреждают 
    о возможных рисках: от потери рабочих мест до более серьёзных угроз безопасности. 
    Важно найти баланс между использованием преимуществ ИИ и минимизацией рисков.
    
    Будущее ИИ во многом зависит от того, как человечество будет развивать эту 
    технологию. Необходимо обеспечить этичное и ответственное развитие ИИ, 
    которое принесёт пользу всему обществу.
    """
    
    print("📊 ДЕМОНСТРАЦИЯ СУММАРИЗАТОРА")
    print("=" * 50)
    print("Исходный текст:")
    print(sample_text)
    print("\n" + "=" * 50)
    
    # Создаём суммарии разной длины
    comparison = summarizer.compare_summaries(sample_text)
    
    for length, data in comparison.items():
        print(f"\n📝 {length.upper()} суммарий:")
        print("-" * 30)
        
        if "summary" in data:
            print(data["summary"])
            print(f"💡 Токенов использовано: {data['tokens']}")
        else:
            print(f"❌ Ошибка: {data['error']}")
    
    return summarizer

# Интерактивный режим суммаризатора
def interactive_summarizer():
    """Интерактивный интерфейс для суммаризации текста"""
    
    summarizer = create_text_summarizer()
    
    print("\n🔧 ИНТЕРАКТИВНЫЙ СУММАРИЗАТОР")
    print("Введите текст для суммаризации (для завершения введите 'END' на новой строке):")
    
    # Сбор многострочного ввода
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        lines.append(line)
    
    text = '\n'.join(lines).strip()
    
    if not text:
        print("❌ Текст не введён!")
        return
    
    # Выбор настроек
    print("\nВыберите тип суммария:")
    print("1. Краткий (1-2 предложения)")  
    print("2. Средний (абзац)")
    print("3. Подробный (2-3 абзаца)")
    
    length_map = {"1": "short", "2": "medium", "3": "detailed"}
    length_choice = input("Введите номер (1-3): ").strip()
    length = length_map.get(length_choice, "medium")
    
    print("\nВыберите стиль:")
    print("1. Профессиональный")
    print("2. Разговорный")  
    print("3. Академический")
    
    style_map = {"1": "professional", "2": "casual", "3": "academic"}
    style_choice = input("Введите номер (1-3): ").strip()
    style = style_map.get(style_choice, "professional")
    
    # Создаём суммарий
    print(f"\n🔄 Создаём {length} суммарий в {style} стиле...")
    
    result = summarizer.summarize(text, length, style)
    
    if result["success"]:
        print(f"\n📋 СУММАРИЙ:")
        print("=" * 40)
        print(result["response"])
        print("=" * 40)
        print(f"💡 Использовано токенов: {result['usage'].get('total_tokens', 'N/A')}")
    else:
        print(f"❌ Ошибка при создании суммария: {result['error']}")
```

## Практические задания

### 🟢 Базовый уровень

**Задание 1: Модификация промпта**
Измените системный промпт чат-бота так, чтобы он вёл себя как:
- Строгий учитель математики
- Дружелюбный гид по городу
- Эксперт по здоровому питанию

Протестируйте каждую роль с одинаковыми вопросами и сравните ответы.

**Задание 2: Анализ токенов**
Создайте функцию, которая подсчитывает примерное количество токенов в тексте по формуле: `количество_символов / 4`. Сравните ваши расчёты с реальным потреблением API.

**Задание 3: Обработка ошибок**
Добавьте в код обработку следующих ситуаций:
- Отсутствие интернет-соединения
- Превышение лимита токенов
- Неверный API ключ

### 🟡 Средний уровень

**Задание 4: Улучшенный суммаризатор**
Расширьте суммаризатор следующими возможностями:
- Определение языка текста и суммаризация на том же языке
- Извлечение ключевых слов из текста
- Оценка тональности исходного текста

**Задание 5: Чат-бот с памятью файлов**
Создайте чат-бота, который:
- Автоматически сохраняет каждый диалог
- Может загружать предыдущие диалоги
- Показывает список сохранённых сессий

**Задание 6: Многомодельный бот**
Реализуйте возможность переключения между разными моделями в процессе диалога. Добавьте команды:
- `/model list` — показать доступные модели
- `/model set <название>` — переключить модель
- `/model compare <вопрос>` — получить ответы от 2-3 разных моделей

### 🔴 Продвинутый уровень

**Задание 7: Система с ролями**
Создайте систему, где пользователь может:
- Создавать собственные роли для бота с описанием
- Сохранять роли в файл
- Быстро переключаться между сохранёнными ролями
- Делиться ролями с другими пользователями (экспорт/импорт)

**Задание 8: Анализатор диалогов**
Разработайте инструмент анализа сохранённых диалогов:
- Статистика по токенам и стоимости
- Самые частые темы и вопросы
- Анализ качества ответов бота
- Визуализация данных (если знаете matplotlib)

**Задание 9: Пакетная обработка**
Создайте приложение для пакетной обработки текстов:
- Загрузка множества файлов
- Применение одинаковой операции ко всем файлам (суммаризация, перевод, анализ)
- Сохранение результатов в структурированном виде
- Прогресс-бар для отслеживания выполнения

## Контрольные вопросы

1. **Что такое токен и как он влияет на стоимость API запросов?**
   <details>
   <summary>Ответ</summary>
   Токен — это единица текста для модели (примерно ¾ слова для русского языка). API взимает плату за количество токенов в запросе и ответе. Понимание токенизации помогает оптимизировать промпты и контролировать расходы. Например, 1000 символов текста ≈ 200-250 токенов.
   </details>

2. **В чём разница между prompt и completion в контексте API?**
   <details>
   <summary>Ответ</summary>
   Prompt — это входной текст, который мы отправляем модели (наш запрос). Completion — это текст, который генерирует модель в ответ на промпт. В стоимость входят токены как промпта, так и completion. В API это разделено для точного подсчёта потребления.
   </details>

3. **Как параметр temperature влияет на генерацию текста?**
   <details>
   <summary>Ответ</summary>
   Temperature контролирует случайность и креативность ответов: 0.0 = детерминированные, предсказуемые ответы; 1.0 = очень креативные, иногда непредсказуемые ответы. Для фактических вопросов используют низкие значения (0.1-0.3), для творчества — высокие (0.7-0.9).
   </details>

4. **Почему важно сохранять историю диалога в messages?**
   <details>
   <summary>Ответ</summary>
   LLM не имеют встроенной памяти между запросами. Передача истории в поле messages позволяет модели понимать контекст диалога, отвечать на уточняющие вопросы, использовать информацию из предыдущих сообщений. Без этого каждый запрос будет обрабатываться изолированно.
   </details>

5. **Как обработать ситуацию превышения лимита токенов?**
   <details>
   <summary>Ответ</summary>
   Варианты решения: 1) Сократить промпт, удалив менее важную информацию; 2) Разбить задачу на части; 3) Использовать модель с большим контекстным окном; 4) Суммаризировать предыдущую историю диалога; 5) Увеличить max_tokens в запросе, если проблема в ответе.
   </details>

## Заключение урока

### Что мы изучили

В этом уроке вы сделали важный переход от теории к практике:

- **Техническая подготовка**: настроили окружение разработчика и получили доступ к LLM API
- **Первое приложение**: создали функцию для отправки запросов к языковой модели
- **API взаимодействие**: изучили структуру запросов и ответов, разобрались с токенами и параметрами
- **Полноценные приложения**: разработали чат-бота с памятью диалога и суммаризатор документов

### Связь с предыдущим уроком

Теоретические знания из первого урока теперь обрели практическое воплощение:
- **Self-attention в действии**: вы видели, как модель использует контекст диалога
- **Ограничения на практике**: столкнулись с лимитами токенов и необходимостью оптимизации
- **Галлюцинации**: могли наблюдать, как модель иногда генерирует неточную информацию

### Что нас ждёт в следующем уроке

В третьем уроке **"Базовый промпт-инжиниринг"** мы изучим искусство создания эффективных промптов:
- Научимся писать промпты, которые дают нужные результаты с первого раза
- Освоим техники управления стилем и форматом ответов
- Изучим few-shot learning — обучение модели на примерах
- Создадим библиотеку готовых промпт-шаблонов

### Ваши достижения

🎉 **Превосходный результат!** За один урок вы:
- ✅ Создали своё первое AI-приложение
- ✅ Научились взаимодействовать с реальной языковой моделью
- ✅ Получили практический опыт работы с API
- ✅ Разработали полноценного чат-бота

Теперь у вас есть работающая основа для создания любых приложений с использованием LLM!

**Готовы стать мастером промптов?** Переходите к [третьему уроку](lesson_3_prompt_engineering.md) и изучайте искусство общения с ИИ!

---

## Дополнительные материалы

### API документация:
- [OpenRouter API Reference](https://openrouter.ai/docs) — полная документация
- [Список доступных моделей](https://openrouter.ai/models) — характеристики и цены

### Инструменты разработчика:
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) — для подсчёта токенов
- [JSON Formatter](https://jsonformatter.org/) — для форматирования API ответов
- [Postman](https://www.postman.com/) — для тестирования API запросов

### Расширенные возможности:
- [Streaming responses](https://openrouter.ai/docs#streaming) — получение ответа по частям
- [Function calling](https://openrouter.ai/docs#function-calling) — интеграция с внешними функциями
- [Vision models](https://openrouter.ai/docs#vision) — работа с изображениями