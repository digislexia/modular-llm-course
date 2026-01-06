# Урок 4: Управление контекстом в длинных диалогах

## Введение

Представьте ситуацию: вы ведёте долгий разговор с AI-ассистентом, обсуждаете сложную тему, делитесь деталями проекта. Внезапно ассистент начинает "забывать" то, что вы говорили в начале разговора, игнорирует важные детали или даже противоречит сам себе. Знакомо?

Эта проблема возникает из-за **ограничений контекстного окна** — фундаментального ограничения современных языковых моделей. В этом уроке мы разберём, как эти ограничения работают и, что важнее, как их обходить с помощью умных стратегий управления контекстом.

**Цель урока**: Научиться строить системы, которые могут поддерживать длинные, содержательные диалоги без потери качества и памяти о предыдущих этапах разговора.

## Проблема: Когда контекста становится слишком много

### Что такое контекстное окно?

**Контекстное окно** — это максимальное количество токенов (примерно слов), которое модель может "удержать" в памяти одновременно. У разных моделей разные лимиты:

- GPT-3.5: ~4,000 токенов (≈ 3,000 слов)
- GPT-4: 8,000 или 32,000 токенов 
- Claude-3: до 200,000 токенов
- Gemini Pro: до 1,000,000 токенов

### Демонстрация проблемы

Давайте посмотрим, что происходит, когда диалог становится слишком длинным:

```python
import openai

# Симуляция очень длинного диалога
messages = [
    {"role": "system", "content": "Ты — помощник по планированию проектов."},
    {"role": "user", "content": "Я планирую запустить интернет-магазин книг. Назовём его 'BookHaven'. Цвет брендинга — синий."},
    {"role": "assistant", "content": "Отличная идея! BookHaven с синим брендингом звучит профессионально..."},
    # ... ещё 50 сообщений о деталях проекта ...
    {"role": "user", "content": "Напомни, как называется наш проект и какой у нас цвет бренда?"},
]

# При превышении лимита модель может "забыть" начало диалога
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
print(response.choices[0].message.content)
# Возможный ответ: "Извините, я не помню названия вашего проекта..."
```

**Проблемы, которые возникают:**
1. **Потеря памяти**: Модель забывает важную информацию из начала разговора
2. **Обрезание контекста**: API просто удаляет старые сообщения, что может нарушить логику диалога
3. **Противоречия**: Без контекста модель может давать противоречивые советы
4. **Ухудшение качества**: Ответы становятся менее персонализированными и точными

## Стратегия 1: Резюмирование диалога

### Принцип работы

Вместо хранения всей истории разговора мы периодически **сжимаем** её в краткое изложение, сохраняя ключевые факты и контекст.

### Реализация

```python
def summarize_conversation(messages):
    """Создаёт краткое резюме диалога для сохранения контекста"""
    
    # Формируем промпт для резюмирования
    conversation_text = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in messages[1:]  # Пропускаем системное сообщение
    ])
    
    summary_prompt = f"""
    Прочитай следующий диалог и создай краткое резюме (2-3 предложения), 
    сохранив самую важную информацию:
    
    {conversation_text}
    
    Резюме должно включать:
    1. Основную тему разговора
    2. Ключевые детали и решения
    3. Текущий статус/прогресс
    
    Резюме:
    """
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Ты — эксперт по сжатию информации."},
            {"role": "user", "content": summary_prompt}
        ],
        max_tokens=150
    )
    
    return response.choices[0].message.content

# Практическое применение
def manage_conversation_with_summary(messages, max_messages=10):
    """Управляет диалогом с помощью резюмирования"""
    
    if len(messages) > max_messages:
        # Берём системное сообщение
        system_message = messages[0]
        
        # Резюмируем старые сообщения (кроме последних 5)
        old_messages = messages[1:-5]
        summary = summarize_conversation([system_message] + old_messages)
        
        # Создаём новый контекст
        new_messages = [
            system_message,
            {"role": "system", "content": f"Контекст предыдущего разговора: {summary}"},
            *messages[-5:]  # Последние 5 сообщений
        ]
        
        return new_messages
    
    return messages

# Пример использования
def chat_with_summary_management():
    conversation = [
        {"role": "system", "content": "Ты — помощник по планированию проектов."}
    ]
    
    while True:
        user_input = input("Вы: ")
        if user_input.lower() == 'выход':
            break
            
        conversation.append({"role": "user", "content": user_input})
        
        # Управляем длиной диалога
        managed_conversation = manage_conversation_with_summary(conversation)
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=managed_conversation
        )
        
        ai_response = response.choices[0].message.content
        print(f"AI: {ai_response}")
        
        conversation.append({"role": "assistant", "content": ai_response})
```

### Преимущества и недостатки

**✅ Преимущества:**
- Сохраняет основную информацию из всего диалога
- Существенно экономит токены
- Подходит для диалогов с четкими фактами и решениями

**❌ Недостатки:**  
- Может потерять нюансы и детали
- Требует дополнительных вызовов API для резюмирования
- Зависит от качества резюмирования модели

## Стратегия 2: Скользящее окно

### Принцип работы

Мы сохраняем только **последние N сообщений**, полагая, что недавний контекст важнее старого.

### Реализация

```python
def sliding_window_context(messages, window_size=10):
    """Реализует стратегию скользящего окна"""
    
    # Всегда сохраняем системное сообщение
    if len(messages) <= window_size:
        return messages
    
    system_message = messages[0] if messages[0]["role"] == "system" else None
    
    # Берём последние window_size сообщений
    recent_messages = messages[-window_size:]
    
    if system_message:
        return [system_message] + recent_messages
    return recent_messages

# Адаптивное скользящее окно
def adaptive_sliding_window(messages, max_tokens=3000):
    """Динамически подстраивает размер окна под лимит токенов"""
    
    def count_tokens_approximate(messages):
        # Приблизительный подсчёт токенов (1 токен ≈ 0.75 слов)
        total_chars = sum(len(msg["content"]) for msg in messages)
        return total_chars // 3  # Грубая оценка
    
    # Начинаем с полного диалога
    current_messages = messages
    
    # Уменьшаем окно, пока не поместимся в лимит
    while count_tokens_approximate(current_messages) > max_tokens and len(current_messages) > 2:
        # Удаляем самое старое сообщение (кроме системного)
        if current_messages[0]["role"] == "system":
            current_messages = [current_messages[0]] + current_messages[2:]
        else:
            current_messages = current_messages[1:]
    
    return current_messages

# Пример использования
conversation = [
    {"role": "system", "content": "Ты — помощник по программированию."},
    {"role": "user", "content": "Помоги написать функцию для сортировки"},
    {"role": "assistant", "content": "Конечно! Какой алгоритм сортировки вас интересует?"},
    # ... много сообщений ...
]

# Применяем скользящее окно
windowed_conversation = adaptive_sliding_window(conversation, max_tokens=2000)
```

### Когда использовать скользящее окно?

**Подходит для:**
- Технических консультаций (код, отладка)
- Коротких сессий вопрос-ответ  
- Диалогов, где важен только недавний контекст

**Не подходит для:**
- Долгосрочного планирования
- Диалогов с накопленной информацией
- Случаев, когда важны ранние детали разговора

## Стратегия 3: Выборочное сохранение (Smart Context)

### Принцип работы

Мы анализируем сообщения и сохраняем только **действительно важные** части диалога, отбрасывая "шум".

### Реализация

```python
def extract_important_messages(messages):
    """Определяет важность сообщений и сохраняет ключевые"""
    
    importance_prompt = """
    Проанализируй следующий диалог и оцени важность каждого сообщения от 1 до 5:
    5 - критически важно (ключевая информация, решения, факты)
    4 - очень важно (важные детали, уточнения)
    3 - умеренно важно (полезная информация)
    2 - мало важно (общие вопросы, подтверждения)
    1 - не важно (приветствия, благодарности)
    
    Диалог:
    {conversation}
    
    Ответ в формате: сообщение_номер:оценка
    """
    
    conversation_text = ""
    for i, msg in enumerate(messages[1:], 1):  # Пропускаем системное сообщение
        conversation_text += f"{i}. {msg['role']}: {msg['content']}\n"
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user", 
            "content": importance_prompt.format(conversation=conversation_text)
        }],
        max_tokens=200
    )
    
    # Парсим оценки важности
    importance_scores = {}
    for line in response.choices[0].message.content.split('\n'):
        if ':' in line:
            try:
                msg_num, score = line.split(':')
                importance_scores[int(msg_num.strip())] = int(score.strip())
            except:
                continue
    
    return importance_scores

def smart_context_management(messages, importance_threshold=3):
    """Сохраняет только важные сообщения"""
    
    if len(messages) <= 8:  # Для коротких диалогов не фильтруем
        return messages
    
    # Получаем оценки важности
    importance_scores = extract_important_messages(messages)
    
    # Всегда сохраняем системное сообщение и последние 3 сообщения
    system_msg = [messages[0]] if messages[0]["role"] == "system" else []
    recent_messages = messages[-3:]
    
    # Фильтруем сообщения по важности
    important_messages = []
    for i, msg in enumerate(messages[1:-3], 1):
        if importance_scores.get(i, 3) >= importance_threshold:
            important_messages.append(msg)
    
    return system_msg + important_messages + recent_messages
```

### Продвинутая версия: Семантическое выборочное сохранение

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticContextManager:
    """Управляет контекстом на основе семантической важности"""
    
    def __init__(self, max_tokens=3000):
        self.max_tokens = max_tokens
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.conversation_memory = []
    
    def add_message(self, message):
        """Добавляет сообщение в память"""
        embedding = self.encoder.encode([message["content"]])[0]
        
        self.conversation_memory.append({
            **message,
            "embedding": embedding,
            "timestamp": len(self.conversation_memory)
        })
    
    def get_relevant_context(self, current_query, top_k=5):
        """Находит наиболее релевантные предыдущие сообщения"""
        
        if not self.conversation_memory:
            return []
        
        # Получаем эмбеддинг текущего запроса
        query_embedding = self.encoder.encode([current_query])[0]
        
        # Вычисляем схожесть с предыдущими сообщениями
        similarities = []
        for msg in self.conversation_memory:
            similarity = np.dot(query_embedding, msg["embedding"])
            similarities.append((similarity, msg))
        
        # Сортируем по релевантности
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Берём top_k наиболее релевантных + последние 2 сообщения
        relevant_messages = [msg for _, msg in similarities[:top_k]]
        recent_messages = self.conversation_memory[-2:]
        
        # Объединяем и убираем дубликаты
        all_messages = relevant_messages + recent_messages
        unique_messages = []
        seen_timestamps = set()
        
        for msg in all_messages:
            if msg["timestamp"] not in seen_timestamps:
                unique_messages.append(msg)
                seen_timestamps.add(msg["timestamp"])
        
        # Сортируем по времени
        unique_messages.sort(key=lambda x: x["timestamp"])
        
        return [{"role": msg["role"], "content": msg["content"]} 
                for msg in unique_messages]

# Пример использования
context_manager = SemanticContextManager()

def chat_with_semantic_context():
    conversation_history = []
    
    while True:
        user_input = input("Вы: ")
        if user_input.lower() == 'выход':
            break
        
        # Добавляем сообщение пользователя в память
        user_msg = {"role": "user", "content": user_input}
        context_manager.add_message(user_msg)
        
        # Получаем релевантный контекст
        relevant_context = context_manager.get_relevant_context(user_input)
        
        # Строим финальный контекст для модели
        messages = [
            {"role": "system", "content": "Ты — умный ассистент с отличной памятью."}
        ] + relevant_context + [user_msg]
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        ai_response = response.choices[0].message.content
        print(f"AI: {ai_response}")
        
        # Добавляем ответ ассистента в память
        assistant_msg = {"role": "assistant", "content": ai_response}
        context_manager.add_message(assistant_msg)
```

## Стратегия 4: Гибридный подход

### Комбинирование лучшего из всех миров

Часто наиболее эффективный подход — это **комбинация** разных стратегий в зависимости от ситуации:

```python
class HybridContextManager:
    """Комбинированный менеджер контекста"""
    
    def __init__(self, max_tokens=3000):
        self.max_tokens = max_tokens
        self.conversation_summary = ""
        self.recent_messages = []
        self.important_facts = []
    
    def manage_context(self, messages):
        """Применяет гибридную стратегию управления контекстом"""
        
        # 1. Если диалог короткий — возвращаем как есть
        if self._count_tokens(messages) <= self.max_tokens:
            return messages
        
        # 2. Создаём резюме старых сообщений
        if len(messages) > 20:
            old_messages = messages[1:-10]  # Исключаем system и последние 10
            self.conversation_summary = self._create_summary(old_messages)
        
        # 3. Извлекаем важные факты
        self.important_facts = self._extract_facts(messages)
        
        # 4. Сохраняем последние сообщения (скользящее окно)
        self.recent_messages = messages[-8:]
        
        # 5. Собираем финальный контекст
        final_context = [messages[0]]  # Системное сообщение
        
        # Добавляем резюме
        if self.conversation_summary:
            final_context.append({
                "role": "system", 
                "content": f"Контекст разговора: {self.conversation_summary}"
            })
        
        # Добавляем важные факты
        if self.important_facts:
            facts_text = "; ".join(self.important_facts)
            final_context.append({
                "role": "system",
                "content": f"Важная информация: {facts_text}"
            })
        
        # Добавляем недавние сообщения
        final_context.extend(self.recent_messages)
        
        return final_context
    
    def _count_tokens(self, messages):
        # Упрощённый подсчёт токенов
        return sum(len(msg["content"].split()) for msg in messages)
    
    def _create_summary(self, messages):
        # Логика создания резюме (как в предыдущем примере)
        pass
    
    def _extract_facts(self, messages):
        # Логика извлечения ключевых фактов
        return ["Проект BookHaven", "Синий брендинг", "Интернет-магазин книг"]
```

## Практическое задание: Создание менеджера контекста

Теперь попробуйте создать собственную систему управления контекстом:

### Задание 1: Базовый менеджер

```python
class BasicContextManager:
    def __init__(self, strategy="sliding_window", max_messages=10):
        """
        strategy: "sliding_window", "summary", "smart"
        max_messages: максимальное количество сообщений
        """
        self.strategy = strategy
        self.max_messages = max_messages
        self.conversation = []
    
    def add_message(self, role, content):
        """Добавляет сообщение в диалог"""
        self.conversation.append({"role": role, "content": content})
    
    def get_context(self):
        """Возвращает управляемый контекст в зависимости от стратегии"""
        
        if self.strategy == "sliding_window":
            return self._sliding_window()
        elif self.strategy == "summary":
            return self._with_summary()
        elif self.strategy == "smart":
            return self._smart_selection()
        else:
            return self.conversation
    
    def _sliding_window(self):
        """Ваша реализация скользящего окна"""
        # TODO: Реализуйте эту функцию
        pass
    
    def _with_summary(self):
        """Ваша реализация с резюмированием"""
        # TODO: Реализуйте эту функцию
        pass
    
    def _smart_selection(self):
        """Ваша реализация умного отбора"""
        # TODO: Реализуйте эту функцию
        pass

# Тест вашего менеджера
manager = BasicContextManager(strategy="sliding_window", max_messages=5)

# Добавляем много сообщений
for i in range(10):
    manager.add_message("user", f"Сообщение пользователя {i}")
    manager.add_message("assistant", f"Ответ ассистента {i}")

# Проверяем, что контекст управляется правильно
context = manager.get_context()
print(f"Количество сообщений в контексте: {len(context)}")
```

### Задание 2: А/Б тестирование стратегий

Сравните эффективность разных стратегий на одном и том же длинном диалоге:

```python
def compare_context_strategies(long_conversation):
    """Сравнивает разные стратегии управления контекстом"""
    
    strategies = {
        "no_management": long_conversation,
        "sliding_window": sliding_window_context(long_conversation, 8),
        "with_summary": manage_conversation_with_summary(long_conversation, 8),
        "smart_selection": smart_context_management(long_conversation, 3)
    }
    
    test_query = "Напомни, о чём мы говорили в самом начале?"
    
    for strategy_name, context in strategies.items():
        print(f"\n=== Стратегия: {strategy_name} ===")
        print(f"Размер контекста: {len(context)} сообщений")
        
        # Добавляем тестовый вопрос
        test_messages = context + [{"role": "user", "content": test_query}]
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=test_messages,
                max_tokens=100
            )
            print(f"Ответ: {response.choices[0].message.content}")
        except Exception as e:
            print(f"Ошибка: {e}")

# Создайте длинный диалог и протестируйте
# compare_context_strategies(your_long_conversation)
```

## Продвинутые техники

### 1. Иерархическое резюмирование

```python
class HierarchicalSummarizer:
    """Создаёт многоуровневое резюме диалога"""
    
    def create_hierarchical_summary(self, messages, levels=3):
        """
        Создаёт резюме на разных уровнях детализации
        levels=1: Очень краткое (1-2 предложения)
        levels=2: Среднее (3-4 предложения)  
        levels=3: Подробное (5-6 предложений)
        """
        summaries = {}
        
        for level in range(1, levels + 1):
            prompt = f"""
            Создай резюме диалога уровня {level}:
            Уровень 1: Очень краткое (1-2 предложения)
            Уровень 2: Средней детализации (3-4 предложения)
            Уровень 3: Подробное (5-6 предложений)
            
            Диалог: {self._format_messages(messages)}
            
            Резюме уровня {level}:
            """
            
            # Вызов к модели для создания резюме
            summaries[level] = self._get_summary_from_model(prompt)
        
        return summaries
```

### 2. Контекстуальная память с тегами

```python
class TaggedContextManager:
    """Менеджер контекста с семантическими тегами"""
    
    def __init__(self):
        self.tagged_memories = {
            "facts": [],      # Конкретные факты
            "decisions": [],  # Принятые решения
            "preferences": [], # Предпочтения пользователя
            "context": []     # Общий контекст
        }
    
    def categorize_and_store(self, message):
        """Категоризирует сообщение и сохраняет в соответствующую категорию"""
        
        categorization_prompt = f"""
        Определи, к какой категории относится это сообщение:
        - facts: конкретные факты, данные, информация
        - decisions: решения, выборы, планы
        - preferences: предпочтения, мнения, вкусы
        - context: общий контекст, фоновая информация
        
        Сообщение: {message["content"]}
        
        Категория: (верни только одно слово)
        """
        
        # Получаем категорию от модели
        category = self._get_category_from_model(categorization_prompt)
        
        if category in self.tagged_memories:
            self.tagged_memories[category].append(message)
    
    def get_relevant_memories(self, query, category=None):
        """Получает релевантные воспоминания по запросу"""
        if category:
            return self.tagged_memories.get(category, [])
        
        # Если категория не указана, ищем во всех
        all_memories = []
        for memories in self.tagged_memories.values():
            all_memories.extend(memories)
        
        return all_memories
```

## Метрики качества управления контекстом

### Как оценить, работает ли ваша система?

```python
def evaluate_context_management(context_manager, test_scenarios):
    """Оценивает качество управления контекстом"""
    
    metrics = {
        "memory_retention": 0,     # Сохранение важной информации
        "relevance": 0,            # Релевантность сохранённого контекста
        "token_efficiency": 0,     # Эффективность использования токенов
        "consistency": 0           # Последовательность ответов
    }
    
    for scenario in test_scenarios:
        # Добавляем диалог в менеджер
        for message in scenario["conversation"]:
            context_manager.add_message(message["role"], message["content"])
        
        # Тестируем запросы
        for test in scenario["tests"]:
            context = context_manager.get_context()
            
            # Проверяем, содержит ли контекст нужную информацию
            expected_info = test["expected_info"]
            context_text = " ".join([msg["content"] for msg in context])
            
            # Метрика сохранения памяти
            if any(info in context_text for info in expected_info):
                metrics["memory_retention"] += 1
            
            # Метрика эффективности токенов
            token_count = sum(len(msg["content"].split()) for msg in context)
            metrics["token_efficiency"] += 1000 / token_count  # Чем меньше токенов, тем лучше
    
    # Нормализуем метрики
    total_tests = sum(len(s["tests"]) for s in test_scenarios)
    for key in metrics:
        metrics[key] /= total_tests
    
    return metrics

# Пример тестовых сценариев
test_scenarios = [
    {
        "conversation": [
            {"role": "system", "content": "Ты помощник по планированию"},
            {"role": "user", "content": "Мой проект называется SuperApp"},
            {"role": "assistant", "content": "Отлично! Расскажи больше о SuperApp"},
            # ... много сообщений ...
            {"role": "user", "content": "Как назывался мой проект?"}
        ],
        "tests": [
            {
                "query": "Как назывался мой проект?",
                "expected_info": ["SuperApp"]
            }
        ]
    }
]
```

## Заключение

Управление контекстом — это одна из самых важных практических проблем при создании LLM-приложений. Хорошо спроектированная система управления контекстом может:

1. **Экономить деньги**: меньше токенов = меньше затрат на API
2. **Улучшать качество**: сохранение важной информации улучшает ответы
3. **Повышать пользовательский опыт**: пользователи не чувствуют, что AI их "забывает"
4. **Масштабировать решение**: позволяет создавать долгосрочные диалоговые системы

### Практические рекомендации:

**Выбор стратегии в зависимости от типа приложения:**

- **Скользящее окно**: для технической поддержки, отладки кода
- **Резюмирование**: для консультаций, планирования проектов  
- **Выборочное сохранение**: для обучающих диалогов, экспертных систем
- **Гибридный подход**: для комплексных приложений общего назначения

**Помните о балансе:**
- Качество vs стоимость (резюмирование требует дополнительных вызовов)
- Полнота vs краткость (больше контекста != лучше качество)
- Автоматизация vs контроль (иногда лучше дать пользователю выбор)

В следующем уроке мы перейдём к построению RAG-систем, которые позволят нашим моделям работать с внешними источниками знаний и ещё более эффективно управлять информацией.

## Домашнее задание

1. **Реализуйте** три базовые стратегии управления контекстом из урока
2. **Протестируйте** их на диалоге из 30+ сообщений
3. **Сравните** результаты: какая стратегия лучше сохраняет важную информацию?
4. **Создайте** простой чат-бот с одной из стратегий управления контекстом

**Дополнительно (по желанию):**
- Реализуйте семантический менеджер контекста с помощью sentence-transformers
- Создайте систему оценки качества управления контекстом
- Экспериментируйте с комбинированием разных стратегий

## Дополнительные материалы

**Статьи:**
- [OpenAI: Managing conversation context](https://platform.openai.com/docs/guides/chat/managing-tokens)
- [Anthropic: Context window strategies](https://www.anthropic.com/news/claude-2-1)
- [LangChain: Memory management](https://python.langchain.com/docs/modules/memory/)

**Библиотеки:**
- `transformers` — для работы с токенизацией
- `sentence-transformers` — для семантических эмбеддингов  
- `langchain` — готовые решения для управления памятью

**Инструменты:**
- [tiktoken](https://github.com/openai/tiktoken) — точный подсчёт токенов для моделей OpenAI
- [Chroma](https://www.trychroma.com/) — векторная БД для семантического поиска
- [Redis](https://redis.io/) — для кэширования контекста

