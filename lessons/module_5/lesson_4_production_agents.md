# Урок 4: Устойчивость, память и масштабирование

## Введение

В предыдущих уроках мы создали работающего ReAct-агента. Но между прототипом и production-системой — большая разница.

**Проблемы прототипа:**
- Агент может зациклиться
- Может вызывать несуществующие инструменты
- Теряет контекст между сессиями
- Нет логирования и мониторинга
- Сложно отлаживать в production

В этом уроке мы превратим прототип в надёжную систему.

## Цели урока

После завершения урока вы сможете:

- ✅ Реализовать защиту от зацикливания и ошибок
- ✅ Добавить долгосрочную память агенту
- ✅ Настроить мониторинг и логирование
- ✅ Понимать основы мультиагентных систем

## Ключевые термины

- **Guardrails** — защитные механизмы агента
- **Rate Limiting** — ограничение частоты действий
- **Short-term Memory** — память в рамках задачи
- **Long-term Memory** — память между сессиями
- **Observability** — наблюдаемость системы

## 1. Проблемы агентов в production

### Типичные проблемы

```
┌────────────────────────────────────────────────────────────────┐
│                   ПРОБЛЕМЫ АГЕНТОВ                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. ЗАЦИКЛИВАНИЕ                                              │
│     THOUGHT: Нужно узнать погоду                              │
│     ACTION: search("погода")                                  │
│     OBSERVATION: Температура 20°C                             │
│     THOUGHT: Нужно узнать погоду  ← ПОВТОР!                  │
│     ACTION: search("погода")                                  │
│     ...бесконечно...                                         │
│                                                                │
│  2. ГАЛЛЮЦИНАЦИЯ ИНСТРУМЕНТОВ                                 │
│     ACTION: super_search("query")  ← НЕ СУЩЕСТВУЕТ!          │
│                                                                │
│  3. НЕВЕРНЫЕ АРГУМЕНТЫ                                        │
│     ACTION: calculator()  ← ЗАБЫЛ АРГУМЕНТ!                  │
│     ACTION: calculator("привет")  ← НЕВЕРНЫЙ ТИП!            │
│                                                                │
│  4. ПОТЕРЯ КОНТЕКСТА                                          │
│     Сессия 1: "Меня зовут Алекс"                             │
│     Сессия 2: "Как меня зовут?" → "Не знаю"  ← ЗАБЫЛ!        │
│                                                                │
│  5. ПРЕВЫШЕНИЕ ЛИМИТОВ                                        │
│     - Токены: слишком длинный контекст                       │
│     - Итерации: бесконечный цикл                             │
│     - API: rate limiting                                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Реальный пример зацикливания

```python
# Лог зацикливающегося агента

"""
--- Итерация 1 ---
THOUGHT: Нужно найти столицу Франции
ACTION: wikipedia("столица Франции")
OBSERVATION: Париж — столица и крупнейший город Франции...

--- Итерация 2 ---
THOUGHT: Нужно найти информацию о столице Франции
ACTION: wikipedia("столица Франции")
OBSERVATION: Париж — столица и крупнейший город Франции...

--- Итерация 3 ---
THOUGHT: Мне нужно узнать столицу Франции
ACTION: wikipedia("столица Франции")
OBSERVATION: Париж — столица и крупнейший город Франции...

... и так до max_iterations ...
"""

# Агент получил ответ в первой итерации, но не понял, что нужно остановиться!
```

## 2. Валидация и защитные механизмы

### Класс RobustReActAgent

```python
"""
Устойчивый ReAct-агент с защитными механизмами.
"""

import os
import re
import json
import time
from typing import Dict, List, Optional, Set
from collections import Counter
from datetime import datetime


class RobustReActAgent:
    """
    ReAct-агент с защитой от типичных проблем:
    - Зацикливание
    - Неверные инструменты
    - Превышение лимитов
    """
    
    def __init__(
        self,
        toolkit: 'ToolKit',
        model: str = "openai/gpt-4-turbo-preview",
        max_iterations: int = 10,
        max_same_action: int = 2,      # Макс. повторов одного действия
        max_consecutive_errors: int = 3,  # Макс. ошибок подряд
        cooldown_seconds: float = 0.5,    # Пауза между итерациями
        verbose: bool = True
    ):
        self.toolkit = toolkit
        self.model = model
        self.max_iterations = max_iterations
        self.max_same_action = max_same_action
        self.max_consecutive_errors = max_consecutive_errors
        self.cooldown_seconds = cooldown_seconds
        self.verbose = verbose
        
        # Трекинг для защиты
        self.action_history: List[str] = []
        self.consecutive_errors: int = 0
    
    def _check_loop(self, action: str) -> bool:
        """
        Проверяет, не зациклился ли агент.
        
        Returns:
            True если обнаружено зацикливание
        """
        # Добавляем действие в историю
        self.action_history.append(action)
        
        # Проверяем последние N действий
        recent_actions = self.action_history[-5:]
        action_counts = Counter(recent_actions)
        
        # Если одно действие повторяется слишком часто
        if action_counts[action] >= self.max_same_action:
            self._log(f"⚠️ Обнаружено зацикливание: '{action}' повторяется {action_counts[action]} раз")
            return True
        
        return False
    
    def _validate_tool(self, tool_name: str) -> tuple[bool, str]:
        """
        Проверяет существование инструмента.
        
        Returns:
            (is_valid, error_message)
        """
        if tool_name not in self.toolkit.tools:
            available = ", ".join(self.toolkit.tools.keys())
            return False, f"Инструмент '{tool_name}' не существует. Доступны: {available}"
        return True, ""
    
    def _validate_arguments(self, tool_name: str, args: dict) -> tuple[bool, str]:
        """
        Проверяет аргументы инструмента.
        
        Returns:
            (is_valid, error_message)
        """
        tool = self.toolkit.tools.get(tool_name)
        if not tool:
            return False, f"Инструмент {tool_name} не найден"
        
        # Проверяем обязательные параметры
        required_params = list(tool.parameters.keys())
        for param in required_params:
            if param not in args:
                return False, f"Отсутствует обязательный параметр: {param}"
        
        # Проверяем типы (упрощённо)
        for param, value in args.items():
            if param in tool.parameters:
                expected_type = tool.parameters[param].get("type", "string")
                if expected_type == "string" and not isinstance(value, str):
                    return False, f"Параметр {param} должен быть строкой"
                if expected_type == "number" and not isinstance(value, (int, float)):
                    return False, f"Параметр {param} должен быть числом"
        
        return True, ""
    
    def _handle_error(self, error: str) -> str:
        """
        Обрабатывает ошибку и формирует сообщение для агента.
        """
        self.consecutive_errors += 1
        
        if self.consecutive_errors >= self.max_consecutive_errors:
            return (f"КРИТИЧЕСКАЯ ОШИБКА: {self.consecutive_errors} ошибок подряд. "
                    f"Последняя: {error}. Попробуй другой подход или дай ANSWER.")
        
        return f"Ошибка: {error}. Попробуй другой инструмент или подход."
    
    def _reset_error_counter(self):
        """Сбрасывает счётчик ошибок после успешного действия"""
        self.consecutive_errors = 0
    
    def run(self, task: str) -> Dict:
        """
        Выполняет задачу с защитными механизмами.
        """
        self._log(f"\n{'='*70}")
        self._log(f"🎯 ЗАДАЧА: {task}")
        self._log(f"{'='*70}\n")
        
        # Сброс состояния
        self.action_history = []
        self.consecutive_errors = 0
        
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": task}
        ]
        
        actions_log = []
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            self._log(f"\n--- Итерация {iteration}/{self.max_iterations} ---\n")
            
            # Cooldown между итерациями
            if self.cooldown_seconds > 0:
                time.sleep(self.cooldown_seconds)
            
            # Запрос к LLM
            response = self._call_llm(messages)
            self._log(f"🤖 Агент:\n{response}\n")
            
            # Проверяем финальный ответ
            answer = self._extract_answer(response)
            if answer:
                return self._create_result(answer, iteration, actions_log, True)
            
            # Парсим и валидируем ACTION
            action = self._parse_action(response)
            
            if action:
                tool_name = action["tool"]
                tool_args = action.get("args", {})
                action_key = f"{tool_name}({json.dumps(tool_args, ensure_ascii=False)})"
                
                # Проверка на зацикливание
                if self._check_loop(action_key):
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": "OBSERVATION: Обнаружено зацикливание! "
                                   "Ты повторяешь одно и то же действие. "
                                   "Используй полученную информацию и дай ANSWER."
                    })
                    continue
                
                # Валидация инструмента
                is_valid, error = self._validate_tool(tool_name)
                if not is_valid:
                    observation = self._handle_error(error)
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})
                    continue
                
                # Выполнение с обработкой ошибок
                try:
                    # Получаем первый параметр инструмента
                    first_param = list(self.toolkit.tools[tool_name].parameters.keys())[0]
                    observation = self.toolkit.execute(tool_name, **{first_param: tool_args})
                    self._reset_error_counter()
                    
                    self._log(f"✅ Результат: {observation[:200]}...")
                    
                    actions_log.append({
                        "iteration": iteration,
                        "tool": tool_name,
                        "args": tool_args,
                        "result": observation[:500],
                        "success": True
                    })
                    
                except Exception as e:
                    observation = self._handle_error(str(e))
                    self._log(f"❌ Ошибка: {observation}")
                    
                    actions_log.append({
                        "iteration": iteration,
                        "tool": tool_name,
                        "args": tool_args,
                        "error": str(e),
                        "success": False
                    })
                
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})
            
            else:
                # Нет ACTION
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "Пожалуйста, укажи ACTION с инструментом или ANSWER с финальным ответом."
                })
        
        # Превышен лимит
        self._log(f"\n❌ Превышен лимит итераций")
        return self._create_result(
            "Не удалось найти ответ за отведённое количество шагов.",
            iteration, actions_log, False
        )
    
    def _create_result(self, answer: str, iterations: int, 
                       actions: List[Dict], success: bool) -> Dict:
        """Формирует результат выполнения"""
        return {
            "answer": answer,
            "iterations": iterations,
            "actions": actions,
            "success": success,
            "stats": {
                "total_actions": len(actions),
                "successful_actions": sum(1 for a in actions if a.get("success", False)),
                "tools_used": list(set(a["tool"] for a in actions))
            }
        }
    
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def _get_system_prompt(self) -> str:
        # Используем промпт из предыдущего урока
        pass
    
    def _call_llm(self, messages: List[Dict]) -> str:
        # Вызов API
        pass
    
    def _parse_action(self, response: str) -> Optional[Dict]:
        # Парсинг ACTION
        pass
    
    def _extract_answer(self, response: str) -> Optional[str]:
        # Извлечение ANSWER
        pass
```

### Тестирование защиты

```python
def test_loop_protection():
    """Тест защиты от зацикливания"""
    
    toolkit = ToolKit()
    toolkit.register(wikipedia_tool)
    
    agent = RobustReActAgent(
        toolkit=toolkit,
        max_same_action=2,
        verbose=True
    )
    
    # Агент с плохим промптом, склонный к зацикливанию
    result = agent.run("Расскажи о Python (используй wikipedia несколько раз)")
    
    # Должен остановиться, а не зациклиться
    assert result["iterations"] < agent.max_iterations
    print("✅ Защита от зацикливания работает")


def test_invalid_tool():
    """Тест обработки несуществующего инструмента"""
    
    toolkit = ToolKit()
    toolkit.register(calculator_tool)
    
    agent = RobustReActAgent(toolkit=toolkit)
    
    # Форсируем вызов несуществующего инструмента
    # (в реальности это сложнее протестировать)
    
    is_valid, error = agent._validate_tool("super_calculator")
    assert not is_valid
    assert "не существует" in error
    print("✅ Валидация инструментов работает")
```

## 3. Долгосрочная память

### Архитектура памяти

```
┌─────────────────────────────────────────────────────────────────┐
│                      СИСТЕМА ПАМЯТИ                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              КРАТКОВРЕМЕННАЯ ПАМЯТЬ                      │   │
│  │              (Short-term Memory)                         │   │
│  │                                                          │   │
│  │  • Контекст текущей задачи                              │   │
│  │  • Промежуточные результаты                             │   │
│  │  • История действий в сессии                            │   │
│  │  • Очищается после завершения задачи                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               ДОЛГОСРОЧНАЯ ПАМЯТЬ                        │   │
│  │               (Long-term Memory)                         │   │
│  │                                                          │   │
│  │  • Факты о пользователе (имя, предпочтения)             │   │
│  │  • Результаты прошлых задач                             │   │
│  │  • Обучение на ошибках                                  │   │
│  │  • Сохраняется между сессиями                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Реализация памяти

```python
"""
Система памяти для агента.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class AgentMemory:
    """
    Система памяти агента с кратковременным и долгосрочным хранением.
    """
    
    def __init__(self, storage_path: str = "agent_memory.json"):
        """
        Инициализация памяти.
        
        Args:
            storage_path: Путь к файлу долгосрочной памяти
        """
        self.storage_path = Path(storage_path)
        
        # Кратковременная память (текущая сессия)
        self.short_term: Dict[str, Any] = {
            "current_task": None,
            "conversation": [],
            "actions": [],
            "intermediate_results": []
        }
        
        # Долгосрочная память (между сессиями)
        self.long_term: Dict[str, Any] = self._load_long_term()
    
    def _load_long_term(self) -> Dict:
        """Загружает долгосрочную память из файла"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Ошибка загрузки памяти: {e}")
        
        return {
            "user_facts": {},      # Факты о пользователе
            "learned_facts": {},   # Выученные факты
            "task_history": [],    # История задач
            "error_patterns": [],  # Паттерны ошибок
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    def _save_long_term(self):
        """Сохраняет долгосрочную память в файл"""
        self.long_term["updated_at"] = datetime.now().isoformat()
        
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.long_term, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения памяти: {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # КРАТКОВРЕМЕННАЯ ПАМЯТЬ
    # ═══════════════════════════════════════════════════════════════
    
    def start_task(self, task: str):
        """Начинает новую задачу"""
        self.short_term = {
            "current_task": task,
            "started_at": datetime.now().isoformat(),
            "conversation": [],
            "actions": [],
            "intermediate_results": []
        }
    
    def add_message(self, role: str, content: str):
        """Добавляет сообщение в разговор"""
        self.short_term["conversation"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_action(self, tool: str, args: dict, result: str, success: bool):
        """Добавляет выполненное действие"""
        self.short_term["actions"].append({
            "tool": tool,
            "args": args,
            "result": result[:500],
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_intermediate_result(self, key: str, value: Any):
        """Добавляет промежуточный результат"""
        self.short_term["intermediate_results"].append({
            "key": key,
            "value": value
        })
    
    def get_context_summary(self) -> str:
        """
        Формирует краткое резюме контекста для добавления в промпт.
        """
        summary_parts = []
        
        # Текущая задача
        if self.short_term.get("current_task"):
            summary_parts.append(f"Текущая задача: {self.short_term['current_task']}")
        
        # Последние действия
        actions = self.short_term.get("actions", [])[-3:]
        if actions:
            summary_parts.append("Последние действия:")
            for action in actions:
                status = "✅" if action["success"] else "❌"
                summary_parts.append(f"  {status} {action['tool']}: {action['result'][:100]}...")
        
        # Промежуточные результаты
        results = self.short_term.get("intermediate_results", [])
        if results:
            summary_parts.append("Промежуточные результаты:")
            for r in results[-5:]:
                summary_parts.append(f"  • {r['key']}: {r['value']}")
        
        return "\n".join(summary_parts)
    
    def clear_short_term(self):
        """Очищает кратковременную память"""
        self.short_term = {
            "current_task": None,
            "conversation": [],
            "actions": [],
            "intermediate_results": []
        }
    
    # ═══════════════════════════════════════════════════════════════
    # ДОЛГОСРОЧНАЯ ПАМЯТЬ
    # ═══════════════════════════════════════════════════════════════
    
    def remember_user_fact(self, key: str, value: str):
        """
        Запоминает факт о пользователе.
        
        Args:
            key: Ключ (например, "имя", "город", "предпочтения")
            value: Значение
        """
        self.long_term["user_facts"][key] = {
            "value": value,
            "remembered_at": datetime.now().isoformat()
        }
        self._save_long_term()
        print(f"📝 Запомнил: {key} = {value}")
    
    def remember_fact(self, key: str, value: str, source: str = "conversation"):
        """
        Запоминает общий факт.
        
        Args:
            key: Ключ факта
            value: Значение
            source: Источник информации
        """
        self.long_term["learned_facts"][key] = {
            "value": value,
            "source": source,
            "learned_at": datetime.now().isoformat()
        }
        self._save_long_term()
    
    def recall_user_fact(self, key: str) -> Optional[str]:
        """Вспоминает факт о пользователе"""
        fact = self.long_term["user_facts"].get(key)
        return fact["value"] if fact else None
    
    def recall_fact(self, key: str) -> Optional[str]:
        """Вспоминает общий факт"""
        fact = self.long_term["learned_facts"].get(key)
        return fact["value"] if fact else None
    
    def search_facts(self, query: str) -> List[Dict]:
        """
        Ищет релевантные факты по запросу.
        (Упрощённый поиск по подстроке)
        """
        results = []
        query_lower = query.lower()
        
        # Поиск в фактах о пользователе
        for key, fact in self.long_term["user_facts"].items():
            if query_lower in key.lower() or query_lower in fact["value"].lower():
                results.append({"type": "user", "key": key, "value": fact["value"]})
        
        # Поиск в общих фактах
        for key, fact in self.long_term["learned_facts"].items():
            if query_lower in key.lower() or query_lower in fact["value"].lower():
                results.append({"type": "learned", "key": key, "value": fact["value"]})
        
        return results
    
    def save_task_result(self, task: str, result: str, success: bool):
        """Сохраняет результат задачи в историю"""
        self.long_term["task_history"].append({
            "task": task,
            "result": result[:500],
            "success": success,
            "actions_count": len(self.short_term.get("actions", [])),
            "completed_at": datetime.now().isoformat()
        })
        
        # Ограничиваем размер истории
        if len(self.long_term["task_history"]) > 100:
            self.long_term["task_history"] = self.long_term["task_history"][-100:]
        
        self._save_long_term()
    
    def log_error_pattern(self, error: str, context: str):
        """Логирует паттерн ошибки для обучения"""
        self.long_term["error_patterns"].append({
            "error": error,
            "context": context[:200],
            "occurred_at": datetime.now().isoformat()
        })
        
        # Ограничиваем размер
        if len(self.long_term["error_patterns"]) > 50:
            self.long_term["error_patterns"] = self.long_term["error_patterns"][-50:]
        
        self._save_long_term()
    
    def get_memory_summary(self) -> str:
        """Формирует резюме долгосрочной памяти для промпта"""
        summary_parts = []
        
        # Факты о пользователе
        if self.long_term["user_facts"]:
            summary_parts.append("Известные факты о пользователе:")
            for key, fact in list(self.long_term["user_facts"].items())[:10]:
                summary_parts.append(f"  • {key}: {fact['value']}")
        
        # Последние успешные задачи
        recent_tasks = [t for t in self.long_term["task_history"][-5:] if t["success"]]
        if recent_tasks:
            summary_parts.append("\nНедавние задачи:")
            for task in recent_tasks:
                summary_parts.append(f"  • {task['task'][:50]}...")
        
        return "\n".join(summary_parts) if summary_parts else ""
```

### Интеграция памяти с агентом

```python
class MemoryEnabledAgent(RobustReActAgent):
    """
    Агент с поддержкой долгосрочной памяти.
    """
    
    def __init__(self, *args, memory_path: str = "agent_memory.json", **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = AgentMemory(memory_path)
        
        # Добавляем инструмент для работы с памятью
        self._add_memory_tools()
    
    def _add_memory_tools(self):
        """Добавляет инструменты для работы с памятью"""
        
        remember_tool = Tool(
            name="remember",
            description="Запоминает важную информацию о пользователе для будущих сессий. "
                        "Используй для сохранения имени, предпочтений, важных фактов.",
            func=lambda key, value: self._remember(key, value),
            parameters={
                "key": {"type": "string", "description": "Что запомнить (например, 'имя_пользователя')"},
                "value": {"type": "string", "description": "Значение"}
            }
        )
        
        recall_tool = Tool(
            name="recall",
            description="Вспоминает ранее сохранённую информацию.",
            func=lambda query: self._recall(query),
            parameters={
                "query": {"type": "string", "description": "Что вспомнить"}
            }
        )
        
        self.toolkit.register(remember_tool)
        self.toolkit.register(recall_tool)
    
    def _remember(self, key: str, value: str) -> str:
        """Обработчик инструмента remember"""
        self.memory.remember_user_fact(key, value)
        return f"Запомнил: {key} = {value}"
    
    def _recall(self, query: str) -> str:
        """Обработчик инструмента recall"""
        # Сначала пробуем точный поиск
        value = self.memory.recall_user_fact(query)
        if value:
            return f"{query}: {value}"
        
        # Затем нечёткий поиск
        results = self.memory.search_facts(query)
        if results:
            return "\n".join([f"{r['key']}: {r['value']}" for r in results])
        
        return f"Не найдено информации по запросу: {query}"
    
    def _get_system_prompt(self) -> str:
        """Добавляет контекст памяти в системный промпт"""
        base_prompt = super()._get_system_prompt()
        memory_context = self.memory.get_memory_summary()
        
        if memory_context:
            return f"{base_prompt}\n\n## Контекст из памяти\n\n{memory_context}"
        
        return base_prompt
    
    def run(self, task: str) -> Dict:
        """Выполняет задачу с сохранением в память"""
        # Начинаем задачу
        self.memory.start_task(task)
        
        # Выполняем
        result = super().run(task)
        
        # Сохраняем результат
        self.memory.save_task_result(task, result["answer"], result["success"])
        
        # Очищаем кратковременную память
        self.memory.clear_short_term()
        
        return result
```

### Демонстрация памяти

```python
def demo_memory():
    """Демонстрация работы памяти агента"""
    
    toolkit = ToolKit()
    toolkit.register(calculator_tool)
    toolkit.register(wikipedia_tool)
    
    agent = MemoryEnabledAgent(
        toolkit=toolkit,
        memory_path="demo_memory.json",
        verbose=True
    )
    
    # Сессия 1: Представление
    print("\n" + "="*50)
    print("СЕССИЯ 1")
    print("="*50)
    
    agent.run("Привет! Меня зовут Алексей, я учу Python. Запомни это.")
    
    # Сессия 2: Проверка памяти
    print("\n" + "="*50)
    print("СЕССИЯ 2 (новая сессия)")
    print("="*50)
    
    agent.run("Как меня зовут и что я изучаю?")
    
    # Сессия 3: Использование памяти
    print("\n" + "="*50)
    print("СЕССИЯ 3")
    print("="*50)
    
    agent.run("Порекомендуй мне ресурсы для изучения, учитывая мои интересы.")


if __name__ == "__main__":
    demo_memory()
```

## 4. Мониторинг и логирование

### Класс MonitoredAgent

```python
"""
Агент с мониторингом и логированием.
"""

import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class AgentMetrics:
    """Метрики работы агента"""
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_iterations: int = 0
    total_tool_calls: int = 0
    total_errors: int = 0
    total_time_seconds: float = 0.0
    tool_usage: Dict[str, int] = field(default_factory=dict)
    errors_by_type: Dict[str, int] = field(default_factory=dict)


class MonitoredAgent:
    """
    Обёртка для агента с мониторингом.
    
    Собирает метрики:
    - Количество задач
    - Успешность
    - Использование инструментов
    - Время выполнения
    - Ошибки
    """
    
    def __init__(
        self,
        agent: 'RobustReActAgent',
        log_file: str = "agent_logs.jsonl",
        metrics_file: str = "agent_metrics.json"
    ):
        self.agent = agent
        self.log_file = log_file
        self.metrics_file = metrics_file
        
        self.metrics = AgentMetrics()
        self._load_metrics()
        
        # Настраиваем логирование
        self._setup_logging()
    
    def _setup_logging(self):
        """Настраивает логирование"""
        self.logger = logging.getLogger("AgentMonitor")
        self.logger.setLevel(logging.INFO)
        
        # Консольный вывод
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
    
    def _load_metrics(self):
        """Загружает сохранённые метрики"""
        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
                self.metrics = AgentMetrics(**data)
        except FileNotFoundError:
            pass
    
    def _save_metrics(self):
        """Сохраняет метрики"""
        with open(self.metrics_file, 'w') as f:
            json.dump({
                "total_tasks": self.metrics.total_tasks,
                "successful_tasks": self.metrics.successful_tasks,
                "failed_tasks": self.metrics.failed_tasks,
                "total_iterations": self.metrics.total_iterations,
                "total_tool_calls": self.metrics.total_tool_calls,
                "total_errors": self.metrics.total_errors,
                "total_time_seconds": self.metrics.total_time_seconds,
                "tool_usage": self.metrics.tool_usage,
                "errors_by_type": self.metrics.errors_by_type
            }, f, indent=2)
    
    def _log_event(self, event_type: str, data: Dict):
        """Записывает событие в лог-файл"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    
    def run(self, task: str) -> Dict:
        """
        Выполняет задачу с мониторингом.
        """
        self.metrics.total_tasks += 1
        start_time = time.time()
        
        self._log_event("task_started", {"task": task})
        self.logger.info(f"Начало задачи: {task[:50]}...")
        
        try:
            result = self.agent.run(task)
            
            # Обновляем метрики
            elapsed_time = time.time() - start_time
            self.metrics.total_time_seconds += elapsed_time
            self.metrics.total_iterations += result["iterations"]
            
            if result["success"]:
                self.metrics.successful_tasks += 1
                self.logger.info(f"Задача успешно завершена за {elapsed_time:.2f}с")
            else:
                self.metrics.failed_tasks += 1
                self.logger.warning(f"Задача завершена с ошибкой")
            
            # Считаем использование инструментов
            for action in result.get("actions", []):
                tool = action["tool"]
                self.metrics.tool_usage[tool] = self.metrics.tool_usage.get(tool, 0) + 1
                self.metrics.total_tool_calls += 1
                
                if not action.get("success", True):
                    self.metrics.total_errors += 1
                    error_type = action.get("error", "unknown")[:50]
                    self.metrics.errors_by_type[error_type] = \
                        self.metrics.errors_by_type.get(error_type, 0) + 1
            
            # Логируем результат
            self._log_event("task_completed", {
                "task": task,
                "success": result["success"],
                "iterations": result["iterations"],
                "elapsed_time": elapsed_time,
                "answer": result["answer"][:200]
            })
            
            result["elapsed_time"] = elapsed_time
            
        except Exception as e:
            self.metrics.failed_tasks += 1
            self.metrics.total_errors += 1
            
            error_type = type(e).__name__
            self.metrics.errors_by_type[error_type] = \
                self.metrics.errors_by_type.get(error_type, 0) + 1
            
            self._log_event("task_error", {
                "task": task,
                "error": str(e),
                "error_type": error_type
            })
            
            self.logger.error(f"Ошибка: {e}")
            raise
        
        finally:
            self._save_metrics()
        
        return result
    
    def get_report(self) -> str:
        """Формирует отчёт о работе агента"""
        m = self.metrics
        
        success_rate = m.successful_tasks / max(m.total_tasks, 1) * 100
        avg_iterations = m.total_iterations / max(m.total_tasks, 1)
        avg_time = m.total_time_seconds / max(m.total_tasks, 1)
        
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ОТЧЁТ О РАБОТЕ АГЕНТА                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Всего задач:          {m.total_tasks:>10}                              ║
║  Успешных:             {m.successful_tasks:>10} ({success_rate:.1f}%)                      ║
║  Неуспешных:           {m.failed_tasks:>10}                              ║
╠══════════════════════════════════════════════════════════════════╣
║  Среднее итераций:     {avg_iterations:>10.1f}                              ║
║  Среднее время (сек):  {avg_time:>10.2f}                              ║
║  Всего вызовов:        {m.total_tool_calls:>10}                              ║
║  Всего ошибок:         {m.total_errors:>10}                              ║
╠══════════════════════════════════════════════════════════════════╣
║  Использование инструментов:                                     ║"""
        
        for tool, count in sorted(m.tool_usage.items(), key=lambda x: -x[1]):
            report += f"\n║    • {tool:<20} {count:>5} вызовов                        ║"
        
        if m.errors_by_type:
            report += "\n╠══════════════════════════════════════════════════════════════════╣"
            report += "\n║  Типы ошибок:                                                    ║"
            for error, count in sorted(m.errors_by_type.items(), key=lambda x: -x[1])[:5]:
                report += f"\n║    • {error[:30]:<30} {count:>5}                        ║"
        
        report += "\n╚══════════════════════════════════════════════════════════════════╝"
        
        return report


def demo_monitoring():
    """Демонстрация мониторинга"""
    
    toolkit = ToolKit()
    toolkit.register(calculator_tool)
    toolkit.register(wikipedia_tool)
    toolkit.register(datetime_tool)
    
    base_agent = RobustReActAgent(toolkit=toolkit, verbose=False)
    agent = MonitoredAgent(base_agent)
    
    # Выполняем несколько задач
    tasks = [
        "Сколько будет 2 + 2?",
        "Кто такой Пушкин?",
        "Какой сейчас год?",
        "Вычисли sqrt(144)",
    ]
    
    for task in tasks:
        try:
            agent.run(task)
        except Exception as e:
            print(f"Ошибка: {e}")
    
    # Выводим отчёт
    print(agent.get_report())


if __name__ == "__main__":
    demo_monitoring()
```

## 5. Мультиагентные системы

### Введение

**Мультиагентная система** — это система, где несколько специализированных агентов работают вместе для решения сложных задач.

```
┌────────────────────────────────────────────────────────────────┐
│                  МУЛЬТИАГЕНТНАЯ СИСТЕМА                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│  │ RESEARCHER  │     │   WRITER    │     │   CRITIC    │      │
│  │   (поиск)   │────▶│  (текст)    │────▶│ (проверка)  │      │
│  └─────────────┘     └─────────────┘     └─────────────┘      │
│         │                   │                   │              │
│         └───────────────────┴───────────────────┘              │
│                            │                                   │
│                    ┌───────────────┐                          │
│                    │  COORDINATOR  │                          │
│                    │ (координатор) │                          │
│                    └───────────────┘                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Простая реализация

```python
"""
Простая мультиагентная система.
"""

from typing import Dict, List


class MultiAgentSystem:
    """
    Система из нескольких специализированных агентов.
    """
    
    def __init__(self):
        self.agents: Dict[str, 'ReActAgent'] = {}
    
    def register_agent(self, name: str, agent: 'ReActAgent', role: str):
        """
        Регистрирует агента в системе.
        
        Args:
            name: Имя агента
            agent: Экземпляр агента
            role: Роль агента (researcher, writer, critic)
        """
        self.agents[name] = {
            "agent": agent,
            "role": role
        }
        print(f"✅ Зарегистрирован агент: {name} ({role})")
    
    def run_pipeline(self, task: str, pipeline: List[str]) -> Dict:
        """
        Выполняет задачу через последовательность агентов.
        
        Args:
            task: Задача
            pipeline: Список имён агентов в порядке выполнения
            
        Returns:
            Итоговый результат
        """
        current_input = task
        results = []
        
        for agent_name in pipeline:
            if agent_name not in self.agents:
                raise ValueError(f"Агент {agent_name} не найден")
            
            agent_info = self.agents[agent_name]
            agent = agent_info["agent"]
            role = agent_info["role"]
            
            print(f"\n{'='*50}")
            print(f"🤖 Агент: {agent_name} ({role})")
            print(f"{'='*50}")
            
            # Формируем задачу для агента
            if results:
                agent_task = f"""
Предыдущие результаты:
{results[-1]['answer']}

Твоя задача: {current_input}
"""
            else:
                agent_task = current_input
            
            # Выполняем
            result = agent.run(agent_task)
            results.append({
                "agent": agent_name,
                "role": role,
                "answer": result["answer"],
                "success": result["success"]
            })
            
            # Обновляем входные данные для следующего агента
            current_input = result["answer"]
        
        return {
            "final_answer": results[-1]["answer"] if results else None,
            "pipeline_results": results,
            "success": all(r["success"] for r in results)
        }


def create_research_pipeline():
    """Создаёт пайплайн для исследования"""
    
    # Общий набор инструментов
    research_toolkit = ToolKit()
    research_toolkit.register(wikipedia_tool)
    research_toolkit.register(calculator_tool)
    
    writing_toolkit = ToolKit()
    writing_toolkit.register(datetime_tool)
    
    # Создаём агентов
    researcher = ReActAgent(
        toolkit=research_toolkit,
        max_iterations=5,
        verbose=True
    )
    # Можно добавить специальный промпт для исследователя
    
    writer = ReActAgent(
        toolkit=writing_toolkit,
        max_iterations=3,
        verbose=True
    )
    # Можно добавить специальный промпт для писателя
    
    # Создаём систему
    system = MultiAgentSystem()
    system.register_agent("researcher", researcher, "исследователь")
    system.register_agent("writer", writer, "писатель")
    
    return system


def demo_multi_agent():
    """Демонстрация мультиагентной системы"""
    
    system = create_research_pipeline()
    
    task = "Напиши краткую статью о Python для начинающих"
    
    result = system.run_pipeline(task, ["researcher", "writer"])
    
    print("\n" + "="*70)
    print("ИТОГОВЫЙ РЕЗУЛЬТАТ")
    print("="*70)
    print(result["final_answer"])


if __name__ == "__main__":
    demo_multi_agent()
```

## Практические задания

### 🟢 Базовый уровень

**Задание 1: Защита от зацикливания**

Протестируйте `RobustReActAgent` на задачах, которые могут вызвать зацикливание:

```python
tasks = [
    "Найди информацию о погоде (используй поиск много раз)",
    "Расскажи о Python, потом о JavaScript, потом снова о Python",
]
```

Проверьте, что агент корректно останавливается.

**Задание 2: Тестирование памяти**

Создайте `MemoryEnabledAgent` и протестируйте:
1. Сохранение факта о пользователе
2. Закрытие и открытие новой сессии
3. Восстановление факта

### 🟡 Средний уровень

**Задание 3: Расширение памяти**

Добавьте в `AgentMemory`:
- Поиск по похожим фактам (например, Levenshtein distance)
- Приоритизацию фактов (важные/обычные)
- Срок хранения (автоудаление старых фактов)

**Задание 4: Dashboard мониторинга**

Создайте простой веб-интерфейс для отображения метрик агента:
- Графики успешности
- Топ используемых инструментов
- Последние ошибки

### 🔴 Продвинутый уровень

**Задание 5: Полноценная мультиагентная система**

Создайте систему для написания статей:
1. **Researcher**: Ищет информацию
2. **Writer**: Пишет текст
3. **Critic**: Проверяет и даёт обратную связь
4. **Editor**: Финальная редактура

**Задание 6: Self-improving Agent**

Реализуйте агента, который:
- Анализирует свои ошибки
- Выявляет паттерны
- Корректирует своё поведение

## Контрольные вопросы

1. **Какие основные проблемы агентов в production?**
   <details>
   <summary>Ответ</summary>
   1. Зацикливание (повторение одних действий)
   2. Галлюцинация инструментов
   3. Неверные аргументы
   4. Потеря контекста между сессиями
   5. Превышение лимитов (токены, время, API)
   </details>

2. **Как защитить агента от зацикливания?**
   <details>
   <summary>Ответ</summary>
   1. Отслеживать историю действий
   2. Считать повторы одинаковых действий
   3. При превышении порога — прерывать и просить ANSWER
   4. Добавить cooldown между итерациями
   </details>

3. **Чем отличается кратковременная память от долгосрочной?**
   <details>
   <summary>Ответ</summary>
   Кратковременная: контекст текущей задачи, очищается после завершения.
   Долгосрочная: факты между сессиями, сохраняется в файл/БД.
   </details>

4. **Какие метрики важны для мониторинга агента?**
   <details>
   <summary>Ответ</summary>
   1. Success rate (% успешных задач)
   2. Average iterations (среднее число итераций)
   3. Tool usage (какие инструменты используются)
   4. Error rate (частота ошибок)
   5. Response time (время выполнения)
   </details>

5. **Когда нужны мультиагентные системы?**
   <details>
   <summary>Ответ</summary>
   1. Сложные задачи с разными этапами (исследование → написание → проверка)
   2. Специализированные знания (разные агенты для разных доменов)
   3. Параллельная обработка
   4. Взаимная проверка (один агент проверяет другого)
   </details>

## Финальный проект модуля

### Personal Assistant Agent

Создайте персонального помощника со всеми изученными возможностями.

**Требования:**

1. **Инструменты (5+ штук):**
   - Калькулятор
   - Поиск (Wikipedia или Web)
   - Текущее время
   - Погода (API)
   - Заметки (remember/recall)

2. **Защитные механизмы:**
   - Защита от зацикливания
   - Валидация инструментов
   - Обработка ошибок

3. **Память:**
   - Кратковременная (контекст задачи)
   - Долгосрочная (факты о пользователе)

4. **Мониторинг:**
   - Логирование всех действий
   - Сбор метрик
   - Отчёт о работе

5. **Интерфейс:**
   - CLI с красивым выводом
   - Команды: /help, /stats, /memory, /clear

**Критерии оценки:**
- Работоспособность: 30%
- Устойчивость: 25%
- Память: 20%
- Мониторинг: 15%
- Качество кода: 10%

## Заключение урока

### Что мы изучили

- **Защитные механизмы**: от зацикливания, невалидных инструментов, ошибок
- **Память**: кратковременная и долгосрочная, инструменты remember/recall
- **Мониторинг**: логирование, метрики, отчёты
- **Мультиагенты**: специализированные агенты, пайплайны

### Связь с другими модулями

| Модуль | Как связан с агентами |
|--------|----------------------|
| Модуль 2 (Промпты) | Chain-of-Thought в THOUGHT шагах |
| Модуль 3 (SchoolBot) | Можно превратить в агента |
| Модуль 4 (RAG) | RAG как инструмент агента |
| Модуль 6 (Финал) | Агент как основа финального проекта |

### Ваш прогресс

🎉 **Поздравляем!** Вы завершили Модуль 5 и теперь умеете:

- ✅ Создавать LLM-агентов с инструментами
- ✅ Реализовывать ReAct цикл
- ✅ Добавлять защитные механизмы
- ✅ Реализовывать долгосрочную память
- ✅ Настраивать мониторинг

**Следующий шаг:** Модуль 6 — Финальный проект, где вы объедините все знания в полноценное AI-приложение!

---

## Дополнительные материалы

### Документация:
- [LangChain Memory](https://python.langchain.com/docs/modules/memory/)
- [LangSmith (мониторинг)](https://docs.smith.langchain.com/)

### Фреймворки:
- [CrewAI](https://github.com/joaomdmoura/crewAI) — мультиагентные системы
- [AutoGen](https://github.com/microsoft/autogen) — от Microsoft
- [Langroid](https://github.com/langroid/langroid) — multi-agent programming

### Статьи:
- [Building Reliable LLM Applications](https://www.anthropic.com/)
- [Production LLM Systems](https://www.latent.space/)

