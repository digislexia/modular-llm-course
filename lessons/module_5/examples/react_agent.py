"""
ReAct агент: Reasoning + Acting.

Агент, который думает (THOUGHT), действует (ACTION) и наблюдает (OBSERVATION)
в цикле до получения финального ответа (ANSWER).

Запуск:
    python react_agent.py

Требования:
    - OPENROUTER_API_KEY в .env файле
    - pip install requests python-dotenv
"""

import os
import re
import json
import requests
import math
import ast
import operator
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from typing import Callable
from dotenv import load_dotenv

load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# ИНСТРУМЕНТЫ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    parameters: dict = field(default_factory=dict)
    
    def execute(self, **kwargs) -> str:
        try:
            return str(self.func(**kwargs))
        except Exception as e:
            return f"Ошибка: {e}"


class ToolKit:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        self.tools[tool.name] = tool
    
    def execute(self, name: str, arg: str) -> str:
        if name not in self.tools:
            available = ", ".join(self.tools.keys())
            return f"Инструмент '{name}' не найден. Доступны: {available}"
        
        tool = self.tools[name]
        first_param = list(tool.parameters.keys())[0]
        return tool.execute(**{first_param: arg})
    
    def describe(self) -> str:
        lines = []
        for t in self.tools.values():
            params = ", ".join(f'"{k}"' for k in t.parameters.keys())
            lines.append(f"• {t.name}({params}): {t.description}")
        return "\n".join(lines)


# Инструменты
def safe_calculator(expression: str) -> float:
    OPERATORS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg,
    }
    FUNCTIONS = {'sqrt': math.sqrt, 'abs': abs, 'round': round}
    CONSTANTS = {'pi': math.pi, 'e': math.e}
    
    def _eval(node):
        if isinstance(node, ast.Num): return node.n
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.Name): return CONSTANTS.get(node.id, 0)
        if isinstance(node, ast.BinOp):
            return OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return OPERATORS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call):
            return FUNCTIONS[node.func.id](*[_eval(a) for a in node.args])
        raise ValueError(f"Unsupported: {type(node)}")
    
    return float(_eval(ast.parse(expression, mode='eval').body))


def wikipedia_search(query: str) -> str:
    url = f"https://ru.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("extract", "")[:500]
        return f"Не найдено: {query}"
    except Exception as e:
        return f"Ошибка: {e}"


def get_current_time(timezone: str = "") -> str:
    now = datetime.now()
    days = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
    months = ["января", "февраля", "марта", "апреля", "мая", "июня",
              "июля", "августа", "сентября", "октября", "ноября", "декабря"]
    return f"{now.day} {months[now.month-1]} {now.year}, {days[now.weekday()]}, {now.strftime('%H:%M')}"


# ═══════════════════════════════════════════════════════════════════════════════
# ReAct АГЕНТ
# ═══════════════════════════════════════════════════════════════════════════════

class ReActAgent:
    """
    Агент с паттерном ReAct (Reasoning + Acting).
    
    Цикл работы:
    1. THOUGHT - рассуждение
    2. ACTION - вызов инструмента
    3. OBSERVATION - результат (от системы)
    4. Повторять до ANSWER
    """
    
    SYSTEM_PROMPT = """Ты — агент, решающий задачи через рассуждение и действия.

## Формат ответа

Следуй СТРОГО этому формату:

THOUGHT: (твои рассуждения о следующем шаге)
ACTION: tool_name("argument")

Жди OBSERVATION от системы, затем продолжай.

Когда готов дать финальный ответ:
THOUGHT: (финальные рассуждения)
ANSWER: (полный ответ на вопрос)

## Доступные инструменты

{tools}

## Правила

1. ВСЕГДА начинай с THOUGHT
2. После THOUGHT пиши ACTION
3. НЕ выдумывай OBSERVATION
4. Используй инструменты для проверки фактов
5. Отвечай на русском

## Пример

User: Сколько будет 15% от 200?

THOUGHT: Нужно вычислить 15% от 200. Это 200 * 0.15.
ACTION: calculator("200 * 0.15")

OBSERVATION: 30.0

THOUGHT: Результат получен.
ANSWER: 15% от 200 равно 30.
"""
    
    def __init__(
        self,
        toolkit: ToolKit,
        model: str = "openai/gpt-4-turbo-preview",
        max_iterations: int = 10,
        verbose: bool = True
    ):
        self.toolkit = toolkit
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError("Не найден OPENROUTER_API_KEY")
        
        self.system_prompt = self.SYSTEM_PROMPT.format(
            tools=toolkit.describe()
        )
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def _call_llm(self, messages: List[Dict]) -> str:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000
            },
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code}")
        
        return response.json()["choices"][0]["message"]["content"]
    
    def _parse_action(self, response: str) -> Optional[Dict]:
        """Извлекает ACTION из ответа"""
        # Паттерн: ACTION: tool_name("argument") или ACTION: tool_name(argument)
        patterns = [
            r'ACTION:\s*(\w+)\s*\(\s*["\']([^"\']*)["\']?\s*\)',
            r'ACTION:\s*(\w+)\s*\(\s*(.+?)\s*\)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return {
                    "tool": match.group(1),
                    "arg": match.group(2).strip('"\'')
                }
        return None
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """Извлекает ANSWER из ответа"""
        if "ANSWER:" in response:
            return response.split("ANSWER:")[-1].strip()
        return None
    
    def run(self, task: str) -> Dict:
        """
        Выполняет задачу через ReAct цикл.
        
        Args:
            task: Задача пользователя
            
        Returns:
            {
                "answer": str,
                "iterations": int,
                "actions": list,
                "success": bool
            }
        """
        self._log(f"\n{'='*70}")
        self._log(f"🎯 ЗАДАЧА: {task}")
        self._log(f"{'='*70}\n")
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task}
        ]
        
        actions_log = []
        
        for iteration in range(1, self.max_iterations + 1):
            self._log(f"\n--- Итерация {iteration} ---\n")
            
            # Запрос к LLM
            response = self._call_llm(messages)
            self._log(f"🤖 Агент:\n{response}\n")
            
            # Проверяем ANSWER
            answer = self._extract_answer(response)
            if answer:
                self._log(f"\n{'='*70}")
                self._log(f"✅ ОТВЕТ: {answer}")
                self._log(f"{'='*70}")
                
                return {
                    "answer": answer,
                    "iterations": iteration,
                    "actions": actions_log,
                    "success": True
                }
            
            # Парсим ACTION
            action = self._parse_action(response)
            
            if action:
                tool_name = action["tool"]
                tool_arg = action["arg"]
                
                self._log(f"🔧 Вызов: {tool_name}(\"{tool_arg}\")")
                
                # Выполняем
                observation = self.toolkit.execute(tool_name, tool_arg)
                self._log(f"👁️ Результат: {observation[:200]}")
                
                actions_log.append({
                    "iteration": iteration,
                    "tool": tool_name,
                    "arg": tool_arg,
                    "result": observation[:500]
                })
                
                # Добавляем в историю
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})
            
            else:
                # Нет ACTION — просим продолжить
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "Укажи ACTION с инструментом или ANSWER с финальным ответом."
                })
        
        self._log(f"\n❌ Превышен лимит итераций ({self.max_iterations})")
        
        return {
            "answer": "Не удалось найти ответ",
            "iterations": self.max_iterations,
            "actions": actions_log,
            "success": False
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Демонстрация ReAct агента"""
    
    print("\n" + "="*70)
    print("ReAct АГЕНТ (Reasoning + Acting)")
    print("="*70 + "\n")
    
    # Создаём инструменты
    toolkit = ToolKit()
    
    toolkit.register(Tool(
        name="calculator",
        description="Вычисляет математические выражения",
        func=safe_calculator,
        parameters={"expression": {"type": "string"}}
    ))
    
    toolkit.register(Tool(
        name="wikipedia",
        description="Ищет информацию в Wikipedia",
        func=wikipedia_search,
        parameters={"query": {"type": "string"}}
    ))
    
    toolkit.register(Tool(
        name="current_time",
        description="Возвращает текущую дату и время",
        func=get_current_time,
        parameters={"timezone": {"type": "string"}}
    ))
    
    # Создаём агента
    try:
        agent = ReActAgent(toolkit, max_iterations=5)
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Тестовые задачи
    tasks = [
        "Сколько будет 23 * 17 + 156?",
        "Кто такой Пушкин и в каком году он родился?",
        "Какой сейчас год?",
    ]
    
    for task in tasks:
        try:
            result = agent.run(task)
            print(f"\n📊 Статистика:")
            print(f"   Итераций: {result['iterations']}")
            print(f"   Действий: {len(result['actions'])}")
            print(f"   Успех: {'✅' if result['success'] else '❌'}")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        
        print("\n" + "-"*70)


def interactive():
    """Интерактивный режим"""
    
    print("\n" + "="*70)
    print("ReAct АГЕНТ - ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("(введите 'exit' для выхода)")
    print("="*70 + "\n")
    
    toolkit = ToolKit()
    toolkit.register(Tool("calculator", "Вычисления", safe_calculator, {"expression": {}}))
    toolkit.register(Tool("wikipedia", "Поиск Wikipedia", wikipedia_search, {"query": {}}))
    toolkit.register(Tool("current_time", "Текущее время", get_current_time, {"timezone": {}}))
    
    try:
        agent = ReActAgent(toolkit, max_iterations=5)
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    while True:
        try:
            query = input("\n👤 Вы: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            if not query:
                continue
            
            agent.run(query)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    
    print("👋 До свидания!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        interactive()
    else:
        main()

