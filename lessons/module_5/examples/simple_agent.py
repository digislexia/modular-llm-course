"""
Простой агент с инструментами.

Демонстрирует базовую интеграцию Function Calling с LLM.
Агент использует инструменты для ответа на вопросы.

Запуск:
    python simple_agent.py

Требования:
    - OPENROUTER_API_KEY в .env файле
    - pip install requests python-dotenv
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from typing import Callable
from dotenv import load_dotenv

# Добавляем путь к tools
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# БАЗОВЫЕ КЛАССЫ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Tool:
    """Инструмент для агента"""
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


class ToolKit:
    """Набор инструментов"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        self.tools[tool.name] = tool
        print(f"✅ Зарегистрирован: {tool.name}")
    
    def get_schemas(self) -> list:
        return [tool.to_schema() for tool in self.tools.values()]
    
    def execute(self, name: str, **kwargs) -> str:
        if name not in self.tools:
            return f"Инструмент '{name}' не найден"
        return self.tools[name].execute(**kwargs)
    
    def describe(self) -> str:
        return "\n".join([
            f"• {t.name}: {t.description}" 
            for t in self.tools.values()
        ])


# ═══════════════════════════════════════════════════════════════════════════════
# ИНСТРУМЕНТЫ
# ═══════════════════════════════════════════════════════════════════════════════

import math
import ast
import operator

def safe_calculator(expression: str) -> float:
    """Безопасный калькулятор"""
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
    """Поиск в Wikipedia"""
    url = f"https://ru.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("extract", "Нет описания")[:500]
        return f"Не найдено: {query}"
    except Exception as e:
        return f"Ошибка: {e}"


def get_current_time(timezone: str = "Europe/Moscow") -> str:
    """Текущее время"""
    from datetime import datetime
    now = datetime.now()
    days = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
    return f"{now.strftime('%d.%m.%Y')} ({days[now.weekday()]}), {now.strftime('%H:%M')}"


# Создаём инструменты
calculator_tool = Tool(
    name="calculator",
    description="Выполняет математические вычисления",
    func=safe_calculator,
    parameters={"expression": {"type": "string", "description": "Выражение, например '2 + 2'"}}
)

wikipedia_tool = Tool(
    name="wikipedia",
    description="Ищет информацию в Wikipedia",
    func=wikipedia_search,
    parameters={"query": {"type": "string", "description": "Поисковый запрос"}}
)

datetime_tool = Tool(
    name="current_time",
    description="Возвращает текущую дату и время",
    func=get_current_time,
    parameters={"timezone": {"type": "string", "description": "Часовой пояс"}}
)


# ═══════════════════════════════════════════════════════════════════════════════
# ПРОСТОЙ АГЕНТ
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleAgent:
    """
    Простой агент с инструментами.
    
    Использует Function Calling для выбора и вызова инструментов.
    """
    
    def __init__(self, toolkit: ToolKit, model: str = "openai/gpt-4-turbo-preview"):
        self.toolkit = toolkit
        self.model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError("Не найден OPENROUTER_API_KEY")
        
        self.system_prompt = f"""Ты — полезный ассистент с инструментами.

Доступные инструменты:
{toolkit.describe()}

Используй инструменты для точных вычислений и поиска информации.
Отвечай на русском языке.
"""
    
    def _call_api(self, messages: List[Dict], tools: List[Dict]) -> Dict:
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
            raise Exception(f"API Error: {response.status_code}")
        
        return response.json()
    
    def run(self, user_message: str, max_tool_calls: int = 3) -> str:
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
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        tool_calls_count = 0
        
        while tool_calls_count < max_tool_calls:
            response = self._call_api(messages, self.toolkit.get_schemas())
            message = response["choices"][0]["message"]
            
            # Проверяем вызовы инструментов
            if "tool_calls" not in message or not message["tool_calls"]:
                # Финальный ответ
                answer = message.get("content", "")
                print(f"\n🤖 Агент: {answer}")
                return answer
            
            # Обрабатываем вызовы
            tool_calls_count += 1
            messages.append(message)
            
            for tool_call in message["tool_calls"]:
                func_name = tool_call["function"]["name"]
                func_args = json.loads(tool_call["function"]["arguments"])
                
                print(f"\n🔧 Вызов: {func_name}({func_args})")
                
                # Получаем первый параметр
                first_param = list(self.toolkit.tools[func_name].parameters.keys())[0]
                arg_value = func_args.get(first_param, list(func_args.values())[0] if func_args else "")
                
                result = self.toolkit.execute(func_name, **{first_param: arg_value})
                print(f"   → Результат: {result[:100]}...")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result
                })
        
        return "Превышено количество вызовов инструментов"


# ═══════════════════════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Демонстрация простого агента"""
    
    print("\n" + "="*60)
    print("ПРОСТОЙ АГЕНТ С ИНСТРУМЕНТАМИ")
    print("="*60 + "\n")
    
    # Создаём набор инструментов
    toolkit = ToolKit()
    toolkit.register(calculator_tool)
    toolkit.register(wikipedia_tool)
    toolkit.register(datetime_tool)
    
    # Создаём агента
    try:
        agent = SimpleAgent(toolkit)
    except ValueError as e:
        print(f"❌ {e}")
        print("Создайте файл .env с OPENROUTER_API_KEY")
        return
    
    # Тестовые запросы
    queries = [
        "Сколько будет 234 * 567?",
        "Кто такой Альберт Эйнштейн?",
        "Какой сегодня день?",
    ]
    
    for query in queries:
        try:
            agent.run(query)
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        print()


def interactive():
    """Интерактивный режим"""
    
    print("\n" + "="*60)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("(введите 'exit' для выхода)")
    print("="*60 + "\n")
    
    toolkit = ToolKit()
    toolkit.register(calculator_tool)
    toolkit.register(wikipedia_tool)
    toolkit.register(datetime_tool)
    
    try:
        agent = SimpleAgent(toolkit)
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    while True:
        try:
            query = input("\n👤 Вы: ").strip()
            if query.lower() in ['exit', 'quit', 'выход']:
                print("👋 До свидания!")
                break
            if not query:
                continue
            
            agent.run(query)
            
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        interactive()
    else:
        main()

