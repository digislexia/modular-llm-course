"""
Research Agent: агент для исследовательских задач.

Расширенная версия ReAct агента, специализированная на:
- Сборе информации из нескольких источников
- Анализе и синтезе данных
- Формировании структурированных отчётов

Запуск:
    python research_agent.py

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
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from typing import Callable
from collections import defaultdict
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
            return f"Инструмент '{name}' не найден"
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
    """Поиск в Wikipedia с более подробным результатом"""
    url = f"https://ru.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    try:
        resp = requests.get(url, headers={"User-Agent": "ResearchAgent/1.0"}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            title = data.get("title", query)
            extract = data.get("extract", "")
            return f"[{title}]\n{extract[:800]}"
        
        # Fallback: поиск
        search_url = f"https://ru.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
        resp = requests.get(search_url, timeout=10)
        if resp.status_code == 200:
            results = resp.json().get("query", {}).get("search", [])
            if results:
                return wikipedia_search(results[0]["title"])
        
        return f"Не найдено: {query}"
    except Exception as e:
        return f"Ошибка: {e}"


def wikipedia_search_en(query: str) -> str:
    """Поиск в английской Wikipedia"""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    try:
        resp = requests.get(url, headers={"User-Agent": "ResearchAgent/1.0"}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return f"[{data.get('title', query)}]\n{data.get('extract', '')[:800]}"
        return f"Not found: {query}"
    except Exception as e:
        return f"Error: {e}"


def get_current_datetime(timezone: str = "") -> str:
    now = datetime.now()
    return f"{now.strftime('%d.%m.%Y %H:%M')} (год: {now.year})"


# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCH AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class ResearchAgent:
    """
    Исследовательский агент.
    
    Особенности:
    - Структурированный вывод
    - Множественные источники
    - Трекинг найденной информации
    """
    
    SYSTEM_PROMPT = """Ты — Research Agent, специализирующийся на исследованиях.

## Твоя задача

Проводить исследования: собирать информацию из разных источников,
анализировать и формировать структурированный ответ.

## Формат работы

THOUGHT: (что нужно исследовать дальше)
ACTION: tool_name("query")

Повторяй, пока не соберёшь достаточно информации.

THOUGHT: (анализ собранных данных)
ANSWER:
## Результаты исследования

### Краткий ответ
(1-2 предложения)

### Детали
(подробная информация по пунктам)

### Источники
(какие запросы были использованы)

## Доступные инструменты

{tools}

## Правила исследования

1. Собирай информацию из нескольких источников
2. Проверяй факты перед включением в ответ
3. Если информация противоречива — укажи это
4. Структурируй ответ
5. Указывай источники
"""
    
    def __init__(
        self,
        toolkit: ToolKit,
        model: str = "openai/gpt-4-turbo-preview",
        max_iterations: int = 8,
        verbose: bool = True
    ):
        self.toolkit = toolkit
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError("Не найден OPENROUTER_API_KEY")
        
        self.system_prompt = self.SYSTEM_PROMPT.format(tools=toolkit.describe())
        
        # Трекинг исследования
        self.research_log = {
            "queries": [],
            "findings": [],
            "sources": []
        }
    
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
                "max_tokens": 1500
            },
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code}")
        
        return response.json()["choices"][0]["message"]["content"]
    
    def _parse_action(self, response: str) -> Optional[Dict]:
        patterns = [
            r'ACTION:\s*(\w+)\s*\(\s*["\']([^"\']*)["\']?\s*\)',
            r'ACTION:\s*(\w+)\s*\(\s*(.+?)\s*\)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return {"tool": match.group(1), "arg": match.group(2).strip('"\'') }
        return None
    
    def _extract_answer(self, response: str) -> Optional[str]:
        if "ANSWER:" in response:
            return response.split("ANSWER:")[-1].strip()
        return None
    
    def research(self, topic: str) -> Dict:
        """
        Проводит исследование по теме.
        
        Args:
            topic: Тема исследования
            
        Returns:
            Результаты исследования
        """
        # Сброс лога
        self.research_log = {"queries": [], "findings": [], "sources": []}
        
        self._log(f"\n{'='*70}")
        self._log(f"🔬 ИССЛЕДОВАНИЕ: {topic}")
        self._log(f"{'='*70}\n")
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Проведи исследование: {topic}"}
        ]
        
        actions_log = []
        
        for iteration in range(1, self.max_iterations + 1):
            self._log(f"\n--- Итерация {iteration} ---\n")
            
            response = self._call_llm(messages)
            self._log(f"🤖 Агент:\n{response}\n")
            
            # Проверяем ANSWER
            answer = self._extract_answer(response)
            if answer:
                self._log(f"\n{'='*70}")
                self._log(f"📋 РЕЗУЛЬТАТЫ ИССЛЕДОВАНИЯ")
                self._log(f"{'='*70}")
                self._log(answer)
                
                return {
                    "topic": topic,
                    "answer": answer,
                    "iterations": iteration,
                    "actions": actions_log,
                    "research_log": self.research_log,
                    "success": True
                }
            
            # Парсим ACTION
            action = self._parse_action(response)
            
            if action:
                tool_name = action["tool"]
                tool_arg = action["arg"]
                
                self._log(f"🔍 Поиск: {tool_name}(\"{tool_arg}\")")
                
                # Выполняем
                observation = self.toolkit.execute(tool_name, tool_arg)
                self._log(f"📖 Найдено: {observation[:300]}...")
                
                # Логируем
                self.research_log["queries"].append(tool_arg)
                self.research_log["findings"].append(observation[:500])
                self.research_log["sources"].append(f"{tool_name}({tool_arg})")
                
                actions_log.append({
                    "iteration": iteration,
                    "tool": tool_name,
                    "query": tool_arg,
                    "result": observation[:500]
                })
                
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})
            
            else:
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "Продолжай исследование (ACTION) или дай ANSWER."
                })
        
        self._log(f"\n❌ Исследование не завершено за {self.max_iterations} итераций")
        
        return {
            "topic": topic,
            "answer": "Исследование не завершено",
            "iterations": self.max_iterations,
            "actions": actions_log,
            "research_log": self.research_log,
            "success": False
        }
    
    def compare(self, items: List[str], criteria: str = "") -> Dict:
        """
        Сравнивает несколько объектов.
        
        Args:
            items: Список объектов для сравнения
            criteria: Критерии сравнения (опционально)
            
        Returns:
            Результаты сравнения
        """
        topic = f"Сравни: {', '.join(items)}"
        if criteria:
            topic += f". Критерии: {criteria}"
        
        return self.research(topic)


# ═══════════════════════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Демонстрация Research Agent"""
    
    print("\n" + "="*70)
    print("RESEARCH AGENT")
    print("="*70 + "\n")
    
    # Создаём инструменты
    toolkit = ToolKit()
    
    toolkit.register(Tool(
        name="calculator",
        description="Математические вычисления",
        func=safe_calculator,
        parameters={"expression": {"type": "string"}}
    ))
    
    toolkit.register(Tool(
        name="wikipedia_ru",
        description="Поиск в русской Wikipedia",
        func=wikipedia_search,
        parameters={"query": {"type": "string"}}
    ))
    
    toolkit.register(Tool(
        name="wikipedia_en",
        description="Поиск в английской Wikipedia",
        func=wikipedia_search_en,
        parameters={"query": {"type": "string"}}
    ))
    
    toolkit.register(Tool(
        name="current_date",
        description="Текущая дата",
        func=get_current_datetime,
        parameters={"timezone": {"type": "string"}}
    ))
    
    # Создаём агента
    try:
        agent = ResearchAgent(toolkit, max_iterations=6)
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Исследовательские задачи
    topics = [
        "Кто основал компанию OpenAI и в каком году?",
        # "Сравни Python и JavaScript для машинного обучения",
    ]
    
    for topic in topics:
        try:
            result = agent.research(topic)
            
            print(f"\n📊 Статистика исследования:")
            print(f"   Итераций: {result['iterations']}")
            print(f"   Запросов: {len(result['research_log']['queries'])}")
            print(f"   Успех: {'✅' if result['success'] else '❌'}")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        
        print("\n" + "="*70)


def interactive():
    """Интерактивный режим"""
    
    print("\n" + "="*70)
    print("RESEARCH AGENT - ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("Команды: /compare item1, item2 - сравнить")
    print("         exit - выход")
    print("="*70 + "\n")
    
    toolkit = ToolKit()
    toolkit.register(Tool("calculator", "Вычисления", safe_calculator, {"expression": {}}))
    toolkit.register(Tool("wikipedia_ru", "Wikipedia RU", wikipedia_search, {"query": {}}))
    toolkit.register(Tool("wikipedia_en", "Wikipedia EN", wikipedia_search_en, {"query": {}}))
    toolkit.register(Tool("current_date", "Текущая дата", get_current_datetime, {"timezone": {}}))
    
    try:
        agent = ResearchAgent(toolkit, max_iterations=6)
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    while True:
        try:
            query = input("\n🔬 Тема: ").strip()
            
            if query.lower() in ['exit', 'quit']:
                break
            if not query:
                continue
            
            if query.startswith("/compare "):
                items = query[9:].split(",")
                items = [i.strip() for i in items]
                agent.compare(items)
            else:
                agent.research(query)
            
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

