"""
Безопасный калькулятор для LLM-агента.

Использует AST-парсинг вместо eval() для безопасности.
Поддерживает: +, -, *, /, **, sqrt, sin, cos, tan, log, abs, pi, e

Пример:
    >>> safe_calculator("2 + 2")
    4.0
    >>> safe_calculator("sqrt(16)")
    4.0
    >>> safe_calculator("pi * 2 ** 2")
    12.566370614359172
"""

import ast
import operator
import math
from dataclasses import dataclass, field
from typing import Callable, Any


# ═══════════════════════════════════════════════════════════════════════════════
# БАЗОВЫЕ КЛАССЫ
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Tool:
    """Базовый класс для инструмента агента"""
    name: str
    description: str
    func: Callable
    parameters: dict = field(default_factory=dict)
    
    def to_schema(self) -> dict:
        """Преобразует в формат для API"""
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
        """Выполняет инструмент"""
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            return f"Ошибка: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
# КАЛЬКУЛЯТОР
# ═══════════════════════════════════════════════════════════════════════════════

# Разрешённые операторы
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
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
    'asin': math.asin,
    'acos': math.acos,
    'atan': math.atan,
    'log': math.log,
    'log10': math.log10,
    'log2': math.log2,
    'exp': math.exp,
    'abs': abs,
    'round': round,
    'floor': math.floor,
    'ceil': math.ceil,
    'factorial': math.factorial,
    'gcd': math.gcd,
}

# Константы
CONSTANTS = {
    'pi': math.pi,
    'e': math.e,
    'tau': math.tau,
    'inf': math.inf,
}


def _eval_node(node) -> float:
    """
    Рекурсивно вычисляет AST-узел.
    
    Args:
        node: Узел AST-дерева
        
    Returns:
        Результат вычисления
        
    Raises:
        ValueError: При неподдерживаемой операции
    """
    # Число
    if isinstance(node, ast.Num):
        return node.n
    
    # Константа (Python 3.8+)
    elif isinstance(node, ast.Constant):
        return node.value
    
    # Имя (константа типа pi, e)
    elif isinstance(node, ast.Name):
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise ValueError(f"Неизвестная константа: {node.id}")
    
    # Бинарная операция (a + b, a * b)
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op = OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Неподдерживаемый оператор: {type(node.op).__name__}")
        return op(left, right)
    
    # Унарная операция (-a, +a)
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op = OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Неподдерживаемый оператор: {type(node.op).__name__}")
        return op(operand)
    
    # Вызов функции (sqrt(x), sin(x))
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Поддерживаются только простые вызовы функций")
        
        func_name = node.func.id
        if func_name not in FUNCTIONS:
            available = ", ".join(FUNCTIONS.keys())
            raise ValueError(f"Неизвестная функция: {func_name}. Доступны: {available}")
        
        args = [_eval_node(arg) for arg in node.args]
        return FUNCTIONS[func_name](*args)
    
    else:
        raise ValueError(f"Неподдерживаемый тип узла: {type(node).__name__}")


def safe_calculator(expression: str) -> float:
    """
    Безопасно вычисляет математическое выражение.
    
    Поддерживаемые операции:
    - Арифметика: +, -, *, /, //, %, **
    - Функции: sqrt, sin, cos, tan, log, exp, abs, round, floor, ceil
    - Константы: pi, e, tau, inf
    
    Args:
        expression: Математическое выражение как строка
        
    Returns:
        Результат вычисления как float
        
    Raises:
        ValueError: При ошибке парсинга или вычисления
        
    Examples:
        >>> safe_calculator("2 + 2")
        4.0
        >>> safe_calculator("sqrt(16)")
        4.0
        >>> safe_calculator("pi * 2 ** 2")
        12.566370614359172
        >>> safe_calculator("sin(pi / 2)")
        1.0
    """
    if not expression or not expression.strip():
        raise ValueError("Выражение не может быть пустым")
    
    # Очищаем выражение
    expression = expression.strip()
    
    try:
        # Парсим выражение в AST
        tree = ast.parse(expression, mode='eval')
        
        # Вычисляем результат
        result = _eval_node(tree.body)
        
        return float(result)
        
    except SyntaxError as e:
        raise ValueError(f"Синтаксическая ошибка в выражении '{expression}': {e}")
    except ZeroDivisionError:
        raise ValueError("Деление на ноль")
    except Exception as e:
        raise ValueError(f"Ошибка вычисления '{expression}': {str(e)}")


# Создаём инструмент для агента
calculator_tool = Tool(
    name="calculator",
    description=(
        "Выполняет математические вычисления. "
        "Поддерживает: +, -, *, /, **, sqrt, sin, cos, tan, log, abs, round, floor, ceil. "
        "Константы: pi, e. "
        "Используй для любых числовых расчётов."
    ),
    func=safe_calculator,
    parameters={
        "expression": {
            "type": "string",
            "description": (
                "Математическое выражение. "
                "Примеры: '2 + 2', 'sqrt(16)', 'pi * 2 ** 2', 'sin(pi / 2)'"
            )
        }
    }
)


# ═══════════════════════════════════════════════════════════════════════════════
# ТЕСТЫ
# ═══════════════════════════════════════════════════════════════════════════════

def test_calculator():
    """Тестирование калькулятора"""
    
    # Базовые операции
    assert safe_calculator("2 + 2") == 4.0
    assert safe_calculator("10 - 3") == 7.0
    assert safe_calculator("5 * 3") == 15.0
    assert safe_calculator("10 / 4") == 2.5
    assert safe_calculator("2 ** 8") == 256.0
    assert safe_calculator("10 // 3") == 3.0
    assert safe_calculator("10 % 3") == 1.0
    
    # Унарные операции
    assert safe_calculator("-5") == -5.0
    assert safe_calculator("--5") == 5.0
    
    # Функции
    assert safe_calculator("sqrt(16)") == 4.0
    assert abs(safe_calculator("sin(0)") - 0.0) < 0.0001
    assert abs(safe_calculator("cos(0)") - 1.0) < 0.0001
    assert safe_calculator("abs(-10)") == 10.0
    assert safe_calculator("round(3.7)") == 4.0
    assert safe_calculator("floor(3.7)") == 3.0
    assert safe_calculator("ceil(3.2)") == 4.0
    
    # Константы
    assert abs(safe_calculator("pi") - 3.14159) < 0.001
    assert abs(safe_calculator("e") - 2.71828) < 0.001
    
    # Сложные выражения
    assert safe_calculator("2 + 3 * 4") == 14.0
    assert safe_calculator("(2 + 3) * 4") == 20.0
    assert abs(safe_calculator("pi * 2 ** 2") - 12.566) < 0.001
    
    print("✅ Все тесты калькулятора пройдены!")


if __name__ == "__main__":
    test_calculator()
    
    # Интерактивный режим
    print("\n📱 Калькулятор (введите 'exit' для выхода)")
    while True:
        expr = input("\n> ")
        if expr.lower() == 'exit':
            break
        try:
            result = safe_calculator(expr)
            print(f"= {result}")
        except Exception as e:
            print(f"❌ {e}")

