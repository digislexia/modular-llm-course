"""
Инструменты для работы с датой и временем.

Включает:
- Получение текущей даты/времени
- Вычисление дат (через N дней, разница)
- Форматирование дат

Пример:
    >>> get_current_datetime()
    "6 января 2026 года, вторник, 14:30 (Europe/Moscow)"
"""

from datetime import datetime, timedelta
from typing import Optional
import re
from dataclasses import dataclass, field
from typing import Callable

try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False
    print("⚠️ pytz не установлен. Часовые пояса ограничены.")


@dataclass
class Tool:
    """Базовый класс для инструмента агента"""
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


# ═══════════════════════════════════════════════════════════════════════════════
# КОНСТАНТЫ
# ═══════════════════════════════════════════════════════════════════════════════

WEEKDAYS_RU = [
    "понедельник", "вторник", "среда", "четверг", 
    "пятница", "суббота", "воскресенье"
]

MONTHS_RU = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря"
]

MONTHS_RU_NOM = [
    "январь", "февраль", "март", "апрель", "май", "июнь",
    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"
]


# ═══════════════════════════════════════════════════════════════════════════════
# ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════════════════

def get_current_datetime(timezone: str = "Europe/Moscow") -> str:
    """
    Возвращает текущую дату и время в заданном часовом поясе.
    
    Args:
        timezone: Часовой пояс (по умолчанию Москва).
                 Примеры: "Europe/Moscow", "UTC", "US/Eastern"
        
    Returns:
        Форматированная строка с датой и временем
        
    Examples:
        >>> get_current_datetime()
        "6 января 2026 года, вторник, 14:30 (Europe/Moscow)"
        
        >>> get_current_datetime("UTC")
        "6 января 2026 года, вторник, 11:30 (UTC)"
    """
    try:
        if HAS_PYTZ:
            tz = pytz.timezone(timezone)
            now = datetime.now(tz)
        else:
            now = datetime.now()
            timezone = "локальное время"
        
        weekday = WEEKDAYS_RU[now.weekday()]
        month = MONTHS_RU[now.month - 1]
        
        return (
            f"{now.day} {month} {now.year} года, {weekday}, "
            f"{now.hour:02d}:{now.minute:02d} ({timezone})"
        )
    
    except Exception as e:
        return f"Ошибка получения времени: {str(e)}"


def calculate_date(operation: str) -> str:
    """
    Выполняет вычисления с датами.
    
    Поддерживаемые операции:
    - "через N дней" — дата через N дней от сегодня
    - "N дней назад" — дата N дней назад
    - "дней до YYYY-MM-DD" — количество дней до указанной даты
    - "дней с YYYY-MM-DD" — количество дней с указанной даты
    
    Args:
        operation: Операция с датой
        
    Returns:
        Результат вычисления
        
    Examples:
        >>> calculate_date("через 7 дней")
        "13.01.2026"
        
        >>> calculate_date("30 дней назад")
        "07.12.2025"
        
        >>> calculate_date("дней до 2026-12-31")
        "359 дней"
    """
    now = datetime.now()
    op = operation.lower().strip()
    
    try:
        # Через N дней
        match = re.search(r'через\s*(\d+)\s*дн', op)
        if match:
            days = int(match.group(1))
            future = now + timedelta(days=days)
            weekday = WEEKDAYS_RU[future.weekday()]
            month = MONTHS_RU[future.month - 1]
            return f"{future.day} {month} {future.year} ({weekday})"
        
        # N дней назад
        match = re.search(r'(\d+)\s*дн\w*\s*назад', op)
        if match:
            days = int(match.group(1))
            past = now - timedelta(days=days)
            weekday = WEEKDAYS_RU[past.weekday()]
            month = MONTHS_RU[past.month - 1]
            return f"{past.day} {month} {past.year} ({weekday})"
        
        # Дней до даты
        match = re.search(r'дней\s*до\s*(\d{4})-(\d{2})-(\d{2})', op)
        if match:
            target = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            diff = (target - now).days
            if diff < 0:
                return f"Дата уже прошла ({abs(diff)} дней назад)"
            return f"{diff} дней до {target.strftime('%d.%m.%Y')}"
        
        # Дней с даты
        match = re.search(r'дней\s*с\s*(\d{4})-(\d{2})-(\d{2})', op)
        if match:
            target = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            diff = (now - target).days
            if diff < 0:
                return f"Дата ещё не наступила (через {abs(diff)} дней)"
            return f"{diff} дней с {target.strftime('%d.%m.%Y')}"
        
        # Через N недель
        match = re.search(r'через\s*(\d+)\s*недел', op)
        if match:
            weeks = int(match.group(1))
            future = now + timedelta(weeks=weeks)
            weekday = WEEKDAYS_RU[future.weekday()]
            month = MONTHS_RU[future.month - 1]
            return f"{future.day} {month} {future.year} ({weekday})"
        
        # Через N месяцев (приблизительно)
        match = re.search(r'через\s*(\d+)\s*месяц', op)
        if match:
            months = int(match.group(1))
            future = now + timedelta(days=months * 30)
            weekday = WEEKDAYS_RU[future.weekday()]
            month = MONTHS_RU[future.month - 1]
            return f"Примерно {future.day} {month} {future.year}"
        
        return (
            "Не удалось распознать операцию. Примеры:\n"
            "• 'через 7 дней'\n"
            "• '30 дней назад'\n"
            "• 'дней до 2026-12-31'\n"
            "• 'дней с 2020-01-01'"
        )
        
    except Exception as e:
        return f"Ошибка: {str(e)}"


def format_date(date_str: str, output_format: str = "full") -> str:
    """
    Форматирует дату в читаемый вид.
    
    Args:
        date_str: Дата в формате YYYY-MM-DD или DD.MM.YYYY
        output_format: Формат вывода:
            - "full": "6 января 2026 года, понедельник"
            - "short": "06.01.2026"
            - "iso": "2026-01-06"
    
    Returns:
        Форматированная дата
    """
    # Парсим дату
    date = None
    
    for fmt in ["%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"]:
        try:
            date = datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue
    
    if date is None:
        return f"Не удалось распознать дату: {date_str}"
    
    if output_format == "short":
        return date.strftime("%d.%m.%Y")
    elif output_format == "iso":
        return date.strftime("%Y-%m-%d")
    else:  # full
        weekday = WEEKDAYS_RU[date.weekday()]
        month = MONTHS_RU[date.month - 1]
        return f"{date.day} {month} {date.year} года, {weekday}"


# ═══════════════════════════════════════════════════════════════════════════════
# ИНСТРУМЕНТЫ
# ═══════════════════════════════════════════════════════════════════════════════

datetime_tool = Tool(
    name="current_datetime",
    description=(
        "Возвращает текущую дату и время. "
        "Используй, когда нужно узнать сегодняшнюю дату, день недели или время."
    ),
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
    description=(
        "Вычисляет даты: через N дней, N дней назад, дней до/с определённой даты. "
        "Используй для расчёта сроков, возраста событий, планирования."
    ),
    func=calculate_date,
    parameters={
        "operation": {
            "type": "string",
            "description": (
                "Операция с датой. Примеры: "
                "'через 7 дней', '30 дней назад', 'дней до 2026-12-31'"
            )
        }
    }
)


# ═══════════════════════════════════════════════════════════════════════════════
# ТЕСТЫ
# ═══════════════════════════════════════════════════════════════════════════════

def test_datetime():
    """Тестирование функций даты/времени"""
    
    # Текущее время
    result = get_current_datetime()
    assert "года" in result
    assert ":" in result
    print(f"✅ Текущее время: {result}")
    
    # UTC
    result = get_current_datetime("UTC")
    assert "UTC" in result
    print(f"✅ UTC: {result}")
    
    # Через N дней
    result = calculate_date("через 7 дней")
    assert "Ошибка" not in result
    print(f"✅ Через 7 дней: {result}")
    
    # N дней назад
    result = calculate_date("30 дней назад")
    assert "Ошибка" not in result
    print(f"✅ 30 дней назад: {result}")
    
    # Дней до даты
    result = calculate_date("дней до 2030-12-31")
    assert "дней" in result
    print(f"✅ Дней до 2030-12-31: {result}")
    
    # Форматирование
    result = format_date("2026-01-06", "full")
    assert "января" in result
    print(f"✅ Форматирование: {result}")
    
    print("\n✅ Все тесты даты/времени пройдены!")


if __name__ == "__main__":
    test_datetime()

