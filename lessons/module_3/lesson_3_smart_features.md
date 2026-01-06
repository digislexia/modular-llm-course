# Урок 3: Добавление интеллектуальных функций

## Введение

В предыдущих уроках мы создали работающее ядро SchoolBot с роутингом моделей и управлением контекстом. Теперь пришло время добавить "интеллект" — функции, которые сделают нашего помощника по-настоящему умным:

- **LLM-as-a-judge** — проверка качества объяснений
- **Генератор тестов** — задания в формате ОГЭ/ЕГЭ
- **Адаптивная сложность** — подстройка под уровень ученика
- **Обработка ошибок** — graceful degradation

## Цели урока

После завершения урока вы сможете:

- ✅ Интегрировать LLM-as-a-judge для самопроверки качества
- ✅ Создать генератор тестовых заданий с проверкой ответов
- ✅ Реализовать адаптивную систему сложности
- ✅ Добавить надёжную обработку ошибок и fallback

## Ключевые термины

- **LLM-as-a-judge** — использование LLM для оценки выходных данных другой (или той же) LLM
- **Адаптивное обучение** — подстройка материала под уровень ученика
- **Graceful degradation** — сохранение работоспособности при частичных сбоях
- **Валидация ответов** — проверка корректности данных перед использованием

## 1. LLM-as-a-judge: Проверка качества объяснений

### Напоминание из Модуля 2

В уроке 3 модуля 2 мы изучили концепцию LLM-as-a-judge — когда одна модель оценивает результаты другой. Это особенно важно для образовательного контента:

- Объяснение должно быть **понятным** для школьника
- Материал должен быть **полным**, но не перегруженным
- Примеры должны быть **релевантными** для возраста
- Уровень сложности должен **соответствовать** классу

### Реализация QualityChecker (`quality.py`)

```python
"""
Система проверки качества объяснений для SchoolBot.
Использует LLM-as-a-judge для оценки и улучшения ответов.
"""

import json
import re
from typing import Dict, Optional, Tuple
from router import ModelRouter


class QualityChecker:
    """
    Проверяет качество образовательного контента с помощью LLM-as-a-judge.
    """
    
    def __init__(self, router: ModelRouter, min_score: float = 7.0, max_improvements: int = 2):
        """
        Args:
            router: Роутер для отправки запросов
            min_score: Минимальная приемлемая оценка (1-10)
            max_improvements: Максимум попыток улучшения
        """
        self.router = router
        self.min_score = min_score
        self.max_improvements = max_improvements
        
        # Статистика
        self.stats = {
            "checks": 0,
            "improvements": 0,
            "avg_score": 0.0
        }
    
    def check_explanation(self, 
                          topic: str, 
                          explanation: str, 
                          grade: int,
                          subject: str = "общий") -> Dict:
        """
        Оценивает качество объяснения для школьника.
        
        Args:
            topic: Тема объяснения
            explanation: Текст объяснения
            grade: Класс ученика (9, 10, 11)
            subject: Предмет
            
        Returns:
            Dict с оценками и рекомендациями
        """
        self.stats["checks"] += 1
        
        check_prompt = f"""Оцени качество следующего объяснения для школьника {grade} класса.

Предмет: {subject}
Тема: {topic}

Объяснение:
---
{explanation}
---

Оцени по критериям (каждый от 1 до 10):

1. ПОНЯТНОСТЬ: Насколько просто и ясно изложен материал? 
   (10 = идеально понятно школьнику, 1 = слишком сложно)

2. ПОЛНОТА: Все ли важные аспекты темы раскрыты?
   (10 = полностью, 1 = поверхностно)

3. ПРИМЕРЫ: Есть ли понятные примеры из жизни?
   (10 = отличные примеры, 1 = нет примеров)

4. СТРУКТУРА: Логично ли организован материал?
   (10 = идеальная структура, 1 = хаотично)

5. СООТВЕТСТВИЕ УРОВНЮ: Подходит ли для ученика {grade} класса?
   (10 = идеально подходит, 1 = не соответствует)

ВАЖНО: Ответь ТОЛЬКО в формате JSON без дополнительного текста:
{{
    "понятность": <число>,
    "полнота": <число>,
    "примеры": <число>,
    "структура": <число>,
    "соответствие_уровню": <число>,
    "общая_оценка": <среднее от всех оценок>,
    "сильные_стороны": ["сторона 1", "сторона 2"],
    "рекомендации": ["рекомендация 1", "рекомендация 2"]
}}"""

        messages = [
            {"role": "system", "content": "Ты — эксперт по оценке образовательного контента для школьников. Отвечай строго в формате JSON."},
            {"role": "user", "content": check_prompt}
        ]
        
        result = self.router.send_request(
            messages=messages,
            task_type="quality_check",
            complexity="easy",
            max_tokens=500,
            temperature=0.3  # Низкая для стабильности
        )
        
        if not result["success"]:
            return {
                "success": False,
                "error": result["error"],
                "общая_оценка": 0
            }
        
        # Парсим JSON из ответа
        try:
            # Извлекаем JSON из ответа (может быть обёрнут в текст)
            content = result["content"]
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                quality_data = json.loads(json_match.group())
            else:
                raise ValueError("JSON не найден в ответе")
            
            quality_data["success"] = True
            
            # Обновляем статистику
            score = quality_data.get("общая_оценка", 0)
            self._update_avg_score(score)
            
            return quality_data
            
        except (json.JSONDecodeError, ValueError) as e:
            return {
                "success": False,
                "error": f"Ошибка парсинга оценки: {e}",
                "общая_оценка": 0,
                "raw_response": result["content"]
            }
    
    def improve_explanation(self,
                            topic: str,
                            explanation: str,
                            recommendations: list,
                            grade: int) -> str:
        """
        Улучшает объяснение на основе рекомендаций.
        
        Args:
            topic: Тема
            explanation: Исходное объяснение
            recommendations: Список рекомендаций по улучшению
            grade: Класс ученика
            
        Returns:
            Улучшенное объяснение
        """
        self.stats["improvements"] += 1
        
        improve_prompt = f"""Улучши следующее объяснение темы "{topic}" для ученика {grade} класса.

Исходное объяснение:
---
{explanation}
---

Рекомендации по улучшению:
{chr(10).join(f"• {r}" for r in recommendations)}

Требования:
1. Сохрани всю важную информацию из исходного объяснения
2. Учти ВСЕ рекомендации по улучшению  
3. Сделай объяснение понятным для школьника {grade} класса
4. Добавь примеры из повседневной жизни, если их не хватает
5. Структурируй материал с помощью списков и выделений

Напиши ТОЛЬКО улучшенное объяснение, без комментариев:"""

        messages = [
            {"role": "system", "content": f"Ты — опытный учитель, который умеет объяснять сложные темы простым языком для школьников {grade} класса."},
            {"role": "user", "content": improve_prompt}
        ]
        
        result = self.router.send_request(
            messages=messages,
            task_type="explain_simple",
            complexity="medium",
            max_tokens=2000,
            temperature=0.7
        )
        
        if result["success"]:
            return result["content"]
        else:
            return explanation  # Возвращаем исходное при ошибке
    
    def ensure_quality(self,
                       topic: str,
                       explanation: str,
                       grade: int,
                       subject: str = "общий") -> Tuple[str, Dict]:
        """
        Проверяет и при необходимости улучшает объяснение до приемлемого качества.
        
        Args:
            topic: Тема объяснения
            explanation: Исходное объяснение
            grade: Класс ученика
            subject: Предмет
            
        Returns:
            Tuple[улучшенное_объяснение, данные_качества]
        """
        current_explanation = explanation
        
        for attempt in range(self.max_improvements + 1):
            # Проверяем качество
            quality = self.check_explanation(topic, current_explanation, grade, subject)
            
            if not quality.get("success"):
                # При ошибке проверки возвращаем как есть
                return current_explanation, quality
            
            score = quality.get("общая_оценка", 0)
            
            # Если качество достаточное — возвращаем
            if score >= self.min_score:
                quality["improved"] = attempt > 0
                quality["improvement_attempts"] = attempt
                return current_explanation, quality
            
            # Если это последняя попытка — возвращаем что есть
            if attempt >= self.max_improvements:
                quality["improved"] = attempt > 0
                quality["improvement_attempts"] = attempt
                quality["note"] = "Достигнут лимит попыток улучшения"
                return current_explanation, quality
            
            # Пробуем улучшить
            recommendations = quality.get("рекомендации", [])
            if recommendations:
                print(f"   📝 Улучшаю объяснение (попытка {attempt + 1})...")
                current_explanation = self.improve_explanation(
                    topic, current_explanation, recommendations, grade
                )
        
        return current_explanation, quality
    
    def _update_avg_score(self, new_score: float):
        """Обновляет среднюю оценку"""
        n = self.stats["checks"]
        old_avg = self.stats["avg_score"]
        self.stats["avg_score"] = old_avg + (new_score - old_avg) / n
    
    def get_stats(self) -> Dict:
        """Возвращает статистику проверок"""
        return self.stats.copy()


# Тестирование
if __name__ == "__main__":
    print("🧪 Тестирование QualityChecker")
    print("="*50)
    
    router = ModelRouter()
    checker = QualityChecker(router, min_score=7.0)
    
    # Тестовое объяснение (намеренно неидеальное)
    test_explanation = """
    Производная функции - это предел отношения приращения функции к приращению 
    аргумента при стремлении последнего к нулю. Формально: f'(x) = lim(Δx→0) [f(x+Δx) - f(x)] / Δx.
    Производная показывает скорость изменения функции в данной точке.
    """
    
    print("\n📝 Проверяем объяснение производной...")
    quality = checker.check_explanation(
        topic="производная",
        explanation=test_explanation,
        grade=11,
        subject="математика"
    )
    
    if quality.get("success"):
        print(f"\n📊 Результаты оценки:")
        print(f"   Общая оценка: {quality.get('общая_оценка', 'N/A')}/10")
        print(f"   Понятность: {quality.get('понятность', 'N/A')}/10")
        print(f"   Примеры: {quality.get('примеры', 'N/A')}/10")
        print(f"\n   💪 Сильные стороны: {quality.get('сильные_стороны', [])}")
        print(f"   📝 Рекомендации: {quality.get('рекомендации', [])}")
    else:
        print(f"❌ Ошибка: {quality.get('error')}")
    
    print("\n" + "="*50)
    print("🔄 Тестируем автоулучшение...")
    
    improved, final_quality = checker.ensure_quality(
        topic="производная",
        explanation=test_explanation,
        grade=11,
        subject="математика"
    )
    
    print(f"\n📊 Финальная оценка: {final_quality.get('общая_оценка', 'N/A')}/10")
    print(f"   Улучшено: {final_quality.get('improved', False)}")
    print(f"   Попыток: {final_quality.get('improvement_attempts', 0)}")
```

## 2. Генератор тестовых заданий

### Особенности заданий ОГЭ и ЕГЭ

Для генерации качественных заданий важно понимать формат экзаменов:

**ОГЭ (9 класс):**
- Базовый уровень сложности
- Задания с кратким ответом
- Задания с выбором ответа
- Практико-ориентированные задачи

**ЕГЭ (10-11 класс):**
- Базовый и профильный уровень
- Задания с кратким ответом
- Задания с развёрнутым ответом
- Задачи повышенной сложности

### Реализация ExamQuizGenerator (`exam_quiz.py`)

```python
"""
Генератор тестовых заданий в формате ОГЭ/ЕГЭ для SchoolBot.
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from router import ModelRouter


@dataclass
class Question:
    """Структура тестового задания"""
    number: int
    text: str
    question_type: str  # "short_answer", "multiple_choice", "detailed"
    options: Optional[List[str]] = None
    correct_answer: str = ""
    explanation: str = ""
    difficulty: str = "medium"
    topic: str = ""


class ExamQuizGenerator:
    """
    Генератор заданий для подготовки к ОГЭ и ЕГЭ.
    """
    
    def __init__(self, router: ModelRouter):
        self.router = router
        
        # Типы заданий по экзаменам
        self.exam_types = {
            "oge": {
                "name": "ОГЭ",
                "grade": 9,
                "types": ["short_answer", "multiple_choice"],
                "difficulty_range": ["easy", "medium"]
            },
            "ege": {
                "name": "ЕГЭ",
                "grade": [10, 11],
                "types": ["short_answer", "multiple_choice", "detailed"],
                "difficulty_range": ["medium", "hard"]
            }
        }
        
        # Статистика
        self.stats = {
            "questions_generated": 0,
            "answers_checked": 0,
            "correct_answers": 0
        }
    
    def generate_questions(self,
                           topic: str,
                           subject: str,
                           exam_type: str = "ege",
                           num_questions: int = 3,
                           difficulty: str = "medium") -> List[Question]:
        """
        Генерирует тестовые задания по теме.
        
        Args:
            topic: Тема заданий
            subject: Предмет
            exam_type: Тип экзамена ('oge' или 'ege')
            num_questions: Количество заданий
            difficulty: Сложность ('easy', 'medium', 'hard')
            
        Returns:
            Список заданий
        """
        exam_info = self.exam_types.get(exam_type, self.exam_types["ege"])
        
        prompt = f"""Создай {num_questions} тестовых заданий по теме "{topic}" для экзамена {exam_info['name']}.

Предмет: {subject}
Уровень сложности: {difficulty}

Требования к заданиям:
1. Задания должны соответствовать формату реального экзамена {exam_info['name']}
2. Включи разные типы заданий:
   - С кратким ответом (число, слово, последовательность)
   - С выбором варианта (4 варианта, один правильный)
   {"- С развёрнутым ответом (для сложных заданий)" if exam_type == "ege" else ""}
3. Сложность должна соответствовать уровню: {difficulty}
4. Каждое задание должно проверять понимание темы

Формат ответа — СТРОГО JSON массив:
[
  {{
    "number": 1,
    "type": "short_answer" или "multiple_choice" или "detailed",
    "text": "Текст задания",
    "options": ["A) вариант", "B) вариант", "C) вариант", "D) вариант"] или null,
    "correct_answer": "правильный ответ",
    "explanation": "почему этот ответ правильный",
    "difficulty": "{difficulty}"
  }},
  ...
]

ВАЖНО: Ответь ТОЛЬКО JSON массивом, без дополнительного текста!"""

        messages = [
            {"role": "system", "content": f"Ты — эксперт по составлению заданий для {exam_info['name']} по предмету {subject}. Отвечай строго в формате JSON."},
            {"role": "user", "content": prompt}
        ]
        
        result = self.router.send_request(
            messages=messages,
            task_type="generate_exam",
            complexity=difficulty,
            max_tokens=2500,
            temperature=0.8  # Выше для разнообразия
        )
        
        if not result["success"]:
            return []
        
        # Парсим JSON
        try:
            content = result["content"]
            # Извлекаем JSON массив
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                questions_data = json.loads(json_match.group())
            else:
                raise ValueError("JSON массив не найден")
            
            questions = []
            for q in questions_data:
                question = Question(
                    number=q.get("number", len(questions) + 1),
                    text=q.get("text", ""),
                    question_type=q.get("type", "short_answer"),
                    options=q.get("options"),
                    correct_answer=q.get("correct_answer", ""),
                    explanation=q.get("explanation", ""),
                    difficulty=q.get("difficulty", difficulty),
                    topic=topic
                )
                questions.append(question)
            
            self.stats["questions_generated"] += len(questions)
            return questions
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️ Ошибка парсинга заданий: {e}")
            return []
    
    def format_questions_for_display(self, questions: List[Question]) -> str:
        """
        Форматирует задания для отображения ученику (без ответов).
        """
        if not questions:
            return "❌ Не удалось сгенерировать задания"
        
        output = []
        for q in questions:
            output.append(f"\n{'═'*50}")
            output.append(f"📝 ЗАДАНИЕ {q.number}")
            if q.difficulty == "hard":
                output.append("   [Повышенной сложности]")
            output.append(f"{'─'*50}")
            output.append(q.text)
            
            if q.options:
                output.append("")
                for option in q.options:
                    output.append(f"   {option}")
            
            output.append("")
            
            if q.question_type == "short_answer":
                output.append("💡 Введите краткий ответ (число, слово или последовательность)")
            elif q.question_type == "multiple_choice":
                output.append("💡 Введите букву правильного ответа (A, B, C или D)")
            else:
                output.append("💡 Напишите развёрнутый ответ")
        
        output.append(f"\n{'═'*50}")
        return "\n".join(output)
    
    def check_answer(self,
                     question: Question,
                     student_answer: str) -> Dict:
        """
        Проверяет ответ ученика на задание.
        
        Args:
            question: Задание
            student_answer: Ответ ученика
            
        Returns:
            Результат проверки
        """
        self.stats["answers_checked"] += 1
        
        # Для заданий с кратким ответом и выбором — сравниваем напрямую
        if question.question_type in ["short_answer", "multiple_choice"]:
            # Нормализуем ответы для сравнения
            correct = self._normalize_answer(question.correct_answer)
            student = self._normalize_answer(student_answer)
            
            is_correct = correct == student
            
            if is_correct:
                self.stats["correct_answers"] += 1
            
            return {
                "is_correct": is_correct,
                "correct_answer": question.correct_answer,
                "student_answer": student_answer,
                "explanation": question.explanation if not is_correct else "Правильно! 🎉",
                "feedback": self._generate_feedback(is_correct, question)
            }
        
        # Для развёрнутых ответов используем LLM
        return self._check_detailed_answer(question, student_answer)
    
    def _normalize_answer(self, answer: str) -> str:
        """Нормализует ответ для сравнения"""
        # Убираем пробелы, приводим к нижнему регистру
        normalized = answer.strip().lower()
        # Убираем точки, запятые в конце
        normalized = re.sub(r'[.,;:!?]+$', '', normalized)
        # Для букв (A, B, C, D) берём только первый символ
        if len(normalized) == 1 and normalized in 'abcdавсд':
            # Преобразуем русские буквы в латинские
            mapping = {'а': 'a', 'в': 'b', 'с': 'c', 'д': 'd'}
            normalized = mapping.get(normalized, normalized)
        return normalized
    
    def _generate_feedback(self, is_correct: bool, question: Question) -> str:
        """Генерирует обратную связь для ученика"""
        if is_correct:
            return "✅ Отлично! Ты правильно решил это задание."
        else:
            return f"""❌ К сожалению, ответ неверный.

📖 Правильный ответ: {question.correct_answer}

💡 Объяснение: {question.explanation}

📝 Совет: Внимательно перечитай условие задания и попробуй разобрать решение."""
    
    def _check_detailed_answer(self, question: Question, student_answer: str) -> Dict:
        """Проверяет развёрнутый ответ с помощью LLM"""
        
        prompt = f"""Проверь развёрнутый ответ ученика на задание.

ЗАДАНИЕ:
{question.text}

ЭТАЛОННЫЙ ОТВЕТ (или ключевые моменты):
{question.correct_answer}

ОТВЕТ УЧЕНИКА:
{student_answer}

Оцени ответ по критериям:
1. Полнота ответа (все ли ключевые моменты раскрыты)
2. Правильность фактов и рассуждений
3. Логика изложения

Ответь в формате JSON:
{{
    "score": <число от 0 до 10>,
    "is_correct": <true если score >= 7, иначе false>,
    "strengths": ["что хорошо в ответе"],
    "weaknesses": ["что нужно улучшить"],
    "feedback": "общий комментарий для ученика"
}}"""

        messages = [
            {"role": "system", "content": "Ты — учитель, проверяющий развёрнутые ответы учеников. Будь объективным, но доброжелательным."},
            {"role": "user", "content": prompt}
        ]
        
        result = self.router.send_request(
            messages=messages,
            task_type="check_answer",
            complexity="medium",
            max_tokens=500,
            temperature=0.3
        )
        
        if not result["success"]:
            return {
                "is_correct": None,
                "error": result["error"],
                "feedback": "Не удалось проверить ответ автоматически"
            }
        
        try:
            content = result["content"]
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                check_result = json.loads(json_match.group())
                check_result["correct_answer"] = question.correct_answer
                check_result["student_answer"] = student_answer
                
                if check_result.get("is_correct"):
                    self.stats["correct_answers"] += 1
                
                return check_result
            else:
                raise ValueError("JSON не найден")
                
        except (json.JSONDecodeError, ValueError):
            return {
                "is_correct": None,
                "feedback": "Не удалось разобрать результат проверки",
                "raw_response": result["content"]
            }
    
    def run_quiz_session(self, questions: List[Question]) -> Dict:
        """
        Проводит интерактивную сессию тестирования.
        
        Returns:
            Результаты сессии
        """
        if not questions:
            return {"error": "Нет заданий для тестирования"}
        
        results = {
            "total": len(questions),
            "correct": 0,
            "wrong": 0,
            "answers": []
        }
        
        print("\n" + "═"*50)
        print("📝 НАЧИНАЕМ ТЕСТИРОВАНИЕ")
        print("═"*50)
        print(f"Всего заданий: {len(questions)}")
        print("Введите 'пропустить' чтобы пропустить задание")
        print("Введите 'стоп' чтобы завершить тест досрочно")
        
        for q in questions:
            print(f"\n{'─'*50}")
            print(f"📌 Задание {q.number} из {len(questions)}")
            print(f"{'─'*50}")
            print(q.text)
            
            if q.options:
                print()
                for option in q.options:
                    print(f"   {option}")
            
            # Получаем ответ
            print()
            answer = input("✏️ Ваш ответ: ").strip()
            
            if answer.lower() == "стоп":
                print("\n⏹️ Тестирование прервано")
                break
            
            if answer.lower() == "пропустить":
                results["wrong"] += 1
                results["answers"].append({
                    "question": q.number,
                    "skipped": True
                })
                continue
            
            # Проверяем ответ
            check_result = self.check_answer(q, answer)
            
            if check_result.get("is_correct"):
                results["correct"] += 1
                print("\n✅ Правильно!")
            else:
                results["wrong"] += 1
                print(f"\n{check_result.get('feedback', 'Неверно')}")
            
            results["answers"].append({
                "question": q.number,
                "student_answer": answer,
                "is_correct": check_result.get("is_correct"),
                "correct_answer": check_result.get("correct_answer")
            })
        
        # Итоги
        print("\n" + "═"*50)
        print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
        print("═"*50)
        print(f"Правильных ответов: {results['correct']} из {results['total']}")
        
        percentage = (results['correct'] / results['total']) * 100 if results['total'] > 0 else 0
        print(f"Процент выполнения: {percentage:.1f}%")
        
        if percentage >= 80:
            print("🌟 Отличный результат!")
        elif percentage >= 60:
            print("👍 Хороший результат, но есть над чем поработать")
        else:
            print("📚 Рекомендую повторить тему и попробовать ещё раз")
        
        return results
    
    def get_stats(self) -> Dict:
        """Возвращает статистику"""
        return self.stats.copy()


# Тестирование
if __name__ == "__main__":
    print("🧪 Тестирование ExamQuizGenerator")
    print("="*50)
    
    router = ModelRouter()
    generator = ExamQuizGenerator(router)
    
    print("\n📝 Генерируем задания по математике...")
    questions = generator.generate_questions(
        topic="квадратные уравнения",
        subject="математика",
        exam_type="oge",
        num_questions=3,
        difficulty="medium"
    )
    
    if questions:
        print(f"\n✅ Сгенерировано {len(questions)} заданий")
        print(generator.format_questions_for_display(questions))
        
        # Тестируем проверку ответа
        print("\n" + "="*50)
        print("🔍 Тестируем проверку ответа...")
        
        test_answer = "5"  # Пробный ответ
        result = generator.check_answer(questions[0], test_answer)
        print(f"Результат: {result}")
    else:
        print("❌ Не удалось сгенерировать задания")
```

## 3. Адаптивная система сложности

Добавим интеллектуальную адаптацию сложности на основе успехов ученика:

```python
"""
Система адаптивной сложности для SchoolBot.
Добавьте этот код в assistant.py или создайте отдельный файл adaptive.py
"""

from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class StudentProgress:
    """Отслеживает прогресс ученика по теме"""
    topic: str
    correct_answers: int = 0
    total_answers: int = 0
    current_difficulty: str = "medium"
    difficulty_history: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Процент правильных ответов"""
        if self.total_answers == 0:
            return 0.0
        return self.correct_answers / self.total_answers
    
    def record_answer(self, is_correct: bool):
        """Записывает результат ответа"""
        self.total_answers += 1
        if is_correct:
            self.correct_answers += 1


class AdaptiveDifficultyManager:
    """
    Управляет адаптивной сложностью на основе успехов ученика.
    """
    
    def __init__(self):
        # Прогресс по темам
        self.progress: Dict[str, StudentProgress] = {}
        
        # Пороги для изменения сложности
        self.thresholds = {
            "increase": 0.8,  # 80%+ правильных → увеличить сложность
            "decrease": 0.4,  # 40%- правильных → уменьшить сложность
            "min_answers": 3  # Минимум ответов для изменения
        }
        
        # Порядок сложностей
        self.difficulty_order = ["easy", "medium", "hard"]
    
    def get_progress(self, topic: str) -> StudentProgress:
        """Получает или создаёт прогресс по теме"""
        if topic not in self.progress:
            self.progress[topic] = StudentProgress(topic=topic)
        return self.progress[topic]
    
    def record_answer(self, topic: str, is_correct: bool) -> str:
        """
        Записывает ответ и возвращает рекомендуемую сложность.
        
        Args:
            topic: Тема
            is_correct: Правильность ответа
            
        Returns:
            Рекомендуемая сложность
        """
        progress = self.get_progress(topic)
        progress.record_answer(is_correct)
        
        # Проверяем, нужно ли менять сложность
        if progress.total_answers >= self.thresholds["min_answers"]:
            new_difficulty = self._calculate_difficulty(progress)
            
            if new_difficulty != progress.current_difficulty:
                progress.difficulty_history.append(progress.current_difficulty)
                progress.current_difficulty = new_difficulty
                self._notify_difficulty_change(topic, new_difficulty, progress.success_rate)
        
        return progress.current_difficulty
    
    def _calculate_difficulty(self, progress: StudentProgress) -> str:
        """Рассчитывает рекомендуемую сложность"""
        rate = progress.success_rate
        current_idx = self.difficulty_order.index(progress.current_difficulty)
        
        if rate >= self.thresholds["increase"]:
            # Увеличиваем сложность
            new_idx = min(current_idx + 1, len(self.difficulty_order) - 1)
        elif rate <= self.thresholds["decrease"]:
            # Уменьшаем сложность
            new_idx = max(current_idx - 1, 0)
        else:
            # Оставляем как есть
            new_idx = current_idx
        
        return self.difficulty_order[new_idx]
    
    def _notify_difficulty_change(self, topic: str, new_difficulty: str, rate: float):
        """Уведомляет об изменении сложности"""
        direction = "повышена" if new_difficulty in ["medium", "hard"] else "понижена"
        emoji = "📈" if direction == "повышена" else "📉"
        
        print(f"\n{emoji} Сложность по теме '{topic}' {direction} до '{new_difficulty}'")
        print(f"   (на основе {rate*100:.0f}% правильных ответов)")
    
    def get_recommendation(self, topic: str) -> Dict:
        """Возвращает рекомендации для темы"""
        progress = self.get_progress(topic)
        
        return {
            "topic": topic,
            "current_difficulty": progress.current_difficulty,
            "success_rate": progress.success_rate,
            "total_answers": progress.total_answers,
            "recommendation": self._get_text_recommendation(progress)
        }
    
    def _get_text_recommendation(self, progress: StudentProgress) -> str:
        """Формирует текстовую рекомендацию"""
        if progress.total_answers < self.thresholds["min_answers"]:
            return "Продолжай решать задания для более точной оценки уровня"
        
        rate = progress.success_rate
        if rate >= 0.8:
            return "Отличные результаты! Можешь переходить к более сложным заданиям"
        elif rate >= 0.6:
            return "Хорошо! Продолжай практиковаться на текущем уровне"
        elif rate >= 0.4:
            return "Есть над чем поработать. Попробуй перечитать теорию"
        else:
            return "Рекомендую вернуться к базовым понятиям темы"
    
    def get_overall_stats(self) -> Dict:
        """Возвращает общую статистику по всем темам"""
        total_correct = sum(p.correct_answers for p in self.progress.values())
        total_answers = sum(p.total_answers for p in self.progress.values())
        
        return {
            "topics_studied": len(self.progress),
            "total_answers": total_answers,
            "total_correct": total_correct,
            "overall_rate": total_correct / total_answers if total_answers > 0 else 0,
            "topics": {
                topic: {
                    "difficulty": p.current_difficulty,
                    "rate": p.success_rate,
                    "answers": p.total_answers
                }
                for topic, p in self.progress.items()
            }
        }
```

## 4. Интеграция в SchoolAssistant

Обновим класс `SchoolAssistant` для использования новых компонентов:

```python
# Добавьте в начало файла assistant.py:
from quality import QualityChecker
from exam_quiz import ExamQuizGenerator, Question
# from adaptive import AdaptiveDifficultyManager  # Если создали отдельный файл

# В методе __init__ добавьте:
def __init__(self):
    # ... существующий код ...
    
    # Инициализация новых компонентов
    self.quality_checker = QualityChecker(
        router=self.router,
        min_score=QUALITY_CONFIG["min_score"],
        max_improvements=QUALITY_CONFIG["max_improvements"]
    )
    print("   ✅ Проверка качества")
    
    self.exam_generator = ExamQuizGenerator(router=self.router)
    print("   ✅ Генератор тестов")
    
    self.adaptive = AdaptiveDifficultyManager()
    print("   ✅ Адаптивная сложность")
    
    # Текущие задания для проверки ответов
    self.current_questions: List[Question] = []


# Обновите метод explain_topic для проверки качества:
def explain_topic(self, topic: str, check_quality: bool = True) -> str:
    """
    Объясняет тему с опциональной проверкой качества.
    """
    if not self.router:
        return "❌ Ошибка: роутер моделей не инициализирован"
    
    # Получаем рекомендуемую сложность из адаптивной системы
    recommendation = self.adaptive.get_recommendation(topic)
    adaptive_difficulty = recommendation["current_difficulty"]
    
    # Используем адаптивную сложность, если есть данные
    if recommendation["total_answers"] >= 3:
        effective_difficulty = adaptive_difficulty
        print(f"📊 Адаптивная сложность: {effective_difficulty}")
    else:
        effective_difficulty = self.current_difficulty
    
    # Формируем промпт
    user_prompt = self.prompts.get_explain_prompt(
        topic=topic,
        grade=self.current_grade,
        difficulty=effective_difficulty
    )
    
    # Добавляем в контекст
    self.context.add_message("user", f"Объясни тему: {topic}", topic)
    
    # Получаем сообщения для API
    messages = self.context.get_messages_for_api()
    messages.append({"role": "user", "content": user_prompt})
    
    # Определяем тип задачи
    task_type = "explain_complex" if effective_difficulty == "hard" else "explain_simple"
    
    # Отправляем запрос
    print(f"🔄 Генерирую объяснение темы '{topic}'...")
    result = self.router.send_request(
        messages=messages,
        task_type=task_type,
        complexity=effective_difficulty,
        max_tokens=2000,
        temperature=0.7
    )
    
    if not result["success"]:
        return f"❌ Ошибка при генерации: {result['error']}"
    
    explanation = result["content"]
    
    # Проверяем и улучшаем качество
    if check_quality and QUALITY_CONFIG["check_explanations"]:
        print("🔍 Проверяю качество объяснения...")
        explanation, quality = self.quality_checker.ensure_quality(
            topic=topic,
            explanation=explanation,
            grade=self.current_grade,
            subject=self.current_subject
        )
        
        if quality.get("improved"):
            print(f"✨ Объяснение улучшено! Оценка: {quality.get('общая_оценка', 'N/A')}/10")
    
    # Сохраняем в контекст
    self.context.add_message("assistant", explanation)
    
    return explanation


# Обновите метод generate_exam:
def generate_exam(self, topic: str, exam_type: str = None, 
                  num_questions: int = 3) -> str:
    """
    Генерирует тестовые задания в формате ОГЭ/ЕГЭ.
    """
    if not self.router:
        return "❌ Ошибка: роутер моделей не инициализирован"
    
    exam = exam_type or self.current_exam_type
    
    # Получаем адаптивную сложность
    recommendation = self.adaptive.get_recommendation(topic)
    if recommendation["total_answers"] >= 3:
        difficulty = recommendation["current_difficulty"]
    else:
        difficulty = self.current_difficulty
    
    print(f"📝 Генерирую {num_questions} заданий {exam.upper()} по теме '{topic}'...")
    
    # Генерируем задания
    questions = self.exam_generator.generate_questions(
        topic=topic,
        subject=self.current_subject,
        exam_type=exam,
        num_questions=num_questions,
        difficulty=difficulty
    )
    
    if not questions:
        return "❌ Не удалось сгенерировать задания. Попробуйте ещё раз."
    
    # Сохраняем для проверки ответов
    self.current_questions = questions
    
    # Форматируем для отображения
    return self.exam_generator.format_questions_for_display(questions)


# Добавьте метод check_answer:
def check_answer(self, answer: str, question_number: int = 1) -> str:
    """
    Проверяет ответ ученика на задание.
    
    Args:
        answer: Ответ ученика
        question_number: Номер задания (начиная с 1)
        
    Returns:
        Результат проверки
    """
    if not self.current_questions:
        return "❌ Сначала сгенерируйте задания командой /exam"
    
    if question_number < 1 or question_number > len(self.current_questions):
        return f"❌ Неверный номер задания. Доступны: 1-{len(self.current_questions)}"
    
    question = self.current_questions[question_number - 1]
    
    # Проверяем ответ
    result = self.exam_generator.check_answer(question, answer)
    
    # Записываем для адаптивной системы
    is_correct = result.get("is_correct", False)
    self.adaptive.record_answer(question.topic, is_correct)
    
    return result.get("feedback", "Результат проверки недоступен")


# Добавьте метод для получения статистики:
def get_learning_stats(self) -> Dict:
    """Возвращает статистику обучения"""
    return {
        "adaptive": self.adaptive.get_overall_stats(),
        "quality": self.quality_checker.get_stats(),
        "exams": self.exam_generator.get_stats(),
        "router": self.router.get_stats() if self.router else {}
    }
```

## 5. Обновление CLI для новых функций

Добавьте в `main.py` обработку проверки ответов:

```python
# В главном цикле обработки команд добавьте:

elif command == "/check" or command == "/answer":
    if args:
        # Парсим: /check [номер] ответ
        parts = args.split(maxsplit=1)
        
        if len(parts) == 2 and parts[0].isdigit():
            num = int(parts[0])
            answer = parts[1]
        else:
            num = 1
            answer = args
        
        result = assistant.check_answer(answer, num)
        print(f"\n{result}")
    else:
        print("❌ Укажите ответ: /check 1 ваш_ответ")
        print("   или просто: /check ваш_ответ (для задания 1)")

elif command == "/progress":
    stats = assistant.get_learning_stats()
    adaptive = stats.get("adaptive", {})
    
    print(f"\n📊 ПРОГРЕСС ОБУЧЕНИЯ")
    print(f"{'─'*40}")
    print(f"Изучено тем: {adaptive.get('topics_studied', 0)}")
    print(f"Всего ответов: {adaptive.get('total_answers', 0)}")
    print(f"Правильных: {adaptive.get('total_correct', 0)}")
    
    overall_rate = adaptive.get('overall_rate', 0)
    print(f"Общий процент: {overall_rate*100:.1f}%")
    
    topics = adaptive.get('topics', {})
    if topics:
        print(f"\n📚 По темам:")
        for topic, data in topics.items():
            print(f"   • {topic}: {data['rate']*100:.0f}% ({data['difficulty']})")
```

## Практические задания

### 🟢 Базовый уровень

**Задание 1: Тестирование проверки качества**
Запустите `python quality.py` и протестируйте:
- Оценку простого объяснения
- Оценку слишком сложного объяснения
- Автоматическое улучшение

**Задание 2: Генерация тестов**
Запустите `python exam_quiz.py` и создайте тесты:
- 3 задания ОГЭ по математике
- 5 заданий ЕГЭ по физике

### 🟡 Средний уровень

**Задание 3: Специализированные критерии оценки**
Добавьте в `QualityChecker` специальные критерии для математики:
- Наличие формул
- Пошаговое решение
- Числовые примеры

**Задание 4: Типы заданий по предметам**
Расширьте `ExamQuizGenerator`, добавив специфичные типы заданий:
- Для русского языка: орфография, пунктуация
- Для истории: хронология, причинно-следственные связи

### 🔴 Продвинутый уровень

**Задание 5: Персонализированные рекомендации**
Расширьте `AdaptiveDifficultyManager`:
- Отслеживание типичных ошибок
- Рекомендации конкретных тем для повторения
- Предсказание успешности на экзамене

**Задание 6: Система достижений**
Добавьте геймификацию:
- Бейджи за серию правильных ответов
- Уровни "мастерства" по темам
- Статистика прогресса за неделю

## Контрольные вопросы

1. **Зачем нужна проверка качества объяснений?**
   <details>
   <summary>Ответ</summary>
   LLM могут генерировать слишком сложные, неполные или не соответствующие уровню ученика объяснения. QualityChecker с помощью LLM-as-a-judge оценивает понятность, полноту, наличие примеров и автоматически улучшает объяснение до приемлемого качества.
   </details>

2. **Как работает адаптивная система сложности?**
   <details>
   <summary>Ответ</summary>
   Система отслеживает процент правильных ответов по каждой теме. Если ученик отвечает правильно на 80%+ заданий — сложность повышается. Если менее 40% — понижается. Для изменения нужно минимум 3 ответа, чтобы избежать случайных колебаний.
   </details>

3. **Чем отличается проверка краткого ответа от развёрнутого?**
   <details>
   <summary>Ответ</summary>
   Краткий ответ (число, слово, буква) проверяется прямым сравнением после нормализации. Развёрнутый ответ требует LLM для оценки полноты, правильности рассуждений и логики изложения.
   </details>

4. **Почему важен формат ОГЭ/ЕГЭ для заданий?**
   <details>
   <summary>Ответ</summary>
   Школьники готовятся к конкретным экзаменам с определённой структурой заданий. Практика в знакомом формате (типы вопросов, оформление ответов) лучше подготавливает к реальному экзамену, чем произвольные задачи.
   </details>

5. **Какие fallback-стратегии используются при ошибках?**
   <details>
   <summary>Ответ</summary>
   При ошибке парсинга JSON — возвращаем безопасные значения по умолчанию. При недоступности модели — пробуем резервные. При ошибке проверки качества — возвращаем исходное объяснение. При ошибке генерации заданий — информируем пользователя.
   </details>

## Заключение урока

### Что мы изучили

В этом уроке мы добавили "интеллект" в SchoolBot:

- **QualityChecker** — автоматическая проверка и улучшение объяснений
- **ExamQuizGenerator** — генерация заданий в формате реальных экзаменов
- **AdaptiveDifficultyManager** — подстройка сложности под успехи ученика
- **Обработка ошибок** — graceful degradation на всех уровнях

### Связь с Модулем 2

Мы применили ключевые концепции:
- **Урок 3:** LLM-as-a-judge → наш QualityChecker
- **Урок 2:** Fallback-стратегии → обработка ошибок везде
- **Продвинутые промпты:** Структурированные ответы в JSON

### Что нас ждёт дальше

В следующем уроке **"Финализация и тестирование"** мы:
- Завершим CLI-интерфейс со всеми командами
- Добавим сохранение/загрузку сессий
- Протестируем на реальных школьных темах
- Создадим документацию проекта

### Ваш прогресс

🎓 **SchoolBot стал умнее!** Теперь он:
- ✅ Проверяет качество своих объяснений
- ✅ Генерирует тесты в формате ОГЭ/ЕГЭ
- ✅ Проверяет ответы учеников
- ✅ Адаптируется к уровню каждого ученика
- ✅ Gracefully обрабатывает ошибки

**Готовы к финишу?** Переходите к [Уроку 4: Финализация и тестирование](lesson_4_finalization.md)!

---

## Дополнительные материалы

### LLM-as-a-Judge:
- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)
- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)

### Адаптивное обучение:
- [Intelligent Tutoring Systems](https://en.wikipedia.org/wiki/Intelligent_tutoring_system)
- [Personalized Learning](https://www.edutopia.org/article/personalized-learning)

### Генерация тестов:
- [Automatic Question Generation](https://aclanthology.org/)
- [ФИПИ — материалы экзаменов](https://fipi.ru/)

