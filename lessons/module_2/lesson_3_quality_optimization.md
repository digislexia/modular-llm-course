# Урок 3: Контроль качества и оптимизация

## Введение

Создать AI-приложение — это полдела. Настоящий вызов — обеспечить стабильно высокое качество ответов и оптимальную стоимость работы системы. В продакшене вы столкнётесь с вариативностью ответов LLM, необходимостью валидации результатов и контролем бюджета.

В этом финальном уроке модуля вы научитесь использовать **LLM как судью** для оценки качества других LLM, создавать пайплайны автоматической проверки и оптимизировать использование токенов. Эти навыки превратят ваши экспериментальные проекты в надёжные продуктовые решения.

## Цели урока

После завершения урока вы сможете:

- ✅ Использовать LLM-as-a-judge для оценки качества ответов
- ✅ Создавать системы автоматической валидации контента
- ✅ Определять и применять критерии качества для разных задач
- ✅ Оптимизировать использование токенов и снижать стоимость
- ✅ Строить пайплайны контроля качества для продакшена

## Ключевые термины

- **LLM-as-a-judge** — использование языковой модели для оценки качества ответов другой модели
- **Критерии качества** — набор параметров для оценки ответа (точность, полнота, стиль, релевантность)
- **Валидационный пайплайн** — автоматизированная система проверки контента
- **Токенная оптимизация** — методы сокращения количества токенов при сохранении качества
- **A/B тестирование промптов** — сравнение эффективности разных вариантов промптов

## 1. Концепция LLM-as-a-judge

### Почему модели могут оценивать других модели?

LLM обладают способностями к:
- **Анализу текста** — понимают структуру, логику, стиль
- **Сравнению с критериями** — могут оценить соответствие требованиям
- **Объективной оценке** — не подвержены усталости и настроению
- **Масштабируемости** — могут проверить тысячи ответов

### Преимущества LLM-as-a-judge

```python
# Сравнение подходов к контролю качества

# Ручная проверка
manual_check = {
    "cost_per_review": 50,  # рублей
    "time_per_review": 300,  # секунд
    "daily_capacity": 100,   # отзывов
    "consistency": 0.7       # человеческий фактор
}

# LLM-as-a-judge
llm_check = {
    "cost_per_review": 0.5,   # рублей (примерно)
    "time_per_review": 5,     # секунд
    "daily_capacity": 10000,  # отзывов
    "consistency": 0.95       # высокая согласованность
}

print("📊 СРАВНЕНИЕ ПОДХОДОВ К КОНТРОЛЮ КАЧЕСТВА")
print("=" * 50)
print(f"Стоимость проверки:")
print(f"  Ручная: {manual_check['cost_per_review']} руб/отзыв")
print(f"  LLM: {llm_check['cost_per_review']} руб/отзыв")
print(f"  Экономия: {manual_check['cost_per_review'] / llm_check['cost_per_review']:.0f}x")
print()
print(f"Скорость обработки:")
print(f"  Ручная: {manual_check['time_per_review']}с/отзыв")  
print(f"  LLM: {llm_check['time_per_review']}с/отзыв")
print(f"  Ускорение: {manual_check['time_per_review'] / llm_check['time_per_review']:.0f}x")
```

### Ограничения подхода

**Важно понимать:**
- LLM не заменяет человеческую оценку полностью
- Требуется валидация самих критериев оценки
- Модели могут быть предвзятыми к собственным ответам
- Необходима периодическая калибровка

## 2. Создание системы LLM-as-a-judge

### Базовая архитектура

```python
import os
import requests
import json
from typing import Dict, List, Union
from dotenv import load_dotenv

load_dotenv()

class LLMJudge:
    """Система использования LLM для оценки качества ответов"""
    
    def __init__(self, judge_model: str = "microsoft/wizardlm-2-8x22b"):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.judge_model = judge_model
        
        if not self.api_key:
            raise ValueError("API ключ не найден!")
        
        # Предопределённые критерии качества
        self.quality_criteria = {
            "accuracy": {
                "name": "Точность",
                "description": "Фактическая корректность информации",
                "weight": 0.3
            },
            "completeness": {
                "name": "Полнота", 
                "description": "Покрывает ли ответ все аспекты вопроса",
                "weight": 0.25
            },
            "clarity": {
                "name": "Ясность",
                "description": "Понятность и структурированность изложения",
                "weight": 0.2
            },
            "relevance": {
                "name": "Релевантность",
                "description": "Соответствие ответа заданному вопросу",
                "weight": 0.25
            }
        }
    
    def create_judge_prompt(self, question: str, answer: str, 
                          criteria: List[str] = None, 
                          custom_requirements: str = "") -> str:
        """Создаёт промпт для модели-судьи"""
        
        if criteria is None:
            criteria = list(self.quality_criteria.keys())
        
        criteria_descriptions = []
        for criterion in criteria:
            if criterion in self.quality_criteria:
                info = self.quality_criteria[criterion]
                criteria_descriptions.append(f"- **{info['name']}**: {info['description']}")
        
        criteria_text = "\n".join(criteria_descriptions)
        
        judge_prompt = f"""
Ты экспертная система для оценки качества ответов ИИ-моделей.

ЗАДАЧА: Оцени качество ответа на основе заданных критериев.

ВОПРОС:
{question}

ОТВЕТ ДЛЯ ОЦЕНКИ:
{answer}

КРИТЕРИИ ОЦЕНКИ:
{criteria_text}

{custom_requirements}

ИНСТРУКЦИИ:
1. Оцени ответ по каждому критерию от 1 до 10
2. Обоснуй каждую оценку конкретными примерами
3. Укажи сильные и слабые стороны ответа
4. Дай рекомендации по улучшению
5. Выведи итоговую оценку

ФОРМАТ ОТВЕТА:
```json
{{
    "overall_score": [1-10],
    "criteria_scores": {{
        "accuracy": {{"score": [1-10], "reasoning": "обоснование"}},
        "completeness": {{"score": [1-10], "reasoning": "обоснование"}},
        "clarity": {{"score": [1-10], "reasoning": "обоснование"}},
        "relevance": {{"score": [1-10], "reasoning": "обоснование"}}
    }},
    "strengths": ["сильная сторона 1", "сильная сторона 2"],
    "weaknesses": ["слабая сторона 1", "слабая сторона 2"], 
    "recommendations": ["рекомендация 1", "рекомендация 2"],
    "verdict": "ОТЛИЧНО/ХОРОШО/УДОВЛЕТВОРИТЕЛЬНО/ПЛОХО"
}}
```

Будь объективен и конструктивен в оценке.
"""
        
        return judge_prompt.strip()
    
    def evaluate_response(self, question: str, answer: str, 
                         criteria: List[str] = None,
                         custom_requirements: str = "") -> Dict:
        """Оценивает качество ответа с помощью LLM-судьи"""
        
        judge_prompt = self.create_judge_prompt(
            question, answer, criteria, custom_requirements
        )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.judge_model,
            "messages": [{"role": "user", "content": judge_prompt}],
            "max_tokens": 1500,
            "temperature": 0.3,  # Низкая температура для консистентности
            "usage": {"include": True}
        }
        
        try:
            response = requests.post(self.url, headers=headers, json=data, timeout=45)
            response.raise_for_status()
            
            result = response.json()
            judge_response = result["choices"][0]["message"]["content"]
            
            # Пытаемся извлечь JSON из ответа
            try:
                # Ищем JSON блок в ответе
                import re
                json_match = re.search(r'```json\n(.*?)\n```', judge_response, re.DOTALL)
                if json_match:
                    evaluation_data = json.loads(json_match.group(1))
                else:
                    # Если нет блока, пытаемся парсить весь ответ
                    evaluation_data = json.loads(judge_response)
                
                return {
                    "success": True,
                    "evaluation": evaluation_data,
                    "raw_response": judge_response,
                    "usage": result.get("usage", {}),
                    "judge_model": self.judge_model
                }
                
            except json.JSONDecodeError:
                # Если не удалось парсить JSON, возвращаем текстовый анализ
                return {
                    "success": True,
                    "evaluation": {"raw_analysis": judge_response},
                    "raw_response": judge_response,
                    "usage": result.get("usage", {}),
                    "judge_model": self.judge_model,
                    "note": "Не удалось парсить структурированную оценку"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

def demo_llm_judge():
    """Демонстрирует работу LLM-судьи"""
    
    judge = LLMJudge()
    
    # Тестовые кейсы: вопрос и два ответа разного качества
    test_cases = [
        {
            "question": "Объясни принцип работы блокчейна простыми словами",
            "good_answer": """
Блокчейн — это цифровая книга учёта, где записи нельзя изменить или удалить.

Представьте тетрадь, где каждая страница связана с предыдущей специальным кодом. 
Если кто-то попытается изменить запись на старой странице, код не совпадёт, 
и все поймут, что данные были подделаны.

Основные принципы:
1. **Децентрализация**: копии тетради есть у множества людей
2. **Прозрачность**: все записи видны всем участникам
3. **Неизменность**: изменить прошлые записи практически невозможно

Так обеспечивается доверие без центрального органа управления.
""",
            "poor_answer": """
Блокчейн это такая штука для криптовалют. Там есть блоки которые связаны 
в цепочку. Используется для биткоина и других монет. Очень безопасно.
"""
        },
        {
            "question": "Как приготовить идеальную яичницу?",
            "good_answer": """
Секрет идеальной яичницы — в контроле температуры и времени:

**Подготовка:**
- Используйте свежие яйца комнатной температуры
- Разогрейте сковороду на среднем огне
- Добавьте немного масла или сливочного масла

**Приготовление:**
1. Аккуратно разбейте яйца в сковороду
2. Убавьте огонь до минимума
3. Готовьте 2-3 минуты, не перемешивая
4. Белок должен стать непрозрачным, желток остаться жидким

**Секреты:**
- Не солите сразу — это может сделать белок жёстким
- Накройте сковороду крышкой для равномерного прогрева
- Подавайте немедленно

Результат: нежный белок и кремовый желток!
""",
            "poor_answer": """
Разбей яйца на сковородку и жари пока не будет готово. Можно посолить.
"""
        }
    ]
    
    print("⚖️ ДЕМОНСТРАЦИЯ LLM-AS-A-JUDGE")
    print("=" * 60)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 ТЕСТ {i}: {case['question']}")
        print("=" * 50)
        
        # Оцениваем хороший ответ
        print("\n🔍 Оценка хорошего ответа:")
        print("-" * 30)
        good_evaluation = judge.evaluate_response(
            case["question"], 
            case["good_answer"]
        )
        
        if good_evaluation["success"]:
            if "evaluation" in good_evaluation and isinstance(good_evaluation["evaluation"], dict):
                eval_data = good_evaluation["evaluation"]
                if "overall_score" in eval_data:
                    print(f"✅ Общая оценка: {eval_data['overall_score']}/10")
                    print(f"📊 Вердикт: {eval_data.get('verdict', 'N/A')}")
                    
                    if "strengths" in eval_data:
                        print("💪 Сильные стороны:")
                        for strength in eval_data["strengths"]:
                            print(f"  + {strength}")
                else:
                    print("📄 Анализ:")
                    print(eval_data.get("raw_analysis", "Анализ недоступен")[:300] + "...")
            else:
                print("📄 Анализ недоступен")
        else:
            print(f"❌ Ошибка: {good_evaluation['error']}")
        
        # Небольшая пауза
        time.sleep(1)
        
        # Оцениваем плохой ответ
        print("\n🔍 Оценка слабого ответа:")
        print("-" * 30)
        poor_evaluation = judge.evaluate_response(
            case["question"], 
            case["poor_answer"]
        )
        
        if poor_evaluation["success"]:
            if "evaluation" in poor_evaluation and isinstance(poor_evaluation["evaluation"], dict):
                eval_data = poor_evaluation["evaluation"]
                if "overall_score" in eval_data:
                    print(f"⚠️ Общая оценка: {eval_data['overall_score']}/10")
                    print(f"📊 Вердикт: {eval_data.get('verdict', 'N/A')}")
                    
                    if "recommendations" in eval_data:
                        print("📝 Рекомендации по улучшению:")
                        for rec in eval_data["recommendations"]:
                            print(f"  → {rec}")
                else:
                    print("📄 Анализ:")
                    print(eval_data.get("raw_analysis", "Анализ недоступен")[:300] + "...")
        else:
            print(f"❌ Ошибка: {poor_evaluation['error']}")
        
        if i < len(test_cases):
            input("\nНажмите Enter для следующего теста...")

if __name__ == "__main__":
    demo_llm_judge()
```

### 🔍 Проверьте себя

Запустите демонстрацию LLM-судьи и проанализируйте:
- Как модель оценивает структурированный vs неструктурированный ответ?
- Какие критерии она считает наиболее важными?
- Насколько полезны её рекомендации по улучшению?

## 3. Создание пайплайна качества

### Двухэтапная система: генерация + проверка

```python
import time
from datetime import datetime

class QualityControlPipeline:
    """Пайплайн контроля качества: генерация → проверка → улучшение"""
    
    def __init__(self, generator_model: str = "microsoft/wizardlm-2-8x22b",
                 judge_model: str = "microsoft/wizardlm-2-8x22b"):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.generator_model = generator_model
        self.judge = LLMJudge(judge_model)
        
        # Минимальные требования качества
        self.quality_thresholds = {
            "overall_score": 7,
            "accuracy": 7,
            "completeness": 6,
            "clarity": 6,
            "relevance": 8
        }
    
    def generate_response(self, prompt: str, context: str = "",
                         max_tokens: int = 1000, temperature: float = 0.7) -> Dict:
        """Генерирует ответ с помощью основной модели"""
        
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.generator_model,
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "usage": {"include": True}
        }
        
        try:
            response = requests.post(self.url, headers=headers, json=data, timeout=45)
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "success": True,
                "response": result["choices"][0]["message"]["content"],
                "usage": result.get("usage", {}),
                "model": self.generator_model
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def improve_response(self, original_prompt: str, response: str, 
                        evaluation: Dict) -> Dict:
        """Улучшает ответ на основе оценки судьи"""
        
        if not evaluation.get("success"):
            return {"success": False, "error": "Нет данных для улучшения"}
        
        eval_data = evaluation.get("evaluation", {})
        
        # Создаём промпт для улучшения
        improvement_prompt = f"""
Исходный вопрос: {original_prompt}

Исходный ответ: {response}

Анализ качества показал следующие проблемы:
"""
        
        if "weaknesses" in eval_data:
            improvement_prompt += "\nСлабые стороны:\n"
            for weakness in eval_data["weaknesses"]:
                improvement_prompt += f"- {weakness}\n"
        
        if "recommendations" in eval_data:
            improvement_prompt += "\nРекомендации по улучшению:\n"
            for rec in eval_data["recommendations"]:
                improvement_prompt += f"- {rec}\n"
        
        improvement_prompt += """
Создай улучшенную версию ответа, устраняющую выявленные недостатки.
Сохрани полезную информацию из исходного ответа, но сделай его:
- Более точным фактически
- Более полным по содержанию  
- Более понятным для читателя
- Более релевантным вопросу

Улучшенный ответ:
"""
        
        return self.generate_response(improvement_prompt, max_tokens=1200, temperature=0.5)
    
    def process_request(self, prompt: str, context: str = "",
                       max_improvement_iterations: int = 2) -> Dict:
        """Полный цикл: генерация → оценка → улучшение (при необходимости)"""
        
        print(f"🔄 Обрабатываем запрос: {prompt[:100]}...")
        
        results = {
            "original_prompt": prompt,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "iterations": []
        }
        
        current_response = None
        
        for iteration in range(max_improvement_iterations + 1):
            print(f"\n📝 Итерация {iteration + 1}")
            print("-" * 30)
            
            # Генерируем или улучшаем ответ
            if iteration == 0:
                # Первоначальная генерация
                print("🤖 Генерируем исходный ответ...")
                generation_result = self.generate_response(prompt, context)
            else:
                # Улучшение на основе предыдущей оценки
                print("🔧 Улучшаем ответ на основе обратной связи...")
                generation_result = self.improve_response(
                    prompt, 
                    current_response, 
                    results["iterations"][-1]["evaluation"]
                )
            
            if not generation_result["success"]:
                print(f"❌ Ошибка генерации: {generation_result['error']}")
                break
            
            current_response = generation_result["response"]
            print(f"✅ Ответ получен ({len(current_response)} символов)")
            
            # Оцениваем качество
            print("⚖️ Оцениваем качество...")
            evaluation_result = self.judge.evaluate_response(prompt, current_response)
            
            iteration_data = {
                "iteration": iteration + 1,
                "generation": generation_result,
                "evaluation": evaluation_result,
                "meets_quality_threshold": False
            }
            
            if evaluation_result["success"]:
                eval_data = evaluation_result.get("evaluation", {})
                
                if "overall_score" in eval_data:
                    overall_score = eval_data["overall_score"]
                    print(f"📊 Общая оценка: {overall_score}/10")
                    
                    # Проверяем пороги качества
                    if overall_score >= self.quality_thresholds["overall_score"]:
                        iteration_data["meets_quality_threshold"] = True
                        print("✅ Качество соответствует требованиям!")
                    else:
                        print("⚠️ Качество ниже порога, требуется улучшение")
                else:
                    print("📄 Получен текстовый анализ")
            else:
                print(f"❌ Ошибка оценки: {evaluation_result['error']}")
            
            results["iterations"].append(iteration_data)
            
            # Если качество удовлетворительно или это последняя итерация
            if (iteration_data["meets_quality_threshold"] or 
                iteration == max_improvement_iterations):
                break
                
            time.sleep(1)  # Пауза между итерациями
        
        # Финальные результаты
        final_iteration = results["iterations"][-1]
        results["final_response"] = final_iteration["generation"]["response"]
        results["total_iterations"] = len(results["iterations"])
        results["quality_achieved"] = final_iteration["meets_quality_threshold"]
        
        return results

def demo_quality_pipeline():
    """Демонстрирует работу пайплайна контроля качества"""
    
    pipeline = QualityControlPipeline()
    
    test_prompts = [
        {
            "prompt": "Как создать стартап в области FinTech?",
            "context": "Ответ предназначен для начинающих предпринимателей без технического образования"
        },
        {
            "prompt": "Объясни квантовые вычисления",
            "context": "Аудитория: студенты IT-специальностей, уровень: средний"
        }
    ]
    
    print("🔄 ДЕМОНСТРАЦИЯ ПАЙПЛАЙНА КОНТРОЛЯ КАЧЕСТВА")
    print("=" * 60)
    
    for i, test_case in enumerate(test_prompts, 1):
        print(f"\n🎯 ТЕСТ {i}")
        print("=" * 40)
        
        results = pipeline.process_request(
            test_case["prompt"],
            test_case["context"],
            max_improvement_iterations=2
        )
        
        print(f"\n📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
        print("-" * 30)
        print(f"Итераций: {results['total_iterations']}")
        print(f"Качество достигнуто: {'✅ Да' if results['quality_achieved'] else '❌ Нет'}")
        print(f"Финальный ответ:")
        print(f"  {results['final_response'][:200]}...")
        
        # Показываем прогресс по итерациям
        print(f"\n🔍 ПРОГРЕСС КАЧЕСТВА:")
        for iteration in results["iterations"]:
            eval_data = iteration["evaluation"].get("evaluation", {})
            score = eval_data.get("overall_score", "N/A")
            verdict = eval_data.get("verdict", "N/A")
            print(f"  Итерация {iteration['iteration']}: {score}/10 ({verdict})")
        
        if i < len(test_prompts):
            input("\nНажмите Enter для следующего теста...")

if __name__ == "__main__":
    demo_quality_pipeline()
```

### 🔍 Проверьте себя

После запуска пайплайна проанализируйте:
- Улучшается ли качество от итерации к итерации?
- Какие аспекты исправляются чаще всего?
- Стоит ли дополнительная стоимость улучшения результата?

## 4. Автоматические проверки и валидация

### Создание системы валидаторов

```python
import re
from typing import Callable, List

class ContentValidator:
    """Система автоматической валидации контента"""
    
    def __init__(self):
        self.validators = {}
        self.register_default_validators()
    
    def register_validator(self, name: str, validator_func: Callable, description: str):
        """Регистрирует новый валидатор"""
        self.validators[name] = {
            "func": validator_func,
            "description": description
        }
    
    def register_default_validators(self):
        """Регистрирует стандартные валидаторы"""
        
        # Проверка минимальной длины
        def min_length_validator(text: str, min_length: int = 100) -> Dict:
            length = len(text.strip())
            passed = length >= min_length
            return {
                "passed": passed,
                "message": f"Длина текста: {length} символов (минимум: {min_length})",
                "score": min(1.0, length / min_length) if min_length > 0 else 1.0
            }
        
        # Проверка наличия структуры (списки, пункты)
        def structure_validator(text: str) -> Dict:
            patterns = [
                r'\d+\.',  # Нумерованные списки: "1.", "2."
                r'[-*•]',  # Маркированные списки
                r'\*\*.*?\*\*',  # Выделение жирным
                r'#{1,6}\s',  # Заголовки
                r'\n\n'  # Разделение на абзацы
            ]
            
            structure_score = 0
            found_elements = []
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    structure_score += 1
                    found_elements.append(pattern)
            
            passed = structure_score >= 2  # Минимум 2 элемента структуры
            
            return {
                "passed": passed,
                "message": f"Найдено {structure_score} элементов структуры: {found_elements}",
                "score": min(1.0, structure_score / 3)
            }
        
        # Проверка на наличие конкретных пунктов
        def bullet_points_validator(text: str, min_points: int = 3) -> Dict:
            # Ищем различные форматы списков
            patterns = [
                r'^\d+\.',  # "1.", "2.", ...
                r'^[-*•]',  # "- ", "* ", "• "
                r'^\w+:',   # "Первое:", "Второе:"
            ]
            
            points_found = 0
            for pattern in patterns:
                matches = re.findall(pattern, text, re.MULTILINE)
                points_found += len(matches)
            
            passed = points_found >= min_points
            
            return {
                "passed": passed,
                "message": f"Найдено {points_found} пунктов (минимум: {min_points})",
                "score": min(1.0, points_found / min_points) if min_points > 0 else 1.0
            }
        
        # Проверка на отсутствие повторов
        def repetition_validator(text: str) -> Dict:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip().lower() for s in sentences if s.strip()]
            
            unique_sentences = set(sentences)
            repetition_ratio = len(unique_sentences) / len(sentences) if sentences else 1
            
            passed = repetition_ratio >= 0.8  # Не более 20% повторов
            
            return {
                "passed": passed,
                "message": f"Уникальность текста: {repetition_ratio:.2%}",
                "score": repetition_ratio
            }
        
        # Проверка наличия ключевых слов
        def keyword_presence_validator(text: str, keywords: List[str] = None) -> Dict:
            if not keywords:
                return {"passed": True, "message": "Ключевые слова не заданы", "score": 1.0}
            
            text_lower = text.lower()
            found_keywords = [kw for kw in keywords if kw.lower() in text_lower]
            coverage = len(found_keywords) / len(keywords) if keywords else 1.0
            
            passed = coverage >= 0.5  # Минимум 50% ключевых слов
            
            return {
                "passed": passed,
                "message": f"Найдено {len(found_keywords)} из {len(keywords)} ключевых слов: {found_keywords}",
                "score": coverage
            }
        
        # Регистрируем валидаторы
        self.register_validator("min_length", min_length_validator, "Проверка минимальной длины")
        self.register_validator("structure", structure_validator, "Проверка структурированности")
        self.register_validator("bullet_points", bullet_points_validator, "Проверка наличия пунктов")
        self.register_validator("repetition", repetition_validator, "Проверка на повторы")
        self.register_validator("keywords", keyword_presence_validator, "Проверка ключевых слов")
    
    def validate_content(self, text: str, enabled_validators: List[str] = None,
                        validator_params: Dict = None) -> Dict:
        """Валидирует контент с помощью выбранных валидаторов"""
        
        if enabled_validators is None:
            enabled_validators = list(self.validators.keys())
        
        if validator_params is None:
            validator_params = {}
        
        results = {
            "overall_passed": True,
            "overall_score": 0.0,
            "validator_results": {},
            "failed_validators": [],
            "warnings": []
        }
        
        total_score = 0.0
        
        for validator_name in enabled_validators:
            if validator_name not in self.validators:
                results["warnings"].append(f"Валидатор '{validator_name}' не найден")
                continue
            
            validator_func = self.validators[validator_name]["func"]
            params = validator_params.get(validator_name, {})
            
            try:
                # Вызываем валидатор с параметрами
                if params:
                    validator_result = validator_func(text, **params)
                else:
                    validator_result = validator_func(text)
                
                results["validator_results"][validator_name] = validator_result
                total_score += validator_result["score"]
                
                if not validator_result["passed"]:
                    results["overall_passed"] = False
                    results["failed_validators"].append(validator_name)
                    
            except Exception as e:
                results["warnings"].append(f"Ошибка в валидаторе '{validator_name}': {e}")
                results["validator_results"][validator_name] = {
                    "passed": False,
                    "message": f"Ошибка выполнения: {e}",
                    "score": 0.0
                }
        
        # Вычисляем общий скор
        active_validators = len([v for v in enabled_validators if v in self.validators])
        if active_validators > 0:
            results["overall_score"] = total_score / active_validators
        
        return results

def demo_content_validation():
    """Демонстрирует систему валидации контента"""
    
    validator = ContentValidator()
    
    # Тестовые примеры разного качества
    test_cases = [
        {
            "name": "Хорошо структурированный ответ",
            "text": """
Создание стартапа в области FinTech требует системного подхода:

**1. Исследование рынка**
- Анализ конкурентов
- Выявление потребностей клиентов
- Изучение регулятивных требований

**2. Техническая разработка**
- Выбор технологического стека
- Обеспечение безопасности данных
- Создание MVP (минимально жизнеспособного продукта)

**3. Юридические аспекты**
- Получение лицензий
- Соответствие требованиям ЦБ РФ
- Защита интеллектуальной собственности

**4. Финансирование**
Для запуска потребуется:
• Начальный капитал: 5-10 млн рублей
• Поиск инвесторов или грантов
• Планирование денежных потоков

Ключевые факторы успеха: инновационность решения, команда экспертов, грамотный маркетинг.
""",
            "keywords": ["стартап", "FinTech", "инвестор", "MVP", "безопасность"],
            "expected_quality": "высокое"
        },
        {
            "name": "Слабо структурированный ответ", 
            "text": """
Чтобы создать стартап нужно придумать идею и найти деньги. Также важно изучить рынок и понять что нужно клиентам. Потом разработать продукт и запустить его. Не забудьте про маркетинг.
""",
            "keywords": ["стартап", "идея", "клиенты", "продукт"],
            "expected_quality": "низкое"
        }
    ]
    
    print("🔍 ДЕМОНСТРАЦИЯ АВТОМАТИЧЕСКОЙ ВАЛИДАЦИИ")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 ТЕСТ {i}: {test_case['name']}")
        print("=" * 40)
        print(f"Ожидаемое качество: {test_case['expected_quality']}")
        
        # Настройки валидаторов для этого теста
        validator_params = {
            "min_length": {"min_length": 200},
            "bullet_points": {"min_points": 4},
            "keywords": {"keywords": test_case["keywords"]}
        }
        
        # Запускаем валидацию
        validation_result = validator.validate_content(
            test_case["text"],
            enabled_validators=["min_length", "structure", "bullet_points", "keywords"],
            validator_params=validator_params
        )
        
        # Выводим результаты
        print(f"\n📊 РЕЗУЛЬТАТЫ ВАЛИДАЦИИ:")
        print(f"Общий результат: {'✅ ПРОШЕЛ' if validation_result['overall_passed'] else '❌ НЕ ПРОШЕЛ'}")
        print(f"Общий скор: {validation_result['overall_score']:.2f}/1.00")
        
        if validation_result["failed_validators"]:
            print(f"Провалившие валидаторы: {validation_result['failed_validators']}")
        
        print("\n🔍 Детальные результаты:")
        for validator_name, result in validation_result["validator_results"].items():
            status = "✅" if result["passed"] else "❌"
            print(f"  {status} {validator_name}: {result['message']} (скор: {result['score']:.2f})")
        
        if validation_result["warnings"]:
            print("\n⚠️ Предупреждения:")
            for warning in validation_result["warnings"]:
                print(f"  - {warning}")
        
        if i < len(test_cases):
            input("\nНажмите Enter для следующего теста...")

if __name__ == "__main__":
    demo_content_validation()
```

## 5. Оптимизация токенов и стоимости

### Стратегии экономии токенов

```python
import tiktoken
from typing import Optional

class TokenOptimizer:
    """Оптимизирует использование токенов для снижения стоимости"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        # Используем токенизатор для подсчёта токенов
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Если модель не поддерживается, используем общий токенизатор
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Подсчитывает количество токенов в тексте"""
        return len(self.encoding.encode(text))
    
    def estimate_cost(self, prompt_tokens: int, response_tokens: int,
                     prompt_price: float = 0.0015, completion_price: float = 0.002) -> float:
        """Оценивает стоимость запроса в долларах (цены за 1K токенов)"""
        prompt_cost = (prompt_tokens / 1000) * prompt_price
        completion_cost = (response_tokens / 1000) * completion_price
        return prompt_cost + completion_cost
    
    def compress_prompt(self, prompt: str, max_reduction: float = 0.3) -> Dict:
        """Сжимает промпт, сохраняя смысл"""
        
        original_tokens = self.count_tokens(prompt)
        
        optimization_strategies = [
            self._remove_redundant_phrases,
            self._shorten_examples,
            self._compress_instructions,
            self._remove_excessive_formatting
        ]
        
        current_prompt = prompt
        optimization_log = []
        
        for strategy in optimization_strategies:
            result = strategy(current_prompt)
            if result["tokens_saved"] > 0:
                current_tokens = self.count_tokens(result["optimized_text"])
                reduction = (original_tokens - current_tokens) / original_tokens
                
                # Применяем оптимизацию, если экономия не превышает лимит
                if reduction <= max_reduction:
                    current_prompt = result["optimized_text"]
                    optimization_log.append({
                        "strategy": result["strategy_name"],
                        "tokens_saved": result["tokens_saved"],
                        "description": result["description"]
                    })
                    
                    # Если достигли целевого сокращения, останавливаемся
                    if reduction >= max_reduction * 0.8:
                        break
        
        final_tokens = self.count_tokens(current_prompt)
        total_saved = original_tokens - final_tokens
        
        return {
            "original_prompt": prompt,
            "optimized_prompt": current_prompt,
            "original_tokens": original_tokens,
            "final_tokens": final_tokens,
            "tokens_saved": total_saved,
            "reduction_percentage": (total_saved / original_tokens) * 100,
            "optimization_log": optimization_log
        }
    
    def _remove_redundant_phrases(self, text: str) -> Dict:
        """Удаляет избыточные фразы и повторы"""
        
        # Паттерны избыточности
        redundant_patterns = [
            (r'\s+', ' '),  # Множественные пробелы
            (r'(\w+)\s+\1\b', r'\1'),  # Повторяющиеся слова
            (r'пожалуйста,?\s*', ''),  # Избыточная вежливость в промптах
            (r'будьте\s+(добры|любезны),?\s*', ''),
            (r'если\s+можно,?\s*', ''),
            (r'я\s+хочу,?\s*что[бы]*\s+', ''),
            (r'мне\s+нужно,?\s*что[бы]*\s+', ''),
            (r'\s*,\s*и\s+также\s*', ', '),
            (r'\s*,\s*а\s+также\s*', ', '),
        ]
        
        optimized = text
        tokens_saved = 0
        
        original_tokens = self.count_tokens(text)
        
        for pattern, replacement in redundant_patterns:
            optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)
        
        optimized = optimized.strip()
        new_tokens = self.count_tokens(optimized)
        tokens_saved = original_tokens - new_tokens
        
        return {
            "strategy_name": "remove_redundancy",
            "optimized_text": optimized,
            "tokens_saved": tokens_saved,
            "description": "Удаление избыточных фраз и повторов"
        }
    
    def _shorten_examples(self, text: str) -> Dict:
        """Сокращает примеры, сохраняя их суть"""
        
        # Ищем блоки с примерами
        example_patterns = [
            r'Пример[ы]?:\s*\n(.*?)(?=\n\n|\n[А-Я]|\Z)',
            r'Например:\s*\n(.*?)(?=\n\n|\n[А-Я]|\Z)',
        ]
        
        optimized = text
        original_tokens = self.count_tokens(text)
        
        for pattern in example_patterns:
            examples = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for example in examples:
                if len(example) > 200:  # Длинные примеры
                    # Берём первые 100 символов + многоточие
                    shortened = example[:100] + "..."
                    optimized = optimized.replace(example, shortened)
        
        new_tokens = self.count_tokens(optimized)
        tokens_saved = original_tokens - new_tokens
        
        return {
            "strategy_name": "shorten_examples",
            "optimized_text": optimized,
            "tokens_saved": tokens_saved,
            "description": "Сокращение длинных примеров"
        }
    
    def _compress_instructions(self, text: str) -> Dict:
        """Сжимает инструкции, делая их более лаконичными"""
        
        # Замены для сжатия инструкций
        compression_rules = [
            (r'подробно\s+', ''),
            (r'детально\s+', ''),
            (r'тщательно\s+', ''),
            (r'в\s+деталях\s+', ''),
            (r'как\s+можно\s+более\s+', ''),
            (r'постарайся\s+', ''),
            (r'попытайся\s+', ''),
            (r'не\s+забудь\s+', ''),
            (r'обязательно\s+', ''),
            (r'внимательно\s+', ''),
            (r'используя\s+всю\s+доступную\s+информацию,?\s*', ''),
            (r'принимая\s+во\s+внимание\s+', 'учитывая '),
            (r'основываясь\s+на\s+', 'по '),
        ]
        
        optimized = text
        original_tokens = self.count_tokens(text)
        
        for pattern, replacement in compression_rules:
            optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)
        
        new_tokens = self.count_tokens(optimized)
        tokens_saved = original_tokens - new_tokens
        
        return {
            "strategy_name": "compress_instructions",
            "optimized_text": optimized,
            "tokens_saved": tokens_saved,
            "description": "Сжатие инструкций и указаний"
        }
    
    def _remove_excessive_formatting(self, text: str) -> Dict:
        """Упрощает избыточное форматирование"""
        
        formatting_rules = [
            (r'\*{3,}', '**'),  # Тройные звёздочки → двойные
            (r'={4,}', '==='),  # Длинные линии
            (r'-{4,}', '---'),  # Длинные дефисы
            (r'\n{3,}', '\n\n'),  # Множественные переносы строк
            (r'\s*\n\s*\n\s*\n\s*', '\n\n'),  # Пустые строки с пробелами
        ]
        
        optimized = text
        original_tokens = self.count_tokens(text)
        
        for pattern, replacement in formatting_rules:
            optimized = re.sub(pattern, replacement, optimized)
        
        new_tokens = self.count_tokens(optimized)
        tokens_saved = original_tokens - new_tokens
        
        return {
            "strategy_name": "simplify_formatting",
            "optimized_text": optimized,
            "tokens_saved": tokens_saved,
            "description": "Упрощение избыточного форматирования"
        }
    
    def create_context_summary(self, long_context: str, target_tokens: int = 500) -> Dict:
        """Создаёт краткое резюме длинного контекста"""
        
        # Простая стратегия: берём начало и конец, плюс ключевые предложения
        sentences = re.split(r'[.!?]+', long_context)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {
                "original_context": long_context,
                "summary": "",
                "original_tokens": 0,
                "summary_tokens": 0,
                "compression_ratio": 0
            }
        
        original_tokens = self.count_tokens(long_context)
        
        if original_tokens <= target_tokens:
            return {
                "original_context": long_context,
                "summary": long_context,
                "original_tokens": original_tokens,
                "summary_tokens": original_tokens,
                "compression_ratio": 1.0
            }
        
        # Берём первые и последние предложения
        num_sentences = len(sentences)
        keep_from_start = max(1, num_sentences // 4)
        keep_from_end = max(1, num_sentences // 4)
        
        summary_sentences = []
        summary_sentences.extend(sentences[:keep_from_start])
        
        if keep_from_start + keep_from_end < num_sentences:
            summary_sentences.append("[...резюме средней части...]")
        
        summary_sentences.extend(sentences[-keep_from_end:])
        
        summary = '. '.join(summary_sentences) + '.'
        summary_tokens = self.count_tokens(summary)
        
        # Если всё ещё слишком длинно, сократим дополнительно
        if summary_tokens > target_tokens and len(summary_sentences) > 3:
            summary_sentences = [summary_sentences[0], "[...краткое резюме...]", summary_sentences[-1]]
            summary = '. '.join(summary_sentences) + '.'
            summary_tokens = self.count_tokens(summary)
        
        return {
            "original_context": long_context,
            "summary": summary,
            "original_tokens": original_tokens,
            "summary_tokens": summary_tokens,
            "compression_ratio": summary_tokens / original_tokens
        }

def demo_token_optimization():
    """Демонстрирует оптимизацию токенов"""
    
    optimizer = TokenOptimizer()
    
    # Примеры промптов для оптимизации
    test_prompts = [
        """
Пожалуйста, будьте добры, подробно и детально объясните мне, что такое машинное обучение.
Я хочу, чтобы вы рассказали обо всех аспектах этой технологии, не забудьте упомянуть основные алгоритмы.
Постарайтесь использовать простые слова и приведите подробные примеры.

Пример: Представьте себе, что у вас есть огромная библиотека, в которой миллионы книг, 
и вы хотите научить компьютер автоматически определять жанр любой книги. Для этого вы показываете 
компьютеру тысячи примеров книг с уже известными жанрами...

Обязательно включите в ответ:
- Что такое машинное обучение
- Какие виды машинного обучения существуют  
- Как это применяется в реальной жизни
- Какие основные алгоритмы используются

Тщательно структурируйте ответ и постарайтесь сделать его как можно более понятным.
""",
        """
Напишите, пожалуйста, подробный план создания стартапа в области финансовых технологий.
Я хочу, чтобы план был максимально детализирован и включал все возможные этапы.

Например, нужно учесть такие аспекты как исследование рынка, разработка продукта, 
поиск финансирования, создание команды, правовые вопросы, маркетинговая стратегия...

Обязательно включите временные рамки для каждого этапа и примерные затраты.
"""
    ]
    
    print("⚡ ДЕМОНСТРАЦИЯ ОПТИМИЗАЦИИ ТОКЕНОВ")
    print("=" * 50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 ПРОМПТ {i}")
        print("=" * 30)
        
        # Анализируем исходный промпт
        original_tokens = optimizer.count_tokens(prompt)
        original_cost = optimizer.estimate_cost(original_tokens, 500)  # Предполагаем 500 токенов ответа
        
        print(f"Исходный промпт: {original_tokens} токенов")
        print(f"Примерная стоимость: ${original_cost:.4f}")
        
        # Оптимизируем промпт
        optimization_result = optimizer.compress_prompt(prompt, max_reduction=0.3)
        
        optimized_tokens = optimization_result["final_tokens"]
        optimized_cost = optimizer.estimate_cost(optimized_tokens, 500)
        
        print(f"\nОптимизированный промпт: {optimized_tokens} токенов")
        print(f"Примерная стоимость: ${optimized_cost:.4f}")
        print(f"Экономия: {optimization_result['reduction_percentage']:.1f}% токенов, ${original_cost - optimized_cost:.4f}")
        
        print(f"\n🔧 Применённые оптимизации:")
        for log_entry in optimization_result["optimization_log"]:
            print(f"  - {log_entry['description']}: -{log_entry['tokens_saved']} токенов")
        
        print(f"\n📄 Оптимизированный промпт:")
        print("-" * 40)
        print(optimization_result["optimized_prompt"][:300] + "...")
        print("-" * 40)
        
        if i < len(test_prompts):
            input("\nНажмите Enter для следующего примера...")
    
    # Демонстрация сжатия контекста
    print(f"\n📚 ДЕМОНСТРАЦИЯ СЖАТИЯ КОНТЕКСТА")
    print("=" * 40)
    
    long_context = """
    Искусственный интеллект быстро развивается и меняет мир. Современные системы ИИ способны 
    выполнять задачи, которые раньше требовали человеческого интеллекта. Машинное обучение 
    стало основой многих технологий. Нейронные сети показывают впечатляющие результаты в 
    распознавании изображений. Обработка естественного языка достигла новых высот. 
    Автономные автомобили используют компьютерное зрение. Рекомендательные системы помогают 
    пользователям находить контент. Медицинская диагностика улучшается благодаря ИИ. 
    Финансовые алгоритмы обнаруживают мошенничество. Роботы становятся более автономными. 
    Голосовые ассистенты понимают команды лучше. Переводчики работают в реальном времени. 
    Творческий ИИ создаёт музыку и искусство. Этические вопросы требуют внимания. 
    Регулирование технологий необходимо. Будущее ИИ полно возможностей и вызовов.
    """ * 3  # Увеличиваем для демонстрации
    
    context_summary = optimizer.create_context_summary(long_context, target_tokens=200)
    
    print(f"Исходный контекст: {context_summary['original_tokens']} токенов")
    print(f"Сжатый контекст: {context_summary['summary_tokens']} токенов")
    print(f"Степень сжатия: {(1 - context_summary['compression_ratio']) * 100:.1f}%")
    print(f"\nСжатый контекст:\n{context_summary['summary']}")

if __name__ == "__main__":
    demo_token_optimization()
```

## Практические задания

### 🟢 Базовый уровень

**Задание 1: Простой LLM-судья**
Создайте судью для оценки качества переводов. Критерии: точность перевода, сохранение смысла, естественность языка.

**Задание 2: Базовые валидаторы**
Реализуйте валидаторы для проверки:
- Наличия заключения в тексте
- Отсутствия нецензурной лексики
- Соответствия заданному стилю (формальный/неформальный)

### 🟡 Средний уровень

**Задание 3: Пайплайн с фильтрацией**
Создайте систему, которая:
- Генерирует несколько вариантов ответа
- Оценивает каждый LLM-судьёй
- Выбирает лучший для отправки пользователю

**Задание 4: Адаптивная оптимизация**
Разработайте систему, которая:
- Анализирует стоимость запросов
- Автоматически применяет оптимизацию при превышении бюджета
- Ведёт статистику экономии токенов

**Задание 5: A/B тестирование промптов**
Создайте инструмент для сравнения эффективности разных промптов с автоматической оценкой качества ответов.

### 🔴 Продвинутый уровень

**Задание 6: Многоуровневая система качества**
Разработайте систему с несколькими уровнями проверки:
- Автоматическая валидация
- LLM-судья для содержания
- Человеческая модерация для спорных случаев

**Задание 7: Интеллектуальное кэширование**
Создайте систему кэширования, которая:
- Определяет похожие запросы
- Адаптирует кэшированные ответы под новый контекст
- Оценивает качество адаптации

**Задание 8: Система мониторинга качества**
Постройте дашборд для отслеживания:
- Трендов качества ответов во времени
- Эффективности разных моделей
- ROI оптимизаций токенов

## Контрольные вопросы

1. **Что означает LLM-as-a-judge?**
   <details>
   <summary>Ответ</summary>
   LLM-as-a-judge — использование языковой модели для оценки качества ответов других моделей или контента. Судья-модель анализирует текст по заданным критериям и выставляет оценки, что позволяет автоматизировать процесс контроля качества с высокой скоростью и консистентностью.
   </details>

2. **Какие критерии качества ответа можно задать?**
   <details>
   <summary>Ответ</summary>
   Основные критерии: точность (фактическая корректность), полнота (покрытие всех аспектов), ясность (понятность изложения), релевантность (соответствие вопросу), структурированность, стиль, оригинальность. Критерии должны быть адаптированы под конкретную задачу и аудиторию.
   </details>

3. **Как можно уменьшить количество токенов в запросе?**
   <details>
   <summary>Ответ</summary>
   Методы оптимизации: удаление избыточных фраз, сокращение примеров, упрощение инструкций, сжатие форматирования, резюмирование длинного контекста, использование аббревиатур, удаление повторов, структурирование информации более компактно.
   </details>

4. **В чём преимущества автоматической валидации перед ручной?**
   <details>
   <summary>Ответ</summary>
   Скорость (секунды vs минуты), стоимость (в разы дешевле), масштабируемость (тысячи проверок в день), консистентность (нет человеческого фактора), круглосуточная работа, автоматическое логирование. Недостаток: может пропустить нюансы, понятные человеку.
   </details>

5. **Когда стоит использовать итеративное улучшение ответов?**
   <details>
   <summary>Ответ</summary>
   Для критически важных задач, когда качество важнее скорости и стоимости. Примеры: медицинские консультации, юридические документы, образовательный контент, техническая документация. Не стоит использовать для простых вопросов или развлекательного контента.
   </details>

## Заключение урока

### Что мы изучили

В финальном уроке модуля вы освоили профессиональные методы контроля качества AI-систем:

- **LLM-as-a-judge**: научились использовать модели для оценки качества других моделей
- **Пайплайны валидации**: создали системы автоматической проверки контента
- **Критерии качества**: определили и применили метрики для разных типов задач
- **Оптимизация токенов**: освоили методы снижения стоимости без потери качества

### Связь с предыдущими уроками модуля

Теперь у вас есть полный арсенал для создания продуктовых AI-решений:
- **Урок 1**: Продвинутые промпты → **Урок 3**: Контроль их качества
- **Урок 2**: Выбор модели → **Урок 3**: Оптимизация стоимости
- **Комбинирование**: Сложные стратегии + Мультимодельность + Контроль качества = Надёжная система

### Связь с модулем 1

От экспериментов к продакшену:
- **Модуль 1**: Основы работы с LLM и базовые навыки
- **Модуль 2**: Продвинутые техники и профессиональные инструменты
- **Результат**: Готовность создавать коммерческие AI-приложения

### Ваши достижения в модуле 2

🏆 **Выдающиеся результаты!** За три урока вы стали экспертом по продвинутой работе с LLM:

**Технические навыки:**
- ✅ Создаёте сложные многоступенчатые промпты (ReAct, Tree-of-Thought)
- ✅ Строите многомодельные системы с автоматическим роутингом
- ✅ Реализуете пайплайны контроля качества и валидации
- ✅ Оптимизируете стоимость и производительность систем

**Продуктовое мышление:**
- ✅ Выбираете оптимальные модели под бизнес-задачи
- ✅ Планируете бюджеты и контролируете расходы
- ✅ Обеспечиваете стабильное качество в продакшене
- ✅ Создаёте надёжные системы с fallback стратегиями

### Что дальше

Вы готовы к следующим шагам в AI-разработке:
- **RAG системы** — работа с внешними базами знаний
- **Fine-tuning** — дообучение моделей под специфические задачи
- **Agent системы** — создание автономных AI-агентов
- **Мультимодальность** — работа с текстом, изображениями, аудио

**Рекомендации для развития:**
1. Практикуйтесь на реальных проектах
2. Изучайте новые модели и техники  
3. Участвуйте в AI-сообществах
4. Следите за исследованиями в области LLM

🎉 **Поздравляем с завершением модуля!** Вы теперь обладаете профессиональными навыками разработки AI-приложений и готовы создавать инновационные решения, которые изменят мир!

---

## Дополнительные материалы

### Исследования по оценке качества:
- [Constitutional AI](https://arxiv.org/abs/2212.08073) — самоконтроль качества моделей
- [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) — исследования по автоматической оценке

### Инструменты для продакшена:
- [LangSmith](https://smith.langchain.com/) — мониторинг и отладка LLM-приложений  
- [Weights & Biases](https://wandb.ai/) — трекинг экспериментов и метрик
- [OpenAI Evals](https://github.com/openai/evals) — фреймворк для оценки моделей

### Оптимизация и производительность:
- [Guidance](https://github.com/guidance-ai/guidance) — эффективное управление генерацией
- [Token counting tools](https://platform.openai.com/tokenizer) — точный подсчёт токенов
- [Cost optimization strategies](https://help.openai.com/en/articles/6643408-managing-costs) — стратегии экономии

### Этика и безопасность AI:
- [AI Safety Guidelines](https://www.anthropic.com/index/introducing-constitutional-ai) — принципы безопасного ИИ
- [Responsible AI practices](https://ai.google/responsibility/responsible-ai-practices/) — ответственная разработка
