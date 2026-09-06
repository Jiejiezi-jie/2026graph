from app.domain.models import QueryMode, QueryResult
from scripts.evaluate_novel_subset import (
    OFFICIAL_SYSTEM_PROMPT,
    build_prediction,
    select_balanced_questions,
)


QUESTION_TYPES = (
    "Fact Retrieval",
    "Complex Reasoning",
    "Contextual Summarize",
    "Creative Generation",
)


def make_question(index: int, question_type: str) -> dict:
    return {
        "id": f"q-{index}",
        "source": "Novel-4128",
        "question": f"Question {index}?",
        "answer": f"Answer {index}",
        "question_type": question_type,
        "evidence": [f"Evidence {index}"],
    }


def test_balanced_sample_takes_three_three_three_one() -> None:
    questions = []
    index = 0
    for question_type in QUESTION_TYPES:
        count = 1 if question_type == "Creative Generation" else 5
        for _ in range(count):
            questions.append(make_question(index, question_type))
            index += 1

    selected = select_balanced_questions(questions, 10)

    counts = {
        question_type: sum(
            item["question_type"] == question_type for item in selected
        )
        for question_type in QUESTION_TYPES
    }
    assert counts == {
        "Fact Retrieval": 3,
        "Complex Reasoning": 3,
        "Contextual Summarize": 3,
        "Creative Generation": 1,
    }


def test_prediction_matches_official_evaluator_shape() -> None:
    question = make_question(1, "Fact Retrieval")
    result = QueryResult(
        query=question["question"],
        mode=QueryMode.MIX,
        answer="Generated answer",
        latency_ms=10,
        chunks=[
            {"content": "First context"},
            {"content": "Second context"},
            {"other": "ignored"},
        ],
    )

    prediction = build_prediction(question, result)

    assert prediction == {
        "id": "q-1",
        "question": "Question 1?",
        "source": "Novel-4128",
        "context": ["First context", "Second context"],
        "evidence": ["Evidence 1"],
        "question_type": "Fact Retrieval",
        "generated_answer": "Generated answer",
        "ground_truth": "Answer 1",
    }


def test_benchmark_prompt_uses_only_supported_lightrag_placeholders() -> None:
    rendered = OFFICIAL_SYSTEM_PROMPT.format(
        response_type="Multiple Paragraphs",
        user_prompt="",
        context_data="retrieved context",
    )

    assert "retrieved context" in rendered
