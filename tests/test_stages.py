import random
import unittest
from typing import List, Dict, TypeVar
from unittest import mock

import pytest

from socratic_bench import Record
from socratic_bench.agents import LLM, LLMProcessingFailure
from socratic_bench.pipeline import CollectSink, Emitter, Stage
from socratic_bench.schemas import ChatHistory, Message
from socratic_bench.stages import SeedStage, ChatStage, EvaluationStage

T = TypeVar('T')  # Input or current type
U = TypeVar('U')  # Output or next type


class AnyLLM(LLM):

    def __init__(self, model_name: str = "any-llm"):
        self._model_name = model_name

    def query(self, messages: List[Dict[str, str]]) -> str:
        return ""

    def healthcheck(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return self._model_name


def any_record_with_seed() -> Record:
    record = Record(id=random.randint(0, 100))
    record.seed.source_content = "Some source content..."
    record.seed.main_topics = "- A list\n- of\n- main topics"
    record.seed.question = "What's up?!"
    record.seed.interaction_type = "test interaction type"
    assert record.has_seed()
    return record


def any_record_with_completed_chat() -> Record:
    record = any_record_with_seed()
    record.chat_history = ChatHistory(root=[
        Message(role="Student", content="I have a question.", end=False),
        Message(
            role="Teacher",
            content="And I have a cristal ball. Why do you think the Rayleigh scattering makes the sky blue?",
            end=False
        ),
        Message(
            role="Student",
            content="By the deflection of light, or other electromagnetic radiation, by particles with a size much "
                    "smaller than the wavelength of the radiation.",
            end=True
        ),
    ])
    assert record.chat_history.has_finished()
    return record


def wrap_stage(stage: Stage[T, U], sink: CollectSink[U], tracker: Dict[str, int]) -> Emitter[T]:
    terminal: Emitter[None] = Emitter(None, None, tracker)
    sink_emitter = Emitter[Record](sink, terminal, tracker)
    return Emitter(stage, sink_emitter, tracker)


class TestSeedStage:

    def test_success(self) -> None:
        seeder = mock.MagicMock()
        seeder.gen_seed.return_value = "why is the sky blue?", "- color\n-Rayleigh scattering"
        seeder.seed_llm.return_value = AnyLLM()
        seeder.interaction_types.return_value = (
            {
                "interaction_type": "an interaction type",
                "context": "a context",
                "main_topics": "some main topics",
                "question": "a question"
            },
        )
        stage = SeedStage(seeder)

        record = Record(id=0)
        record.seed.source_content = "Color of the sky..."
        sink = CollectSink[Record]()
        tracker: Dict[str, int] = {}
        emitter = wrap_stage(stage, sink, tracker)
        emitter.emit(record)

        assert len(sink.items) == 1
        assert sink.items[0].seed.question == "why is the sky blue?"
        assert sink.items[0].seed.main_topics == "- color\n-Rayleigh scattering"
        assert sink.items[0].metadata.seed_llm == "any-llm"
        assert sink.items[0].seed.interaction_type == "an interaction type"
        assert tracker["seed.in"] == 1
        assert tracker["seed.out"] == 1

    def test_missing_source_content(self) -> None:
        seeder = mock.MagicMock()
        seeder.seed_llm.return_value = AnyLLM()
        seeder.interaction_types.return_value = (
            {
                "interaction_type": "an interaction type",
                "context": "a context",
                "main_topics": "some main topics",
                "question": "a question"
            },
        )
        stage = SeedStage(seeder)

        record = Record(id=0)
        record.seed.source_content = None  # not set
        sink = CollectSink[Record]()
        tracker: Dict[str, int] = {}
        emitter = wrap_stage(stage, sink, tracker)
        emitter.emit(record)

        assert len(sink.items) == 1
        assert sink.items[0].seed.question is None
        assert sink.items[0].seed.main_topics is None
        assert sink.items[0].metadata.seed_llm == "any-llm"
        assert sink.items[0].failure
        assert sink.items[0].failure_reason == "failed_seed / missing_source_content"
        assert tracker["seed.in"] == 1
        assert tracker["seed.out"] == 1
        assert tracker["seed.missing"] == 1

    def test_llm_processing_failure(self) -> None:
        seeder = mock.MagicMock()
        seeder.gen_seed.side_effect = LLMProcessingFailure("Boom!")
        seeder.seed_llm.return_value = AnyLLM()
        seeder.interaction_types.return_value = (
            {
                "interaction_type": "an interaction type",
                "context": "a context",
                "main_topics": "some main topics",
                "question": "a question"
            },
        )
        stage = SeedStage(seeder)

        record = Record(id=0)
        record.seed.source_content = "Color of the sky..."
        sink = CollectSink[Record]()
        tracker: Dict[str, int] = {}
        emitter = wrap_stage(stage, sink, tracker)
        emitter.emit(record)

        assert len(sink.items) == 1
        assert sink.items[0].seed.question is None
        assert sink.items[0].seed.main_topics is None
        assert sink.items[0].metadata.seed_llm == "any-llm"
        assert sink.items[0].seed.interaction_type == "an interaction type"
        assert sink.items[0].failure
        assert sink.items[0].failure_reason == "failed_seed / Boom!"
        assert tracker["seed.in"] == 1
        assert tracker["seed.out"] == 1
        assert tracker["seed.failure"] == 1


class TestChatStage:

    def test_success(self) -> None:
        record = any_record_with_seed()
        record.seed.question = "Hello, I have a question about the color of the sky?"
        record.seed.main_topics = "- The color of the sky"

        teacher = mock.MagicMock()
        teacher.llm.return_value = AnyLLM(model_name="any-teacher")
        teacher.query.return_value = "Hey, I can help you. The sky is blue because..."

        student = mock.MagicMock()
        student.llm.return_value = AnyLLM(model_name="any-student")
        student.student_types.return_value = ("a student type",)
        student.query.return_value = "...Rayleigh scattering?", True

        stage = ChatStage(student, teacher, max_interactions=3)

        sink = CollectSink[Record]()
        tracker: Dict[str, int] = {}
        emitter = wrap_stage(stage, sink, tracker)
        emitter.emit([record])

        assert len(sink.items) == 1
        output = sink.items[0]
        assert len(output) == 1

        teacher.query.assert_called_with("Student: Hello, I have a question about the color of the sky?")
        student.query.assert_called_with(
            "Student: Hello, I have a question about the color of the sky?"
            "\n"
            "Teacher: Hey, I can help you. The sky is blue because...",
            student_type="a student type",
            main_topics="- The color of the sky"
        )
        assert output[0].metadata.teacher_llm == "any-teacher"
        assert output[0].metadata.student_llm == "any-student"
        assert output[0].metadata.max_interactions == 3
        assert output[0].chat_history == ChatHistory(root=[
            Message(role="Student", content="Hello, I have a question about the color of the sky?", end=False),
            Message(role="Teacher", content="Hey, I can help you. The sky is blue because...", end=False),
            Message(role="Student", content="...Rayleigh scattering?", end=True),
        ])
        assert tracker["chat_stage.eligible"] == 1
        assert tracker["chat_stage.success"] == 1

    def test_teacher_failure(self) -> None:
        record = any_record_with_seed()
        record.seed.question = "Hello, I have a question about the color of the sky?"

        teacher = mock.MagicMock()
        teacher.llm.return_value = AnyLLM()
        teacher.query.side_effect = LLMProcessingFailure("whoops")

        student = mock.MagicMock()
        student.student_types.return_value = ("a student type",)

        stage = ChatStage(student, teacher)

        sink = CollectSink[Record]()
        tracker: Dict[str, int] = {}
        emitter = wrap_stage(stage, sink, tracker)
        emitter.emit([record])

        assert len(sink.items) == 1
        output = sink.items[0]
        assert len(output) == 1

        teacher.query.assert_called_with("Student: Hello, I have a question about the color of the sky?")
        assert output[0].chat_history == ChatHistory(root=[
            Message(role="Student", content="Hello, I have a question about the color of the sky?", end=False),
        ])
        assert output[0].failure
        assert output[0].failure_reason == "failed_teacher / whoops"
        assert tracker["chat_stage.eligible"] == 1
        assert tracker["chat_stage.success"] == 0

    def test_student_failure(self) -> None:
        record = any_record_with_seed()
        record.seed.question = "Hello, I have a question about the color of the sky?"
        record.seed.main_topics = "- The color of the sky"

        teacher = mock.MagicMock()
        teacher.llm.return_value = AnyLLM()
        teacher.query.return_value = "Hey, I can help you. The sky is blue because..."

        student = mock.MagicMock()
        student.llm.return_value = AnyLLM()
        student.student_types.return_value = ("a student type",)
        student.query.side_effect = LLMProcessingFailure("boom!")

        stage = ChatStage(student, teacher)

        sink = CollectSink[Record]()
        tracker: Dict[str, int] = {}
        emitter = wrap_stage(stage, sink, tracker)
        emitter.emit([record])

        assert len(sink.items) == 1
        output = sink.items[0]
        assert len(output) == 1

        teacher.query.assert_called_with("Student: Hello, I have a question about the color of the sky?")
        student.query.assert_called_with(
            "Student: Hello, I have a question about the color of the sky?"
            "\n"
            "Teacher: Hey, I can help you. The sky is blue because...",
            student_type="a student type",
            main_topics="- The color of the sky"
        )
        assert output[0].chat_history == ChatHistory(root=[
            Message(role="Student", content="Hello, I have a question about the color of the sky?", end=False),
            Message(role="Teacher", content="Hey, I can help you. The sky is blue because...", end=False),
        ])
        assert output[0].failure
        assert output[0].failure_reason == "failed_student / boom!"
        assert tracker["chat_stage.eligible"] == 1
        assert tracker["chat_stage.success"] == 0


class TestEvaluationStage:

    @pytest.mark.parametrize("summary,assessment", [
        ("Great Example", True),
        ("Bad Example", False)
    ])
    def test_success(self, summary: str, assessment: bool) -> None:
        record = any_record_with_seed()
        record.seed.main_topics = "- Color of the sky"
        record.chat_history = ChatHistory(root=[
            Message(role="Student", content="Why is the sky blue?", end=False),
            Message(
                role="Teacher",
                content="Why do you think the Rayleigh scattering makes the sky blue?",
                end=False
            ),
            Message(
                role="Student",
                content="By the deflection of light, or other electromagnetic radiation, by particles with a size much "
                        "smaller than the wavelength of the radiation.",
                end=True
            ),
        ])

        judge = mock.MagicMock()
        judge.llm.return_value = AnyLLM(model_name="any-judge")
        judge.evaluate.return_value = summary, assessment

        stage = EvaluationStage(judge)

        sink = CollectSink[Record]()
        tracker: Dict[str, int] = {}
        emitter = wrap_stage(stage, sink, tracker)
        emitter.emit(record)

        assert len(sink.items) == 1

        judge.evaluate.assert_called_with(
            "- Color of the sky",
            "Student: Why is the sky blue?\n"
            "Teacher: Why do you think the Rayleigh scattering makes the sky blue?\n"
            "Student: By the deflection of light, or other electromagnetic radiation, by particles with a size much "
            "smaller than the wavelength of the radiation."
        )
        assert sink.items[0].metadata.judge_llm == "any-judge"
        assert tracker["judge.in"] == 1
        assert tracker["judge.out"] == 1
        assert tracker["judge.accepted"] == (1 if assessment else 0)
        assert tracker["judge.rejected"] == (1 if not assessment else 0)

    def test_judge_failure(self) -> None:
        record = any_record_with_completed_chat()

        judge = mock.MagicMock()
        judge.llm.return_value = AnyLLM()
        judge.evaluate.return_value = "answer that cannot be parsed", None

        stage = EvaluationStage(judge)

        sink = CollectSink[Record]()
        tracker: Dict[str, int] = {}
        emitter = wrap_stage(stage, sink, tracker)
        emitter.emit(record)

        assert len(sink.items) == 1

        assert tracker["judge.in"] == 1
        assert tracker["judge.out"] == 1
        assert tracker["judge.failed_evaluation"] == 1
        assert sink.items[0].failure
        assert sink.items[0].failure_reason == "failed_evaluation / failed parsing assessment"
        assert sink.items[0].feedback == "answer that cannot be parsed"
        assert sink.items[0].assessment is None


if __name__ == "__main__":
    unittest.main()
