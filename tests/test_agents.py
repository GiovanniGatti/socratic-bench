import random
import unittest
from unittest import mock

import pytest

from socratic_bench.agents import ConversationSeederAgent, LLMProcessingFailure, StudentAgent, JudgeAgent


class TestConversationSeederAgent:

    def test_seed_gen(self) -> None:
        llm = mock.MagicMock()
        llm.query.return_value = " [MAIN_TOPICS]- light\n- earth\n- love[\MAIN_TOPICS]What's up!?[\QUESTION]\n"

        agent = ConversationSeederAgent(llm)

        question, topics = agent.gen_seed("The sky is blue", **random.choice(agent.interaction_types()))

        assert question == "What's up!?"
        assert topics == "- light\n- earth\n- love"

    def test_failed_seed_gen(self) -> None:
        llm = mock.MagicMock()
        llm.query.return_value = "[MAIN_TOPICS]- light\n- earth\n- love[\MAIN_TOPICS]What's up!?[\Q]"

        agent = ConversationSeederAgent(llm)

        with pytest.raises(LLMProcessingFailure) as excinfo:
            agent.gen_seed("The sky is blue", **random.choice(agent.interaction_types()))

        assert "MAIN_TOPICS]- light\n- earth\n- love[\\MAIN_TOPICS]What's up!?[\\Q]" in str(excinfo.value)


class TestStudentAgent:

    @pytest.mark.parametrize("instruction,expected_end", [
        ("[CONTINUE]", False),
        ("[END]", True),
        ("\n[CONTINUE]", False),
        ("\n[END]", True),
        ("  [CONTINUE]", False),
        ("  [END]", True)
    ])
    def test_query(self, instruction: str, expected_end: bool) -> None:
        llm = mock.MagicMock()
        llm.query.return_value = "I have a problem, can you help me?" + instruction

        agent = StudentAgent(llm)

        reply, end = agent.query(
            "Student: Hey\nTeacher: Hello",
            main_topics="- topic 1\n- topic 2",
            student_type=random.choice(agent.student_types())
        )

        assert reply == "I have a problem, can you help me?"
        assert expected_end == end

    def test_failed_query(self) -> None:
        llm = mock.MagicMock()
        llm.query.return_value = "I have a problem, can you help me?"  # missing stopping condition

        agent = StudentAgent(llm)

        with pytest.raises(LLMProcessingFailure) as excinfo:
            agent.query(
                "Student: Hey\nTeacher: Hello",
                main_topics="- topic 1\n- topic 2",
                student_type=random.choice(agent.student_types())
            )

        assert "output: I have a problem, can you help me?" in str(excinfo.value)


class TestJudgeAgent:

    @pytest.mark.parametrize("expected_assessment", [
        True, False
    ])
    def test_query(self, expected_assessment: bool) -> None:
        llm = mock.MagicMock()
        llm.query.return_value = "And hence my decision. [ASSESSMENT]" + str(expected_assessment)

        agent = JudgeAgent(llm)

        result, assessment = agent.evaluate(
            chat_history="Student: Hey\nTeacher: Hello",
            main_topics="- topic 1\n- topic 2"
        )

        assert result == "And hence my decision."
        assert assessment == expected_assessment

    @pytest.mark.parametrize("end_str", [
        "[ASSESSMENT] ",  # missing value
        "[ASSESSMENT] F",
        "[ASSESSMENT] T",
        "True",
        "False",
        "[ASSESMENT] False",
    ])
    def test_failed_query(self, end_str: str) -> None:
        llm = mock.MagicMock()
        llm.query.return_value = "And hence my decision. " + end_str

        agent = JudgeAgent(llm)

        result, assessment = agent.evaluate(
            chat_history="Student: Hey\nTeacher: Hello",
            main_topics="- topic 1\n- topic 2"
        )

        assert "And hence my decision. " + end_str in str(result)
        assert assessment is None


if __name__ == "__main__":
    unittest.main()
