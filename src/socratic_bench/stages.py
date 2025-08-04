import random
from typing import List, Tuple

from socratic_bench.agents import Judge, LLMProcessingFailure, ConversationSeeder, Student, Teacher
from socratic_bench.pipeline import Stage, Emitter
from socratic_bench.schemas import Record, Message, ChatHistory


class SeedStage(Stage[Record, Record]):
    """
    Pipeline stage that seeds a record with an initial student question and main topics using a conversation seeder.
    """

    def __init__(self, seeder: ConversationSeeder):
        """
        Initializes the stage with a conversation seeder.

        Args:
            seeder: An instance of a ConversationSeeder used to generate questions and topics.
        """
        self._seeder = seeder
        self._num_interactions = len(self._seeder.interaction_types())

    def counters(self) -> Tuple[str, ...]:
        """
        Returns the set of counter names used during the seeding process.

        Counters:
            - seed.in: A sample entered the stage.
            - seed.missing: Source content was missing.
            - seed.failure: The seeder failed to generate output.
            - seed.out: A sample was emitted from the stage.
        """
        return "seed.in", "seed.missing", "seed.failure", "seed.out"

    def process(self, sample: Record, emitter: Emitter[Record]) -> None:
        """
        Processes a record by attempting to generate a seed question and topic from source content.

        Args:
            sample: The input record containing source content.
            emitter: The emitter used to emit processed records and increment counters.
        """
        interaction_type = random.choice(self._seeder.interaction_types())

        emitter.increment("seed.in")

        sample.metadata.seed_llm = self._seeder.seed_llm().model_name
        sample.seed.interaction_type = interaction_type["interaction_type"]
        source_content = sample.seed.source_content

        question, topics = None, None

        if source_content is None:
            sample.failure = True
            sample.failure_reason = "failed_seed / missing_source_content"
            emitter.increment("seed.missing")
        else:
            try:
                question, topics = self._seeder.gen_seed(source_content, **interaction_type)
            except LLMProcessingFailure as e:
                emitter.increment("seed.failure")
                sample.failure = True
                sample.failure_reason = f"failed_seed / {repr(e)}"

        sample.seed.question = question
        sample.seed.main_topics = topics

        emitter.emit(sample)

        emitter.increment("seed.out")


class ChatStage(Stage[List[Record], List[Record]]):
    """
    Pipeline stage that simulates a Socratic conversation between a teacher and student.

    Each record undergoes up to `max_interactions` message exchanges between the teacher and student agents.
    """

    def __init__(self, student: Student, teacher: Teacher, max_interactions: int = 16):
        """
        Initializes the stage with student and teacher agents.

        Args:
            student: The simulated student agent.
            teacher: The simulated teacher agent.
            max_interactions: Maximum number of message exchanges in the chat (default: 16).
        """
        self._student = student
        self._teacher = teacher
        self._max_interactions = max_interactions

    def counters(self) -> Tuple[str, ...]:
        """
        Returns the set of counter names used during the chat simulation.

        Counters:
            - chat_stage.eligible: Record has a valid seed and can begin chat.
            - chat_stage.failure: An LLM failed during the conversation.
            - chat_stage.success: Number of records completed without failure.
        """
        return "chat_stage.eligible", "chat_stage.failure", "chat_stage.success"

    def process(self, sample: List[Record], emitter: Emitter[List[Record]]) -> None:
        """
        Simulates full conversations between the teacher and student for all eligible records in the batch.

        Args:
            sample: A list of records, each with a seed.
            emitter: The emitter used to emit processed records and increment counters.
        """
        for s in filter(lambda r: r.has_seed(), sample):
            chat_history = ChatHistory(
                root=[
                    Message(
                        role="Student",
                        content=s.seed.question,  # type: ignore[arg-type] # content guaranteed from has_seed
                        end=False
                    )
                ]
            )
            s.chat_history = chat_history
            s.metadata.max_interactions = self._max_interactions
            s.metadata.student_llm = self._student.llm().model_name
            s.metadata.teacher_llm = self._teacher.llm().model_name
            s.student_type = random.choice(self._student.student_types())
            emitter.increment("chat_stage.eligible")

        for i in range(self._max_interactions):
            for s in filter(lambda r: not r.failure and r.chat_history and not r.chat_history.has_finished(), sample):
                try:
                    teacher_reply = self._teacher.query(str(s.chat_history))
                except LLMProcessingFailure as e:
                    s.failure = True
                    s.failure_reason = f"failed_teacher / {repr(e)}"
                    emitter.increment("chat_stage.failure")
                    continue
                s.chat_history.root.append(Message(role="Teacher", content=teacher_reply, end=False))

            for s in filter(lambda r: not r.failure and r.chat_history and not r.chat_history.has_finished(), sample):
                try:
                    student_reply, end = self._student.query(
                        str(s.chat_history),
                        student_type=s.student_type,
                        main_topics=s.seed.main_topics
                    )
                except LLMProcessingFailure as e:
                    s.failure = True
                    s.failure_reason = f"failed_student / {repr(e)}"
                    emitter.increment("chat_stage.failure")
                    continue

                s.chat_history.root.append(Message(role="Student", content=student_reply, end=end))

        emitter.increment("chat_stage.success", len(list(filter(lambda _s: not _s.failure, sample))))

        emitter.emit(sample)


class EvaluationStage(Stage[Record, Record]):
    """
    Pipeline stage that evaluates completed conversations using a judge agent to determine success.

    Adds structured feedback and pass/fail assessment to each record.
    """

    def __init__(self, judge: Judge):
        """
        Initializes the stage with a judge for evaluation.

        Args:
            judge: The evaluation agent used to score the Socratic interactions.
        """
        self._judge = judge

    def counters(self) -> Tuple[str, ...]:
        """
        Returns the set of counter names used during the evaluation stage.

        Counters:
            - judge.in: A sample entered the evaluation stage.
            - judge.accepted: The judge marked the sample as successful.
            - judge.rejected: The judge marked the sample as unsuccessful.
            - judge.failed_evaluation: Evaluation failed or returned an undecidable result.
            - judge.out: A sample was emitted from the stage.
        """
        return "judge.in", "judge.accepted", "judge.rejected", "judge.failed_evaluation", "judge.out"

    def process(self, sample: Record, emitter: Emitter[Record]) -> None:
        """
        Evaluates the sample using the judge if the conversation has finished and no prior failure occurred.

        Adds feedback and pass/fail status to the record.

        Args:
            sample: The record to evaluate.
            emitter: The emitter used to emit the result and update counters.
        """
        sample.metadata.judge_llm = self._judge.llm().model_name
        emitter.increment("judge.in")
        if not sample.failure and sample.has_seed() and sample.chat_history and sample.chat_history.has_finished():
            reason, assessment = self._judge.evaluate(
                sample.seed.main_topics,  # type: ignore[arg-type] # has_seed guarantees main_topics
                str(sample.chat_history)
            )

            if assessment:
                emitter.increment("judge.accepted")
            elif assessment is False:
                emitter.increment("judge.rejected")
            else:
                emitter.increment("judge.failed_evaluation")

            sample.feedback = reason
            sample.assessment = assessment

            if assessment is None:
                sample.failure = True
                sample.failure_reason = "failed_evaluation / failed parsing assessment"

        emitter.emit(sample)
        emitter.increment("judge.out")
