import random
from typing import List, Any

from agents import Judge, LLMProcessingFailure, ConversationSeeder, Student, Teacher
from pipeline import Stage, Emitter
from schemas import Record, Message, ChatHistory


class SeedStage(Stage[str, Record]):

    def __init__(self, seeder: ConversationSeeder):
        self._seeder = seeder
        self._num_interactions = len(self._seeder.interaction_types())
        self._id = 0

    def process(self, sample: str, emitter: Emitter[Record, Record, Any]) -> None:
        interaction_type = random.choice(self._seeder.interaction_types())

        emitter.increment("seed.in")

        record = Record(id=self._id)
        record.metadata.seed_llm = self._seeder.seed_llm().model_name
        record.seed.interaction_type = interaction_type["interaction_type"]
        record.seed.source_content = sample

        question, topics = None, None
        try:
            question, topics = self._seeder.gen_seed(sample, **interaction_type)
        except LLMProcessingFailure as e:
            emitter.increment("seed.failure")
            record.failure = True
            record.failure_reason = f"failed_seed / {repr(e)}"

        record.seed.question = question
        record.seed.main_topics = topics

        emitter.emit(record)

        self._id += 1
        emitter.increment("seed.out")


class ChatStage(Stage[List[Record], List[Record]]):

    def __init__(self, student: Student, teacher: Teacher, max_interactions: int = 16):
        self._student = student
        self._teacher = teacher
        self._max_interactions = max_interactions

    def process(self, sample: List[Record], emitter: Emitter[List[Record], List[Record], Any]) -> None:
        for s in filter(lambda _s: _s.has_seed(), sample):
            s: Record
            chat_history = ChatHistory(root=[Message(role="Student", content=s.seed.question, end=False)])
            s.chat_history = chat_history
            s.metadata.max_interactions = self._max_interactions
            s.metadata.student_llm = self._student.llm().model_name
            s.metadata.teacher_llm = self._teacher.llm().model_name
            s.student_type = random.choice(self._student.student_types())
            emitter.increment("chat_stage.eligible")

        for i in range(self._max_interactions):
            for s in filter(lambda _s: not _s.failure and not _s.chat_history.has_finished(), sample):
                try:
                    teacher_reply = self._teacher.query(str(s.chat_history))
                except LLMProcessingFailure as e:
                    s.failure = True
                    s.failure_reason = f"failed_teacher / {repr(e)}"
                    emitter.increment("chat_stage.failure")
                    continue
                s.chat_history.root.append(Message(role="Teacher", content=teacher_reply, end=False))

            for s in filter(lambda _s: not _s.failure and not _s.chat_history.has_finished(), sample):
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

    def __init__(self, judge: Judge):
        self._judge = judge

    def process(self, sample: Record, emitter: Emitter[Record, Record, Any]) -> None:
        sample.metadata.judge_llm = self._judge.llm().model_name
        emitter.increment("judge.in")
        if not sample.failure and sample.has_seed() and sample.chat_history.has_finished():
            reason, assessment = self._judge.evaluate(sample.seed, sample.chat_history)

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
