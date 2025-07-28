from typing import Optional

import pydantic


class Metadata(pydantic.BaseModel):
    seed_llm: Optional[str] = None
    student_llm: Optional[str] = None
    teacher_llm: Optional[str] = None
    judge_llm: Optional[str] = None
    max_interactions: Optional[int] = None


class Seed(pydantic.BaseModel):
    source_content: Optional[str] = None
    question: Optional[str] = None
    main_topics: Optional[str] = None
    interaction_type: Optional[str] = None


class Message(pydantic.BaseModel):
    role: str
    content: str
    end: bool

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"


class ChatHistory(pydantic.RootModel):
    root: list[Message]

    def __str__(self) -> str:
        return "\n".join(str(m) for m in self.root)

    def __len__(self) -> int:
        return len(self.root)

    def has_finished(self) -> bool:
        return self.root[-1].end


class Record(pydantic.BaseModel):
    metadata: Metadata = Metadata()
    id: int
    seed: Seed = Seed()
    student_type: Optional[str] = None
    chat_history: Optional[ChatHistory] = None
    feedback: Optional[str] = None
    assessment: Optional[bool] = None

    failure: bool = False
    failure_reason: Optional[str] = None

    def has_seed(self) -> bool:
        return self.seed.question is not None and self.seed.main_topics is not None
