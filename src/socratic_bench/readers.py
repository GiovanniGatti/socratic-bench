from typing import Generic, Type, Generator, TypeVar

from datasets import load_dataset  # type: ignore[attr-defined] # missing stubs from HF

from socratic_bench.pipeline import DataSource
from socratic_bench.schemas import Seed

T = TypeVar('T')


class PrincetonChapters(Generic[T], DataSource[T]):

    def __init__(
            self,
            record_cls: Type[T],
            num_conversations: int,
            max_chapter_size: int = 3000,
    ):
        self._record_cls = record_cls
        self._num_conversations = num_conversations
        self._max_chapter_size = max_chapter_size

    def read(self) -> Generator[T, None, None]:
        chapters = load_dataset("princeton-nlp/TextbookChapters", split="train")
        chapters = chapters.shuffle()

        i = 0
        while i < self._num_conversations and i < len(chapters):
            page = chapters[i]
            text: str = page["chapter"][:self._max_chapter_size]

            record = self._record_cls(  # type: ignore[call-arg]  # keeping lightweight record instantiation
                id=i,
                seed=Seed(source_content=text),
            )
            yield record
            i += 1
