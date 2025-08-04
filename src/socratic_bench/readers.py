from typing import Generic, Type, Generator, TypeVar

from datasets import load_dataset  # type: ignore[attr-defined] # missing stubs from HF

from socratic_bench.pipeline import DataSource
from socratic_bench.schemas import Seed

T = TypeVar('T')


class PrincetonChapters(Generic[T], DataSource[T]):
    """
    A data source that yields chapter-based records from the Princeton TextbookChapters dataset.

    This class loads, shuffles, and slices textbook chapters, converting them into user-defined record objects for use
    in downstream pipelines.
    """

    def __init__(
            self,
            record_cls: Type[T],
            num_conversations: int,
            max_chapter_size: int = 3000,
    ):
        """
        Initializes the data source with the record type, number of records to yield, and maximum chapter size.

        Args:
            record_cls: The class used to wrap each chapter into a data record. Must accept `id` and `seed` arguments.
            num_conversations: The number of records to yield from the dataset.
            max_chapter_size: Maximum number of characters to include from each chapter (default is 3000).
        """
        self._record_cls = record_cls
        self._num_conversations = num_conversations
        self._max_chapter_size = max_chapter_size

    def read(self) -> Generator[T, None, None]:
        """
        Yields a sequence of record instances constructed from shuffled textbook chapters.

        Each record contains the truncated chapter content wrapped in a `Seed` object, passed to `record_cls`.

        Yields:
            Instances of type `T`, each representing a textbook chapter.
        """
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
