import abc
from typing import TypeVar, Generic, Optional, Dict, Generator, List, Type, Tuple, Any

T = TypeVar('T')  # Input or current type
U = TypeVar('U')  # Output or next type


class Emitter(Generic[T]):

    def __init__(
            self,
            stage: Optional['Stage[T, U]'],
            next_emitter: Optional['Emitter[U]'],
            tracker: Dict[str, int]
    ):

        if (stage is None) != (next_emitter is None):
            raise ValueError("Must specify both next_stage and next_emitter or neither.")

        self._stage = stage
        self._next_emitter = next_emitter
        self._tracker = tracker

    def emit(self, sample: T) -> None:
        if self._stage and self._next_emitter:
            self._stage.process(sample, self._next_emitter)
            self._stage.cleanup(self._next_emitter)

    def increment(self, name: str, value: int = 1) -> None:
        self._tracker[name] = self._tracker.get(name, 0) + value


class Stage(Generic[T, U], abc.ABC):

    @abc.abstractmethod
    def process(self, sample: T, emitter: Emitter[U]) -> None:
        ...

    def cleanup(self, emitter: Emitter[U]) -> None:
        ...


class DataSource(Generic[T], abc.ABC):

    @abc.abstractmethod
    def read(self) -> Generator[T, None, None]:
        ...


class BufferStage(Stage[T, List[T]]):

    def __init__(self) -> None:
        self._buffer: List[T] = []

    def process(self, sample: T, emitter: Emitter[List[T]]) -> None:
        self._buffer.append(sample)

    def cleanup(self, emitter: Emitter[List[T]]) -> None:
        emitter.emit(self._buffer)
        self._buffer.clear()


class FlattenStage(Stage[List[T], T]):

    def process(self, sample: List[T], emitter: Emitter[T]) -> None:
        for item in sample:
            emitter.emit(item)


class CollectSink(Stage[T, None]):

    def __init__(self) -> None:
        self.items: list[T] = []

    def process(self, sample: T, emitter: Emitter[None]) -> None:
        self.items.append(sample)


class PipelineStep(Generic[T, U]):
    def __init__(self, previous: Optional['PipelineStep[Any, T]'], stage: Optional[Stage[T, U]]):
        self.previous = previous
        self.stage = stage


class SocraticBench(Generic[T]):

    def __init__(self, source: DataSource[T], step: Optional[PipelineStep[T, U]] = None):
        self._source = source
        self._last_step = step

    @classmethod
    def from_data(cls: Type["SocraticBench[T]"], source: DataSource[T]) -> "SocraticBench[T]":
        return cls(source)

    def apply(self, stage: Stage[T, U]) -> 'SocraticBench[U]':
        last_step = PipelineStep(self._last_step, stage)
        return SocraticBench(self._source, last_step)

    def batch(self: 'SocraticBench[T]') -> 'SocraticBench[List[T]]':
        buffered: BufferStage[T] = BufferStage()
        return self.apply(buffered)

    def flatten(self: 'SocraticBench[List[U]]') -> 'SocraticBench[U]':
        flattened: FlattenStage[U] = FlattenStage()
        return self.apply(flattened)

    def run(self) -> Tuple[List[T], Dict[str, int]]:
        tracker: Dict[str, int] = {}
        sink: CollectSink[T] = CollectSink()
        terminal: Emitter[None] = Emitter(None, None, tracker)
        final_sink: Emitter[T] = Emitter(sink, terminal, tracker)
        current_emitter = final_sink

        step = self._last_step
        while step and step.stage is not None:
            current_emitter = Emitter(step.stage, current_emitter, tracker)
            step = step.previous

        for sample in self._source.read():
            current_emitter.emit(sample)

        return sink.items, tracker
