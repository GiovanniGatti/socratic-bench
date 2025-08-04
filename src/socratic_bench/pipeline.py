import abc
from typing import TypeVar, Generic, Optional, Dict, Generator, List, Tuple, Any, Type

T = TypeVar('T')  # Input or current type
U = TypeVar('U')  # Output or next type


class Emitter(Generic[T]):
    """
    Handles the forwarding of processed data between stages in a pipeline and maintains counters for metrics.
    """

    def __init__(
            self,
            stage: Optional['Stage[T, U]'],
            next_emitter: Optional['Emitter[U]'],
            tracker: Dict[str, int]
    ):
        """
        Initializes the emitter with an optional stage and next emitter in the pipeline.

        Raises:
            ValueError: If only one of `stage` or `next_emitter` is provided.
            ValueError: If any counter declared by the stage is already present in the tracker.
        """
        if (stage is None) != (next_emitter is None):
            raise ValueError("Must specify both next_stage and next_emitter or neither.")

        self._stage = stage
        self._next_emitter = next_emitter
        self._tracker = tracker

        if stage:
            for c in stage.counters():
                if c in self._tracker:
                    raise ValueError(f"counter {c} already declared")
                self._tracker[c] = 0

    def emit(self, sample: T) -> None:
        """
        Processes and forwards a sample to the next emitter via the associated stage.
        """
        if self._stage and self._next_emitter:
            self._stage.process(sample, self._next_emitter)
            self._stage.cleanup(self._next_emitter)

    def increment(self, name: str, value: int = 1) -> None:
        """
        Increments a named counter by the specified value.

        Raises:
            ValueError: If the counter name is not declared.
        """
        if name not in self._tracker:
            raise ValueError(f"undeclared counter {name}")
        self._tracker[name] = self._tracker[name] + value


class Stage(Generic[T, U], abc.ABC):
    """
    Abstract base class representing a single transformation stage in the pipeline.
    """

    @abc.abstractmethod
    def process(self, sample: T, emitter: Emitter[U]) -> None:
        """
        Processes a single input sample and emits zero or more output samples.

        To be implemented by subclasses.
        """
        ...

    def cleanup(self, emitter: Emitter[U]) -> None:
        """
        Called after all samples are processed, allowing the stage to emit final output or perform cleanup.
        """
        ...

    def counters(self) -> Tuple[str, ...]:
        """
        Returns a tuple of counter names used by the stage.

        Defaults to an empty tuple.
        """
        return ()


class DataSource(Generic[T], abc.ABC):
    """
    Abstract base class for data sources that generate input samples for the pipeline.
    """

    @abc.abstractmethod
    def read(self) -> Generator[T, None, None]:
        """
        Yields input samples for the pipeline to process.

        To be implemented by subclasses.
        """
        ...


class BufferStage(Stage[T, List[T]]):
    """
    Collects incoming samples into a list and emits the batch during cleanup.
    """

    def __init__(self) -> None:
        """
        Initializes the internal buffer.
        """
        self._buffer: List[T] = []

    def process(self, sample: T, emitter: Emitter[List[T]]) -> None:
        """
        Appends the input sample to the internal buffer.
        """
        self._buffer.append(sample)

    def cleanup(self, emitter: Emitter[List[T]]) -> None:
        """
        Emits the collected buffer as a list and clears the internal buffer.
        """
        emitter.emit(self._buffer)
        self._buffer.clear()


class FlattenStage(Stage[List[T], T]):
    """
    Takes a list of samples and emits each item individually.
    """

    def process(self, sample: List[T], emitter: Emitter[T]) -> None:
        """
        Emits each item in the input list separately.
        """
        for item in sample:
            emitter.emit(item)


class CollectSink(Stage[T, None]):
    """
    Terminal stage that collects all emitted samples into a list.
    """

    def __init__(self) -> None:
        """
        Initializes the internal list for collecting samples.
        """
        self.items: list[T] = []

    def process(self, sample: T, emitter: Emitter[None]) -> None:
        """
        Appends the sample to the internal collection list.
        """
        self.items.append(sample)


class Identity(Stage[T, T]):
    """
    Stage that passes input samples through unchanged.
    """

    def process(self, sample: T, emitter: Emitter[T]) -> None:
        """
        Emits the input sample without modification.
        """
        emitter.emit(sample)


class PipelineStep(Generic[T, U]):
    """
    Represents a node in the pipeline construction chain, containing a stage and reference to the previous step.
    """
    def __init__(self, previous: Optional['PipelineStep[Any, T]'], stage: Optional[Stage[T, U]]):
        """
        Initializes a pipeline step with a reference to a previous step and a stage.
        """
        self.previous = previous
        self.stage = stage


IN = TypeVar('IN')  # Input type for the entire pipeline
OUT = TypeVar('OUT')  # Output type for the entire pipeline


class SocraticBench(Generic[IN, OUT]):
    """
    Main class for building and running a data processing pipeline composed of stages.
    """

    def __init__(self, source: DataSource[IN], step: Optional[PipelineStep[Any, OUT]] = None):
        """
        Initializes the pipeline with a data source and optionally a final pipeline step.
        """
        self._source = source
        self._last_step = step

    @classmethod
    def from_data(cls: Type["SocraticBench[IN, IN]"], source: DataSource[IN]) -> "SocraticBench[IN, IN]":
        """
        Creates a new pipeline starting with an identity stage, preserving input samples as-is.
        """
        step = PipelineStep[IN, IN](previous=None, stage=Identity[IN]())
        return cls(source, step)

    def apply(self: "SocraticBench[IN, T]", stage: Stage[T, U]) -> "SocraticBench[IN, U]":
        """
        Adds a new stage to the pipeline and returns the updated pipeline.
        """
        last_step = PipelineStep[T, U](self._last_step, stage)
        return SocraticBench[IN, U](self._source, last_step)

    def batch(self: "SocraticBench[IN, T]") -> "SocraticBench[IN, List[T]]":
        """
        Adds a buffering stage to collect samples into a batch before further processing.
        """
        buffered: BufferStage[T] = BufferStage()
        return self.apply(buffered)

    def flatten(self: "SocraticBench[IN, List[T]]") -> "SocraticBench[IN, T]":
        """
        Adds a stage that flattens batches back into individual samples.
        """
        flattened = FlattenStage[T]()
        return self.apply(flattened)

    def run(self) -> Tuple[List[OUT], Dict[str, int]]:
        """
        Executes the pipeline and returns the collected output and counter statistics.

        Returns:
            A tuple containing the list of output items and a dictionary of counter values.
        """
        tracker: Dict[str, int] = {}
        sink: CollectSink[OUT] = CollectSink()
        terminal: Emitter[None] = Emitter(None, None, tracker)
        final_sink: Emitter[OUT] = Emitter(sink, terminal, tracker)
        current_emitter: Emitter[Any] = final_sink

        step = self._last_step
        while step and step.stage is not None:
            current_emitter = Emitter(step.stage, current_emitter, tracker)
            step = step.previous

        for sample in self._source.read():
            current_emitter.emit(sample)

        return sink.items, tracker
