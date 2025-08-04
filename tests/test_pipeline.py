import unittest
from typing import List, Generator, Dict, Tuple

import pytest

from socratic_bench.pipeline import Emitter, Stage, SocraticBench, DataSource


class DummySource(DataSource[int]):
    def __init__(self, data: List[int]):
        self.data = data

    def read(self) -> Generator[int, None, None]:
        for x in self.data:
            yield x


def test_emitter_requires_both_or_none():
    with pytest.raises(ValueError):
        Emitter(stage=None, next_emitter=Emitter(None, None, {}), tracker={})


def test_emitter_increment():
    tracker: Dict[str, int] = {}
    tracker["count"] = 0
    emitter = Emitter(None, None, tracker)
    emitter.increment("count")
    emitter.increment("count", 2)
    assert tracker["count"] == 3


def test_pipeline_from_data():
    source = DummySource([1, 2, 3])
    bench = SocraticBench.from_data(source)
    output, tracker = bench.run()
    assert output == [1, 2, 3]


def test_pipeline_batch_and_flatten():
    source = DummySource([1, 2, 3])
    bench = SocraticBench.from_data(source).batch().flatten()
    output, tracker = bench.run()
    assert output == [1, 2, 3]


def test_pipeline_chained_apply():
    class Double(Stage[int, int]):
        def process(self, sample: int, emitter: Emitter[int]) -> None:
            emitter.emit(sample * 2)

    source = DummySource([1, 2])
    bench = SocraticBench.from_data(source).apply(Double())
    output, tracker = bench.run()
    assert output == [2, 4]


def test_pipeline_tracker_counts():
    class CountingStage(Stage[int, int]):
        def counters(self) -> Tuple[str, ...]:
            return ("processed",)

        def process(self, sample: int, emitter: Emitter[int]) -> None:
            emitter.increment("processed")
            emitter.emit(sample)

    source = DummySource([10, 20])
    bench = SocraticBench.from_data(source).apply(CountingStage())
    output, tracker = bench.run()
    assert output == [10, 20]
    assert tracker["processed"] == 2


if __name__ == "__main__":
    unittest.main()
