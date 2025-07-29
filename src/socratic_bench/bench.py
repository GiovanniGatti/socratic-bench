from socratic_bench.agents import Student, Teacher, Judge, ConversationSeeder
from socratic_bench.pipeline import SocraticBench, DataSource
from socratic_bench.schemas import Record
from socratic_bench.stages import SeedStage, ChatStage, EvaluationStage


def socratic_bench(
        data_source: DataSource[Record],
        seeder: ConversationSeeder,
        student: Student,
        teacher: Teacher,
        judge: Judge,
        max_interactions: int = 16,
) -> SocraticBench[Record, Record]:
    return (
        SocraticBench.from_data(data_source)
        .apply(SeedStage(seeder))
        .batch()
        .apply(ChatStage(student, teacher, max_interactions=max_interactions))
        .flatten()
        .apply(EvaluationStage(judge))
    )
