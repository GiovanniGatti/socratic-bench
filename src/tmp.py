from __future__ import annotations

from openai import Client

from socratic_bench.agents import StudentAgent, TeacherAgent, JudgeAgent, ConversationSeederAgent
from socratic_bench.llms import OpenAILLM
from socratic_bench.pipeline import SocraticBench
from socratic_bench.readers import PrincetonChapters
from socratic_bench.schemas import Record
from socratic_bench.stages import SeedStage, ChatStage, EvaluationStage

if __name__ == "__main__":
    client = Client()
    llm = OpenAILLM("gpt-4o-mini", client)
    llm.healthcheck()

    student = StudentAgent(llm)
    teacher = TeacherAgent(llm)
    judge = JudgeAgent(llm)
    seeder = ConversationSeederAgent(llm)

    bench = SocraticBench.from_data(PrincetonChapters(Record, 2, 100))
    seeded = bench.apply(SeedStage(seeder))
    batched = seeded.batch()
    chatted = batched.apply(ChatStage(student, teacher))
    flattened = chatted.flatten()
    evaluated = flattened.apply(EvaluationStage(judge))
    out, t = evaluated.run()
    # b2 = bench.apply(Tokenize())
    # b3 = b2.apply(Count())
    print(out)
    print(t)

    # API - pipeline
    # s = SocraticBench.default()
    # s.run()

    # API - selecting pipeline stages
    # s = (SocraticBench.from_data(PrincetonChapters(Record, num_conversations=10))
    #      .with_stage(SeedStage(None))
    #      .with_state()
    #      # .with_stage(DumpStage())
    #      # .with_stage(DumpStage())
    #      )
    # s.run()

    # API - fine control (simulating single chat)

    # API - overriding behavior

    # API - using local models (HF transformers)
    pass
