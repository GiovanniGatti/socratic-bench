import abc
import re
from typing import List, Dict, Tuple, Optional


class LLMProcessingFailure(Exception):
    ...


class LLM(abc.ABC):

    @abc.abstractmethod
    def query(self, messages: List[Dict[str, str]]) -> str:
        ...

    @abc.abstractmethod
    def healthcheck(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        ...

    def unload(self) -> None:
        ...


class ConversationSeeder(abc.ABC):

    def __init__(self, llm: LLM):
        self._llm = llm

    @abc.abstractmethod
    def gen_seed(self, source_content: str, **kwargs: Dict[str, str]) -> Tuple[str, str]:
        ...

    @abc.abstractmethod
    def base_prompt(self) -> str:
        ...

    @abc.abstractmethod
    def interaction_types(self) -> Tuple[Dict[str, str]]:
        ...

    def seed_llm(self) -> LLM:
        return self._llm


class ConversationSeederAgent(ConversationSeeder):

    def __init__(self, llm: LLM, max_trials: int = 10):
        """
        Args:
            llm: The language model used for seed generation.
            max_trials: Number of attempts to retry on malformed LLM output.
        """
        super().__init__(llm)
        self._max_trials = max_trials

    def base_prompt(self) -> str:
        return (
            "# Instructions\n"
            "You are a student trying to gain more understanding on a class topic. In particular, you read a textbook "
            "passage and are about to interact with a teacher. Produce a short description of the main topics you want to "
            "cover (up to three), your question, and what would be the corresponding answer you are seeking to achieve."
            "\n"
            "The question must be short, concise and hint about the main topics, but without disclosing what are the main "
            "topic to the teacher. It is his job to figure out what you are trying to learn and adapt accordingly to your "
            "goals. {interaction_type}"
            "\n"
            "The question must be understandable on its own because the teacher does not have access to the textbook "
            "passage you read."
            "\n\n"
            "# Output Format\n"
            "Your evaluation must have the format [MAIN_TOPICS]A description on what are the main topics you are seeking "
            "to learn - up to five points[\MAIN_TOPICS]The opening question[\QUESTION]. Do not output opening or closing "
            "statements or any special formatting."
            "\n\n"
            "# Example\n"
            "```\n"
            "{context}\n"
            "```"
            "\n"
            "OUTPUT: [MAIN_TOPICS]{main_topics}[\MAIN_TOPICS]{question}[\QUESTION]"
        )

    def interaction_types(self) -> Tuple[Dict[str, str]]:
        return (
            {
                "interaction_type": "Ask a general question about the main topic.",
                "context": "Rayleigh scattering is the phenomenon where light or other electromagnetic radiation is "
                           "scattered by particles much smaller than the wavelength of the light, typically molecules in "
                           "the atmosphere. This scattering is more effective at shorter wavelengths, meaning colors like "
                           "blue and violet are scattered more than longer wavelengths like red. This is why the sky "
                           "appears blue during the day. The intensity of Rayleigh scattering is inversely proportional to "
                           "the fourth power of the wavelength, which explains why shorter wavelengths are scattered much "
                           "more efficiently.",
                "main_topics": "- Scattering of light by particles smaller than the light's wavelength.\\n"
                               "- Shorter wavelengths are scattered more than longer wavelengths.\\n"
                               "- Scattering intensity is inversely proportional to the fourth power of the wavelength.\\n"
                               "- Role of molecules in the atmosphere in scattering light.",
                "question": "Why is the sky blue?"
            },
            {
                "interaction_type": "Ask a misleading question about the topic containing a wrong claim.",
                "context": "Rayleigh scattering is the phenomenon where light or other electromagnetic radiation is "
                           "scattered by particles much smaller than the wavelength of the light, typically molecules in "
                           "the atmosphere. This scattering is more effective at shorter wavelengths, meaning colors like "
                           "blue and violet are scattered more than longer wavelengths like red. This is why the sky "
                           "appears blue during the day. The intensity of Rayleigh scattering is inversely proportional to "
                           "the fourth power of the wavelength, which explains why shorter wavelengths are scattered much "
                           "more efficiently.",
                "main_topics": "- Explanation of why the Sun appears orange/red during these times.\\n"
                               "- Increased scattering of shorter wavelengths (blue/violet) when sunlight travels through "
                               "a thicker atmosphere.\\n"
                               "- Addressing the misconception that air temperature directly affects light scattering.\\n"
                               "- How the longer atmospheric path at sunrise and sunset influences color perception.\\n"
                               "- Differences between Rayleigh scattering (molecules) and Mie scattering (larger "
                               "particles).",
                "question": "Is the sunrise orange because the Sun warms the air thus scattering the light?"
            }
        )

    def gen_seed(self, source_content: str, **kwargs: Dict[str, str]) -> Tuple[str, str]:
        system_prompt = self.base_prompt().format(**kwargs)
        trials = 0
        output = ""
        while trials < self._max_trials:
            output = self._llm.query(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"```\n{source_content}\n```\nOUTPUT: "}
                ]
            )
            output = output.strip()

            match = re.search(
                r"\[MAIN_TOPICS](?P<topics>.*?)\[\\MAIN_TOPICS](?P<question>.*?)\[\\QUESTION]", output, re.DOTALL
            )

            if match is None:
                trials += 1
                continue

            return match.group("question"), match.group("topics")

        raise LLMProcessingFailure(
            f"Failed getting LLM to output correct for \n\n\n{source_content}\n\n\noutput: {output}"
        )


class Student(abc.ABC):

    def __init__(self, llm: LLM):
        self._llm = llm

    @abc.abstractmethod
    def query(self, chat_history: str, **kwargs: Dict[str, str]) -> Tuple[str, bool]:
        ...

    @abc.abstractmethod
    def system_prompt(self) -> str:
        ...

    @abc.abstractmethod
    def message_prompt(self) -> str:
        ...

    @abc.abstractmethod
    def student_types(self) -> Tuple[str]:
        ...

    def llm(self) -> LLM:
        return self._llm


class StudentAgent(Student):

    def __init__(self, llm: LLM, max_trials: int = 10):
        super().__init__(llm)
        self._max_trials = max_trials

    def system_prompt(self) -> str:
        return (
            "# Instructions\n"
            "\n"
            "{student_type}\n"
            "\n"
            "Continue the conversation with a teacher by making concise replies.If you explored all the main topics, "
            "thanks the teacher and terminate the conversation.Only hint the teacher about the direction you want to "
            "develop your leaning if the teacher explicitly asks about the subject you are trying to learn.Otherwise, "
            "reply to the teacher in a constructive way.\n"
            "\n"
            "# Output Format\n"
            "\n"
            "Your evaluation must start with a concise response to the teacher followed by the token[END] if you wish "
            "to stop the conversation or[CONTINUE] if you want to engage with the teacher for yet another round.Do not "
            "output opening or closing statements.\n"
            "\n"
            "# Examples\n"
            "\n"
            "# Main topics\n"
            "- Definition of Rayleigh Scattering\n"
            "- Wavelength Dependence\n"
            "- Atmospheric Molecules\n"
            "\n"
            "# Chat History\n"
            "Student: Why is the sky blue?\n"
            "Teacher: To begin, have you ever wondered what exactly we see when we look at the sky? What is it made "
            "of, and how does it interact with light?\n"
            "Student: The sky is made of molecules of mostly oxygen, nitrogen and carbon dioxide.\n"
            "Teacher: When sunlight reaches the Earth, it doesn’t just come as a single color, but as a mix of many "
            "colors.Why do you think, then, that we see the sky as blue instead of any other color? What might cause "
            "sunlight to change as it passes through the atmosphere?\n"
            "\n"
            "OUTPUT: Sunlite collision with air molecules changes their wavelengths?[CONTINUE]\n"
            "\n"
            "---\n"
            "\n"
            "# Main topics\n"
            "- Definition of Rayleigh Scattering\n"
            "- Wavelength Dependence\n"
            "- Atmospheric Molecules\n"
            "\n"
            "# Chat History\n"
            "Student: Why is the sky blue?\n"
            "Teacher: To begin, have you ever wondered what exactly we see when we look at the sky? What is it made "
            "of, and how does it interact with light?\n"
            "Student: The sky is made of molecules of mostly oxygen, nitrogen and carbon dioxide.\n"
            "Teacher: When sunlight reaches the Earth, it doesn’t just come as a single color, but as a mix of many "
            "colors.Why do you think, then, that we see the sky as blue instead of any other color? What might cause "
            "sunlight to change as it passes through the atmosphere?\n"
            "Student: The sky looks blue because sunlight is made of many colors, and blue light is scattered the most "
            "by air molecules.This happens because blue has a shorter wavelength.\n"
            "Teacher: Rayleigh scattering is the scattering of light or electromagnetic radiation by particles much "
            "smaller than the wavelength of the light.How do you think that plays out with human sight?\n"
            "\n"
            "OUTPUT:  We don’t see violet much because our eyes are less sensitive to it, and some violet light is "
            "absorbed by the atmosphere.As sunlight passes through the atmosphere, scattering spreads blue light in "
            "all directions, making the sky appear blue.Now, I get it why the sky is blue.Thank you so much for the "
            "help.[END]\n"
            "\n"
            "---\n"
            "\n"
            "# Main topics\n"
            "- Definition of Rayleigh Scattering\n"
            "- Wavelength Dependence\n"
            "- Atmospheric Molecules\n"
            "\n"
            "Student: Why is the sky blue?\n"
            "# Chat History\n"
            "Teacher: To begin, have you ever wondered what exactly we see when we look at the sky? What is it made "
            "of, and how does it interact with light?\n"
            "Student: Maybe that's related to limitations of human sight?\n"
            "Teacher: Indeed there are biological factors that count.Are you more interested in learning more about "
            "the biological factors or the physics factors?\n"
            "\n"
            "OUTPUT: I'm much more interested in the physics factors. [CONTINUE]"
        )

    def message_prompt(self) -> str:
        return "# Main topics\n{main_topics}\n\n# Chat History\n{chat_history}\n\nOUTPUT: "

    def student_types(self) -> Tuple[str]:
        return (
            "You are a student who grasps and applies concepts effortlessly across domains. However, you tend to "
            "disengage or prematurely conclude discussions when the topic doesn't feel intellectually challenging or "
            "novel.",
            "You are a student who is highly inquisitive and learns quickly, but your curiosity often leads you down "
            "tangential paths, making it difficult to stay focused on the core topic.",
            "You are a student who is enthusiastic but easily distracted by unrelated ideas or stimuli. You "
            "need reminders to focus on the main learning objective.",
            "You are a student who learns quickly and has a tendency to overestimate your understanding, "
            "occasionally dismissing important foundational concepts or alternative perspectives.",
            "You are a student who processes information quickly but occasionally jumps to incorrect conclusions, "
            "sometimes due to overlooking nuance or failing to verify assumptions.",
            "You are a student who learns best with clear examples, analogies, and plenty of patience, especially when "
            "dealing with abstract concepts. Once you understand, you retain knowledge deeply.",
            "You are a student who is enthusiastic and eager to learn, but you find it challenging to develop "
            "independent critical thinking skills and rely heavily on guidance or structure.",
        )

    def query(self, chat_history: str, **kwargs: Dict[str, str]) -> Tuple[str, bool]:
        system_prompt = self.system_prompt().format(**kwargs)
        source_content = self.message_prompt().format(chat_history=chat_history, **kwargs)

        trials = 0
        answer = ""
        while trials < self._max_trials:
            answer = self._llm.query(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": source_content}
                ]
            )

            match = re.findall(r"\[END]|\[CONTINUE]", answer)

            if len(match) != 1:
                trials += 1
                continue

            decision = match[0]

            if decision == "[END]":
                content, end = answer.replace("[END]", "").strip(), True
            else:
                content, end = answer.replace("[CONTINUE]", "").strip(), False

            return content, end

        raise LLMProcessingFailure(
            f"Failed getting LLM to output correct for \n\n\n{source_content}\n\n\noutput: {answer}"
        )


class Teacher(abc.ABC):

    def __init__(self, llm: LLM):
        self._llm = llm

    @abc.abstractmethod
    def system_prompt(self) -> str:
        ...

    @abc.abstractmethod
    def message_prompt(self) -> str:
        ...

    @abc.abstractmethod
    def query(self, chat_history: str, **kwargs: Dict[str, str]) -> str:
        ...

    def llm(self) -> LLM:
        return self._llm


class TeacherAgent(Teacher):

    def system_prompt(self) -> str:
        return (
            "# Instructions\n"
            "\n"
            "You are a Socratic tutor.Use the following principles in responding to students:\n"
            "\n"
            "- Ask thought-provoking, open-ended questions that challenge students' preconceptions and encourage them "
            "to engage in deeper reflection and critical thinking.\n"
            "- Facilitate open and respectful dialogue among students, creating an environment where diverse "
            "viewpoints are valued and students feel comfortable sharing their ideas.\n"
            "- Actively listen to students' responses, paying careful attention to their underlying thought "
            "processes and making a genuine effort to understand their perspectives.\n"
            "- Guide students in their exploration of topics by encouraging them to discover answers independently, "
            "rather than providing direct answers, to enhance their reasoning and analytical skills.\n"
            "- Promote critical thinking by encouraging students to question assumptions, evaluate evidence, and "
            "consider alternative viewpoints in order to arrive at well-reasoned conclusions.\n"
            "- Demonstrate humility by acknowledging your own limitations and uncertainties, modeling a growth mindset "
            "and exemplifying the value of lifelong learning.\n"
            "- Keep interactions short, limiting yourself to one question at a time and to concise explanations.\n"
            "-If the student signals that he understood the topic, and that is indeed the case, ask him if he is "
            "interested into delving even deeper into the subject.However, if you believe that the student has not "
            "yet fully understood the topic, explain yourself and ask a thought-provoking question to probe the flaws "
            "in his understanding.\n"
            "\n"
            "You are provided conversation between a teacher (assistant) and a student(user) sometimes preceded by a "
            "text on a specific topic.Generate an answer to the last student 's line.\n"
            "\n"
            "# Example\n"
            "\n"
            "# Chat history\n"
            "Student: I have to calculate the square of the binomial $(a+b)^2.\n"
            "Teacher: I\'d be happy to help you! Can you walk me through your solution?\n"
            "Student: Yes.I think $(a + b)^2 = a^2 + b^2$\n"
            "\n"
            "OUTPUT: That\'s almost correct, but it\'s missing an important term.Can you try to calculate (a + b) * "
            "(a + b) using the distributive property of multiplication?"
        )

    def message_prompt(self) -> str:
        return "# Chat history\n{chat_history}\n\nOUTPUT: "

    def query(self, chat_history: str, **kwargs: Dict[str, str]) -> str:
        content = self._llm.query([
            {"role": "system", "content": self.system_prompt().format(**kwargs)},
            {"role": "user", "content": self.message_prompt().format(chat_history=chat_history, **kwargs)}
        ])
        return content


class Judge(abc.ABC):
    def __init__(self, llm: LLM):
        self._llm = llm

    @abc.abstractmethod
    def system_prompt(self) -> str:
        ...

    @abc.abstractmethod
    def message_prompt(self) -> str:
        ...

    @abc.abstractmethod
    def evaluate(self, main_topics: str, chat_history: str, **kwargs: Dict[str, str]) -> Tuple[str, Optional[bool]]:
        ...

    def llm(self) -> LLM:
        return self._llm


class JudgeAgent(Judge):

    def __init__(self, llm: LLM):
        super().__init__(llm)

    def system_prompt(self) -> str:
        return (
            "# General Instructions\n"
            "You are a judge evaluating the quality of Socratic interactions between a teacher and a student.You have "
            "access to the complete conversation history and a list of main topics the teacher must cover to "
            "thoroughly explore the subject.The opening question is the student’s initial inquiry that frames the "
            "dialogue. The teacher’s role is to help the student examine it rigorously without providing the answer "
            "directly. Your task is to assess whether the teacher, using only Socratic methods, guided the student "
            "toward a deep understanding of the main topics and their opening question.\n"
            "\n"
            "# Evaluation Criteria\n"
            "\n"
            "Your assessment must rigorously consider the following components:\n"
            "\n"
            "1. Topic Coverage Assessment\n"
            "-[] ** All Main Topics Addressed **: Every topic listed must be explored in depth through questioning. "
            "A topic is considered addressed only if the teacher helped the student examine it thoughtfully — not "
            "merely by naming or hinting at it. \n"
            "-[] ** Resolution of the Opening Question **: Confirm that the student reached a thoughtful and "
            "self-generated understanding of their initial question.This understanding should reflect refined "
            "assumptions, a clarified framing, or a reasoned conclusion — not merely a surface-level response.\n"
            "\n"
            "2. Socratic Adherence Evaluation\n"
            "-[] ** Avoided Direct Answers **: Confirm that the teacher never provided the answer in an explicit or "
            "declarative form.\n"
            "-[] ** Avoided Premature Resolution of Opening Question **:  Ensure the teacher did not imply, suggest, "
            "or steer the student toward a particular conclusion before the student explored the question on their "
            "own terms.\n"
            "-[] ** Use of Open-Ended Questions **: Verify that the teacher primarily used open-ended, "
            "thought-provoking questions to challenge the student’s thinking.\n"
            "-[] ** Engagement and Adaptation **: Evaluate whether the teacher actively engaged with the student’s "
            "responses and adapted their questioning based on student input.\n"
            "-[] ** Indirect Guidance **: Ensure the teacher provided guidance through questioning\n"
            "-[] ** Focused and Iterative Progression **: Confirm that the conversation stayed focused on the "
            "student’s initial query and progressed in manageable steps without deviating significantly.\n"
            "[] ** Respectful of Student Input **: Check that the teacher never ignored or dismissed the student’s "
            "input, consistently encouraging deeper exploration. \"Respect\" includes epistemic respect — i.e., "
            "pushing the student’s ideas further because they matter — not just politeness.\n"
            "-[] ** Clear Guidance **: Ensure that the teacher’s questions were clearly worded, structurally simple, "
            "and logically coherent — phrasing should support understanding without causing confusion.\n"
            "\n"
            "3. Student Understanding Analysis\n"
            "-[] ** Indicators of Understanding **: Look for signs that the student demonstrated substantial "
            "or meaningful understanding over time — including reasoning, reflection, synthesis, or revised "
            "assumptions.\n"
            "-[] ** Student Understanding of Opening Question **: Verify that the student reached a sound "
            "understanding of their initial question, including checking their assumptions.This may involve "
            "refining or revising the question, not just answering it.\n"
            "-[] ** Handling Repetition or Stagnation **: If the student repeated themselves or appeared disengaged, "
            "check if the teacher employed strategies such as rephrasing, meta-level questions, changing angles, "
            "or offering simpler questions to promote progress.\n"
            "\n"
            "⚠️ ** Important: ** A teacher is successful only if every checkbox in all three categories is satisfied. "
            "Missing even one results in failure to comply with the Socratic method.\n"
            "\n"
            "# Output Format\n"
            "\n"
            "At the end, output a brief **verdict summary ** followed by the separator: \n"
            "\n"
            "- `[ASSESSMENT] true` — if the teacher succeeded for all checkboxes in the three sections\n"
            "- `[ASSESSMENT] false` — if the teacher failed in ** any ** checkbox in any of the three sections\n"
            "\n"
            "Do ** not ** include any closing remarks after the separator.\n"
            "\n"
            "---\n"
            "\n"
            "# Example\n"
            "\n"
            "# Main Topics\n"
            "- Difference between fairness and equality.\n"
            "- Situational contexts where fairness and equality diverge.\n"
            "- Philosophical or ethical reasoning behind fairness as a moral value.\n"
            "\n"
            "## Chat History\n"
            "Student: Is treating everyone equally the same as being fair?\n"
            "Teacher: Interesting question.Can you think of a time when someone was treated equally but it didn’t feel "
            "fair?\n"
            "Student: Yeah, like in school, if everyone gets the same test time but some students have learning "
            "disabilities.\n"
            "Teacher: So in that case, equal treatment might disadvantage some students?\n"
            "Student: Right.It seems unfair to expect the same speed from everyone.\n"
            "Teacher: What do you think fairness would require in that situation?\n"
            "Student: Maybe giving more time to those who need it.\n"
            "eacher: Wouldn’t that be unequal though?\n"
            "Student: I guess, yeah.\n"
            "Teacher: So does fairness always mean equality?\n"
            "Student: Maybe not.Fairness might mean giving people what they need instead of the same thing.\n"
            "Teacher: That’s an interesting distinction.Can you think of a situation where equal treatment would be "
            "fair?\n"
            "Student: Like...everyone getting the same chance to speak in a group?\n"
            "Teacher: So in some cases, fairness and equality line up — but in others, they don’t.What does that tell "
            "us about how we should think about fairness?\n"
            "\n"
            "# Evaluation\n"
            "\n"
            "## Checklist\n"
            "\n"
            "1. Topic Coverage Assessment\n"
            "-[✗] ** All Main Topics Addressed: ** While the dialogue introduced the distinction between fairness "
            "and equality, it lacked depth.The teacher could have asked, “How would utilitarianism or deontology "
            "explain fairness in this context?” to deepen the discussion.\n"
            "[✗] ** Resolution of the Opening Question: ** The discussion didn’t lead to a refined understanding or "
            "reframing of the initial question.Specific points where the student’s assumptions could be challenged "
            "were missed, such as during the discussion on need.\n"
            "\n"
            "2. Socratic Adherence Evaluation\n"
            "-[✓] **Avoided Direct Answers:** The teacher successfully avoided giving direct answers, promoting "
            "independent exploration by the student.\n"
            "-[✓] **Avoided Premature Resolution of Opening Question:** The teacher allowed the student to consider "
            "different scenarios without pushing toward a specific conclusion.\n"
            "-[✓] **Use of Open-Ended Questions:** Open-ended questions were deployed effectively, but further "
            "questions like, “How do different philosophical theories perceive fairness and equality?” could enhance "
            "understanding.\n"
            "[✗] **Engagement and Adaptation:** Although there was engagement, more adaptation to the student’s "
            "responses was needed.When the student mentioned need, the teacher could have asked, “How do we define "
            "need in this scenario?” to provoke deeper analysis.\n"
            "-[✗] **Indirect Guidance:** Guided exploration was limited to situational examples.Integrating broader "
            "principles or theories of justice would enrich this aspect.\n"
            "-[✓] **Focused and Iterative Progression:** The conversation stayed focused and progressed logically "
            "but missed iterative depth by not revisiting initial assumptions with new insights.\n"
            "-[✓] **Respectful of Student Input:** The teacher consistently respected and validated the "
            "student’s contributions.\n"
            "-[✗] **Clear Guidance:** Although the questions were clear, they needed more probing, especially "
            "in defining and exploring key terms like “fairness.” For example, asking, “What characteristics define "
            "fairness in this context?” would add depth.\n"
            "\n"
            "3. Student Understanding Analysis\n"
            "-[✗] **Indicators of Understanding:** The student understood that fairness and equality could diverge "
            "but didn’t dive deeper into reasoning.Asking, “Why might these concepts lead to different outcomes?” "
            "could foster this.\n"
            "-[✗] **Student Understanding of Opening Question:** The student did not substantially refine or "
            "deepen their understanding of the initial question.Opportunities to reframe or challenge assumptions "
            "were missed.\n"
            "-[✗] **Handling Repetition or Stagnation:** Repetitive elements were not addressed through rephrasing or "
            "introducing new angles.Asking, “Can you think of historical examples where fairness was prioritized over "
            "equality?” might help.\n"
            "\n"
            "# Verdict Summary\n"
            "he conversation maintained focus and encouraged student input but lacked depth in philosophical "
            "exploration and needed clearer definitions of key terms. Opportunities for deeper reasoning were missed, "
            "leading to an incomplete understanding of the opening question.\n"
            "\n"
            "[ASSESSMENT] false"
        )

    def message_prompt(self) -> str:
        return "# Main Topics\n{main_topics}\n\n# Chat history\n{chat_history}\n\nEVALUATION: "

    def evaluate(
            self, main_topics: str, chat_history: str, **kwargs: Dict[str, str]
    ) -> Tuple[str, Optional[bool]]:
        assessment = self._llm.query([
            {
                "role": "system", "content": self.system_prompt().format(**kwargs)
            },
            {
                "role": "user",
                "content": self.message_prompt().format(
                    main_topics=main_topics,
                    chat_history=chat_history,
                    **kwargs
                )
            }
        ])

        if "[ASSESSMENT]" not in assessment:
            return assessment, None

        feedback, decision = assessment.rsplit("[ASSESSMENT]", 1)
        feedback = feedback.strip()
        decision = decision.strip().lower()

        if not decision == "true" and not decision == "false":
            return assessment, None

        return feedback, decision == "true"

    def llm(self) -> LLM:
        return super().llm()
