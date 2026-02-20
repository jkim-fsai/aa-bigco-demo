"""DSPy module definitions."""

import dspy
from dspy.primitives.prediction import Prediction


class BasicQA(dspy.Module):
    """Basic question-answering module using chain-of-thought.

    This module takes a context and question as input and produces
    an answer using DSPy's ChainOfThought prompting strategy.

    Example:
        qa = BasicQA()
        result = qa(context="...", question="What is X?")
        print(result.answer)
    """

    def __init__(self) -> None:
        """Initialize the QA module with a ChainOfThought predictor."""
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, context: str, question: str) -> Prediction:
        """Generate an answer for the given context and question.

        Args:
            context: Background information to answer the question.
            question: The question to answer.

        Returns:
            Prediction object containing the answer.
        """
        return self.generate_answer(context=context, question=question)

    async def aforward(self, context: str, question: str) -> Prediction:
        """Async version of forward for parallel evaluation.

        Args:
            context: Background information to answer the question.
            question: The question to answer.

        Returns:
            Prediction object containing the answer.
        """
        return await self.generate_answer.acall(context=context, question=question)


class MultipleChoiceQA(dspy.Module):
    """Multiple-choice question-answering module using chain-of-thought.

    This module takes a question and formatted answer choices as input
    and produces a single letter answer (A/B/C/D). Designed for datasets
    like ARC-Challenge where the model must select from provided options.

    Example:
        qa = MultipleChoiceQA()
        result = qa(question="Which is largest?", choices="A) ant\\nB) whale")
        print(result.answer)  # "B"
    """

    def __init__(self) -> None:
        """Initialize the multiple-choice QA module with a ChainOfThought predictor."""
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question, choices -> answer")

    def forward(self, question: str, choices: str) -> Prediction:
        """Generate an answer for the given question and choices.

        Args:
            question: The question to answer.
            choices: Formatted answer choices (e.g., "A) option1\\nB) option2").

        Returns:
            Prediction object containing the answer.
        """
        return self.generate_answer(question=question, choices=choices)

    async def aforward(self, question: str, choices: str) -> Prediction:
        """Async version of forward for parallel evaluation.

        Args:
            question: The question to answer.
            choices: Formatted answer choices.

        Returns:
            Prediction object containing the answer.
        """
        return await self.generate_answer.acall(question=question, choices=choices)


class BooleanQA(dspy.Module):
    """Boolean question-answering module using chain-of-thought.

    This module takes a question (no context) and produces a yes/no answer.
    Designed for datasets like StrategyQA where the model must reason
    from parametric knowledge.

    Example:
        qa = BooleanQA()
        result = qa(question="Did Aristotle use a laptop?")
        print(result.answer)  # "no"
    """

    def __init__(self) -> None:
        """Initialize the boolean QA module with a ChainOfThought predictor."""
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question: str) -> Prediction:
        """Generate a yes/no answer for the given question.

        Args:
            question: The question to answer.

        Returns:
            Prediction object containing the answer.
        """
        return self.generate_answer(question=question)

    async def aforward(self, question: str) -> Prediction:
        """Async version of forward for parallel evaluation.

        Args:
            question: The question to answer.

        Returns:
            Prediction object containing the answer.
        """
        return await self.generate_answer.acall(question=question)
