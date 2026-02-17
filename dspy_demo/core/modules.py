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
