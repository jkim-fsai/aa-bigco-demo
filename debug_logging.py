"""Debug script to diagnose logging issues with MIPROv2."""
import logging
import re

import dspy
from datasets import load_dataset
from dspy.primitives.example import Example


class DebugTracker(logging.Handler):
    """Captures all log messages for inspection."""

    def __init__(self) -> None:
        super().__init__()
        self.all_messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        self.all_messages.append(msg)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
debug_tracker = DebugTracker()

# Attach to DSPy's MIPROv2 logger
logging.getLogger("dspy.teleprompt.mipro_optimizer_v2").addHandler(debug_tracker)

# Also try other potential logger names
logging.getLogger("dspy").addHandler(debug_tracker)
logging.getLogger().addHandler(debug_tracker)

# Configure DSPy
lm = dspy.LM("openai/gpt-4.1-nano")
dspy.configure(lm=lm, async_max=10)


def format_context(ctx: dict) -> str:
    """Format context titles and sentences into a readable string."""
    paragraphs = []
    for title, sentences in zip(ctx["title"], ctx["sentences"]):
        text = "".join(sentences)
        paragraphs.append(f"[{title}]\n{text}")
    return "\n\n".join(paragraphs)


def load_hotpotqa(split: str) -> list[Example]:
    """Load examples from HotPotQA with formatted context."""
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    examples = []
    for item in ds:
        ex = Example(
            question=item["question"],
            context=format_context(item["context"]),
            answer=item["answer"],
        ).with_inputs("question", "context")
        examples.append(ex)
    return examples


# Load minimal dataset
print("Loading minimal dataset...")
trainset = load_hotpotqa("train[:20]")
valset = load_hotpotqa("validation[:20]")
print(f"Loaded {len(trainset)} train, {len(valset)} val examples\n")


# Define QA module
class BasicQA(dspy.Module):
    def __init__(self) -> None:
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, context: str, question: str):
        return self.generate_answer(context=context, question=question)


def validate_answer(example: Example, pred, trace=None) -> bool:
    return example.answer.lower() in pred.answer.lower()


# Run optimizer
print("Running MIPROv2 with minimal config...")
optimizer = dspy.MIPROv2(
    metric=validate_answer,
    auto="light",
    num_threads=10,
    max_bootstrapped_demos=1,
    max_labeled_demos=1,
)

optimized_qa = optimizer.compile(BasicQA(), trainset=trainset, valset=valset)

# Analyze captured logs
print("\n" + "=" * 60)
print("LOGGING DIAGNOSTICS")
print("=" * 60)
print(f"Total messages captured: {len(debug_tracker.all_messages)}")

# Show all unique logger names that emitted messages
print("\nSearching for score-related messages...")
score_messages = [msg for msg in debug_tracker.all_messages if "score" in msg.lower()]
print(f"Found {len(score_messages)} messages containing 'score'")

if score_messages:
    print("\nSample score messages:")
    for msg in score_messages[:5]:
        print(f"  - {msg}")

print("\nSearching for instruction-related messages...")
instruction_messages = [msg for msg in debug_tracker.all_messages if "instruction" in msg.lower()]
print(f"Found {len(instruction_messages)} messages containing 'instruction'")

if instruction_messages:
    print("\nSample instruction messages:")
    for msg in instruction_messages[:5]:
        print(f"  - {msg}")

print("\nAll messages (first 20):")
for i, msg in enumerate(debug_tracker.all_messages[:20]):
    print(f"{i+1}. {msg[:150]}")

print("\nAll messages (last 20):")
for i, msg in enumerate(debug_tracker.all_messages[-20:]):
    print(f"{i+1}. {msg[:150]}")
print("=" * 60)
