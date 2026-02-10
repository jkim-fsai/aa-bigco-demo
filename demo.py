import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Sequence

import dspy
from datasets import load_dataset
from dspy.primitives.example import Example
from dspy.primitives.prediction import Prediction


class OptimizationTracker(logging.Handler):
    """Custom logging handler to capture per-iteration optimization metrics."""

    def __init__(self, output_dir: Path = Path("log_viz/runs")) -> None:
        super().__init__()
        self.trials: list[dict] = []
        self.instructions: list[str] = []
        self.current_instruction_idx = 0

        # JSONL logging for real-time visualization
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trials_jsonl_path = self.output_dir / f"trials_{self.run_id}.jsonl"
        self.jsonl_file = None
        self.written_trial_ids = set()  # Deduplication

    def open_jsonl(self) -> None:
        """Open JSONL file for writing."""
        if self.jsonl_file is None:
            self.jsonl_file = open(self.trials_jsonl_path, "a", buffering=1)
            metadata = {
                "type": "metadata",
                "run_id": self.run_id,
                "timestamp": datetime.now().isoformat(),
                "status": "started"
            }
            self.jsonl_file.write(json.dumps(metadata) + "\n")
            self.jsonl_file.flush()

    def close_jsonl(self) -> None:
        """Close JSONL file."""
        if self.jsonl_file:
            metadata = {
                "type": "metadata",
                "run_id": self.run_id,
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "total_trials": len(self.trials)
            }
            self.jsonl_file.write(json.dumps(metadata) + "\n")
            self.jsonl_file.flush()
            self.jsonl_file.close()
            self.jsonl_file = None

    def write_trial_to_jsonl(self, trial: dict) -> None:
        """Write trial to JSONL with deduplication."""
        trial_id = f"{trial['trial']}_{trial['score']}_{trial.get('eval_type', '')}"

        if trial_id in self.written_trial_ids:
            return

        self.written_trial_ids.add(trial_id)

        if self.jsonl_file:
            trial_entry = {
                "type": "trial",
                "run_id": self.run_id,
                "timestamp": datetime.now().isoformat(),
                **trial
            }
            try:
                self.jsonl_file.write(json.dumps(trial_entry) + "\n")
                self.jsonl_file.flush()
            except Exception as e:
                logging.error(f"Failed to write trial to JSONL: {e}")

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()

        # Capture proposed instructions
        if "Proposed Instructions for Predictor" in msg:
            self.current_instruction_idx = 0
        elif re.match(r"^\d+:", msg):
            # Instruction candidate line like "0: Given the fields..."
            match = re.match(r"^(\d+):\s*(.+)$", msg)
            if match:
                idx, instruction = match.groups()
                self.instructions.append({
                    "index": int(idx),
                    "instruction": instruction.strip(),
                })

        # Capture trial scores (MIPROv2 format with optional minibatch info)
        # Format 1: "Score: 70.0 with parameters ['...']."
        # Format 2: "Score: 60.0 on minibatch of size 35 with parameters ['...']."
        score_match = re.search(r"Score:\s*([\d.]+)(?:\s+on minibatch of size \d+)?\s+with parameters\s+(\[.+?\])", msg)
        if score_match:
            score = float(score_match.group(1))
            params = score_match.group(2)
            # Check if this is a minibatch or full eval
            is_minibatch = "minibatch" in msg
            self.trials.append({
                "trial": len(self.trials) + 1,
                "score": score,
                "parameters": params,
                "optimizer": "mipro",
                "eval_type": "minibatch" if is_minibatch else "full",
            })
            self.write_trial_to_jsonl(self.trials[-1])

        # Capture best scores and default program scores
        # Format: "Default program score: 66.5" or "New best full eval score! Score: 68.5"
        default_or_best_match = re.search(r"(?:Default program score|best full eval score.*Score):\s*([\d.]+)", msg)
        if default_or_best_match and not score_match:  # Only if not already captured above
            score = float(default_or_best_match.group(1))
            is_best = "best" in msg.lower()
            is_default = "Default" in msg
            self.trials.append({
                "trial": len(self.trials) + 1,
                "score": score,
                "parameters": "Default program" if is_default else "Best program",
                "optimizer": "mipro",
                "eval_type": "full",
                "is_best": is_best,
            })
            self.write_trial_to_jsonl(self.trials[-1])

        # Capture GEPA iteration scores
        gepa_iter_match = re.search(r"Iteration (\d+): (?:Valset score for new program|Best score on valset): ([\d.]+)", msg)
        if gepa_iter_match:
            iteration = int(gepa_iter_match.group(1))
            score = float(gepa_iter_match.group(2)) * 100  # Convert to percentage
            self.trials.append({
                "trial": iteration,
                "score": score,
                "parameters": f"GEPA Iteration {iteration}",
                "optimizer": "gepa",
            })
            self.write_trial_to_jsonl(self.trials[-1])

        # Capture best score updates
        if "Best full score so far!" in msg:
            if self.trials:
                self.trials[-1]["is_best"] = True

    def get_summary(self) -> dict:
        return {
            "instructions_proposed": self.instructions,
            "trials": self.trials,
            "best_trial": max(self.trials, key=lambda x: x["score"]) if self.trials else None,
        }


# Configure logging to capture DSPy optimizer output
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

# Add our custom tracker
optimization_tracker = OptimizationTracker()
logging.getLogger("dspy.teleprompt.mipro_optimizer_v2").addHandler(optimization_tracker)  # MIPROv2
logging.getLogger("dspy.teleprompt.signature_opt_typed").addHandler(optimization_tracker)  # COPRO
logging.getLogger("dspy.teleprompt.gepa").addHandler(optimization_tracker)  # GEPA
logging.getLogger("dspy").addHandler(optimization_tracker)  # Catch-all

# Configure DSPy with OpenAI (async concurrency for faster evals)
# Using gpt-4.1-nano: fastest/cheapest GPT-4.1 model with 1M context
# Alternatives: gpt-4o-mini ($0.15/$0.60), gpt-5-nano ($0.05/$0.40)
lm = dspy.LM("openai/gpt-4.1-nano")
dspy.configure(lm=lm, async_max=50)

# Load HotPotQA dataset with context (distractor setting)
print("Loading HotPotQA dataset...")


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


trainset = load_hotpotqa("train[:200]")
valset = load_hotpotqa("validation[:100]")  # For optimization
testset = load_hotpotqa("validation[100:200]")  # For final evaluation (separate)
print(f"Loaded {len(trainset)} train, {len(valset)} val, {len(testset)} test examples\n")


# Define a QA module that uses context
class BasicQA(dspy.Module):
    def __init__(self) -> None:
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, context: str, question: str) -> Prediction:
        return self.generate_answer(context=context, question=question)

    async def aforward(self, context: str, question: str) -> Prediction:
        return await self.generate_answer.acall(context=context, question=question)


# Metric: check if gold answer appears in predicted answer
def validate_answer(example: Example, pred: Prediction, trace: object = None) -> bool:
    return example.answer.lower() in pred.answer.lower()


# GEPA-compatible metric wrapper (requires specific signature)
def gepa_metric(example: Example, pred: Prediction, trace=None, student_code=None, teacher_code=None) -> float:
    """GEPA requires a metric that returns float and accepts 5 parameters."""
    correct = example.answer.lower() in pred.answer.lower()
    return 1.0 if correct else 0.0


# Evaluate a module on the devset (async for parallel execution)
async def evaluate_async(module: BasicQA, devset: Sequence[Example]) -> float:
    async def evaluate_one(example: Example) -> bool:
        try:
            pred = await module.acall(context=example.context, question=example.question)
            return validate_answer(example, pred)
        except Exception:
            return False

    results = await asyncio.gather(*[evaluate_one(ex) for ex in devset])
    return sum(results) / len(devset) * 100


def extract_optimized_prompt_info(optimized_module: BasicQA) -> dict:
    """Extract the optimized instruction and few-shot demos from the compiled module."""
    predictor = optimized_module.generate_answer

    info = {
        "timestamp": datetime.now().isoformat(),
        "instruction": None,
        "demos": [],
    }

    # Extract instruction from the predictor's signature
    if hasattr(predictor, "extended_signature"):
        sig = predictor.extended_signature
        if hasattr(sig, "instructions"):
            info["instruction"] = sig.instructions

    # Extract few-shot demos
    if hasattr(predictor, "demos") and predictor.demos:
        for demo in predictor.demos:
            demo_info = {
                "question": demo.question[:100] + "..." if len(demo.question) > 100 else demo.question,
                "answer": demo.answer,
            }
            if hasattr(demo, "reasoning"):
                demo_info["reasoning"] = demo.reasoning[:200] + "..." if len(demo.reasoning) > 200 else demo.reasoning
            info["demos"].append(demo_info)

    return info


def print_optimization_progress(tracker: OptimizationTracker) -> None:
    """Print per-iteration optimization metrics."""
    summary = tracker.get_summary()

    print("\n" + "=" * 60)
    print("PER-ITERATION OPTIMIZATION METRICS")
    print("=" * 60)

    # Print instruction candidates
    if summary["instructions_proposed"]:
        print("\n### INSTRUCTION CANDIDATES ###")
        for inst in summary["instructions_proposed"]:
            # Truncate long instructions
            text = inst["instruction"]
            if len(text) > 80:
                text = text[:77] + "..."
            print(f"  [{inst['index']}] {text}")

    # Print trial-by-trial results
    if summary["trials"]:
        print("\n### TRIAL-BY-TRIAL SCORES ###")
        print(f"{'Trial':<6} {'Score':<8} {'Parameters':<50} {'Best?':<5}")
        print("-" * 70)
        for trial in summary["trials"]:
            is_best = "***" if trial.get("is_best") else ""
            params = trial["parameters"]
            if len(params) > 48:
                params = params[:45] + "..."
            print(f"{trial['trial']:<6} {trial['score']:<8.1f} {params:<50} {is_best}")

        # Summary statistics
        scores = [t["score"] for t in summary["trials"]]
        print("-" * 70)
        print(f"Min: {min(scores):.1f}%  Max: {max(scores):.1f}%  "
              f"Mean: {sum(scores)/len(scores):.1f}%  Trials: {len(scores)}")

    print("=" * 60)


def print_prompt_evolution(baseline_instruction: str, optimized_info: dict) -> None:
    """Print a comparison of baseline vs optimized prompt."""
    print("\n" + "=" * 60)
    print("PROMPT EVOLUTION ANALYSIS")
    print("=" * 60)

    print("\n### BASELINE INSTRUCTION ###")
    print(f'"{baseline_instruction}"')

    print("\n### OPTIMIZED INSTRUCTION ###")
    if optimized_info["instruction"]:
        print(f'"{optimized_info["instruction"]}"')
    else:
        print("(Same as baseline or not extractable)")

    if optimized_info["demos"]:
        print(f"\n### FEW-SHOT DEMONSTRATIONS ({len(optimized_info['demos'])} examples) ###")
        for i, demo in enumerate(optimized_info["demos"], 1):
            print(f"\n--- Demo {i} ---")
            print(f"Q: {demo['question']}")
            print(f"A: {demo['answer']}")
            if "reasoning" in demo:
                print(f"Reasoning: {demo['reasoning']}")

    print("\n" + "=" * 60)


async def main() -> None:
    # Open JSONL file for real-time logging
    optimization_tracker.open_jsonl()

    try:
        # Capture baseline instruction
        baseline_qa = BasicQA()
        baseline_instruction = "Given the fields `context`, `question`, produce the fields `answer`."

        # Step 1: Evaluate baseline on TEST set (unoptimized)
        print("=" * 60)
        print("BASELINE EVALUATION (no optimization)")
        print("=" * 60)
        baseline_accuracy = await evaluate_async(baseline_qa, testset)
        print(f"Baseline Accuracy: {baseline_accuracy:.1f}%\n")

        # Step 2: Run GEPA optimization
        print("=" * 60)
        print("RUNNING GEPA OPTIMIZATION")
        print("=" * 60)
        print("Optimizing instructions via evolutionary search...")
        print("(Watch the logs below to see prompt evolution)\n")

        # GEPA needs a separate LM for reflection/instruction proposal
        reflection_lm = dspy.LM("openai/gpt-4.1-nano", temperature=1.0)

        optimizer = dspy.GEPA(
            metric=gepa_metric,
            auto="light",
            reflection_lm=reflection_lm,
            num_threads=10,
        )
        optimized_qa = optimizer.compile(BasicQA(), trainset=trainset)

        # Step 3: Display per-iteration metrics
        print_optimization_progress(optimization_tracker)

        # Step 4: Extract and display prompt evolution
        optimized_info = extract_optimized_prompt_info(optimized_qa)
        print_prompt_evolution(baseline_instruction, optimized_info)

        # Include iteration metrics in the saved results
        tracker_summary = optimization_tracker.get_summary()
        optimized_info["optimization_trials"] = tracker_summary["trials"]
        optimized_info["instruction_candidates"] = tracker_summary["instructions_proposed"]

        # Save optimization results to JSON for documentation
        results_file = Path("optimization_results.json")
        results_file.write_text(json.dumps(optimized_info, indent=2))
        print(f"\nOptimization details saved to: {results_file}")

        # Step 5: Evaluate optimized module on TEST set (held-out)
        print("\n" + "=" * 60)
        print("OPTIMIZED EVALUATION (on held-out test set)")
        print("=" * 60)
        optimized_accuracy = await evaluate_async(optimized_qa, testset)
        print(f"Optimized Accuracy: {optimized_accuracy:.1f}%\n")

        # Step 6: Show comparison
        print("=" * 60)
        print("RESULTS COMPARISON")
        print("=" * 60)
        print(f"Baseline Accuracy:  {baseline_accuracy:.1f}%")
        print(f"Optimized Accuracy: {optimized_accuracy:.1f}%")
        improvement = optimized_accuracy - baseline_accuracy
        print(f"Improvement:        {improvement:+.1f}%")
        print("=" * 60)

        # Update results file with final scores
        optimized_info["baseline_accuracy"] = baseline_accuracy
        optimized_info["optimized_accuracy"] = optimized_accuracy
        optimized_info["improvement"] = improvement
        results_file.write_text(json.dumps(optimized_info, indent=2))

    finally:
        # Close JSONL file
        optimization_tracker.close_jsonl()


if __name__ == "__main__":
    asyncio.run(main())
