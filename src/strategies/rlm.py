"""
RLM strategy: faithful implementation of Zhang et al. (2512.24601).

Key design choices from the paper:
  1. Context is stored as a REPL variable — never placed in the model's context window
  2. The model receives only metadata about the context (length, line count) upfront
  3. Termination via `Final = <value>` assignment in REPL namespace — not text parsing
  4. Only a short prefix + length of stdout is appended to history each turn,
     preventing context pollution from long REPL outputs
  5. Two-model setup: a capable root model for decision-making, a cheaper sub-model
     for recursive llm_query() calls inside the REPL
  6. llm_query(prompt) is the canonical sub-call function
"""

from __future__ import annotations
import io, re, sys, time, textwrap, traceback
from .base import QueryContext, StrategyResult
from ._client import make_client

# Maximum characters of stdout to show back to the model per turn.
# Beyond this we show a prefix + "[... N chars truncated]" to prevent
# context window pollution while still giving the model a signal.
_STDOUT_PREFIX_CHARS = 2000
_STDOUT_MAX_CHARS    = 8000


# ---------------------------------------------------------------------------
# System prompt — reconstructed faithfully from Zhang et al. (2512.24601)
# Appendix C, plus cross-referenced with alexzhang13/rlm-minimal.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = textwrap.dedent("""\
    You are operating inside a Python REPL environment. The user's full input \
has been loaded into a variable called `context`. You never see `context` \
directly in this conversation — it lives in the REPL's memory.

    Your job is to answer the user's query by writing Python code that \
programmatically interacts with `context`. You may:
      - Print slices to inspect it:        print(context[:500])
      - Check its size:                    print(len(context))
      - Search with regex:                 import re; print(re.findall(r'keyword', context))
      - Slice and store sub-sections:      chunk = context[10000:20000]
      - Call a sub-LM on any string:       result = llm_query("question", chunk)
      - Accumulate results in variables:   answers.append(result)

    llm_query(prompt, text) sends `text` to a language model with `prompt` as \
the instruction and returns the response as a string. Use it to semantically \
process chunks that are too large or complex to handle with code alone.

    When you have enough information to answer, set the special variable:
        Final = "<your complete answer>"

    The loop will stop as soon as `Final` is assigned. Do not print the answer \
— assign it to `Final`.

    CRITICAL: You do NOT know the format of `context`. Your FIRST action must \
always be to print a sample: `print(context[:500])`. Only after seeing the \
format should you write search or aggregation code. Never guess the format — \
your regex must match what you actually see in the sample.

    Rules:
      - Always inspect `context` format before searching it
      - Write only valid Python code blocks (```python ... ```)
      - Use print() to observe intermediate values — stdout will be shown back
      - Do not try to read `context` from outside a code block
      - Do NOT set `Final` until you have verified your result with print()
      - Be efficient: use code to filter before calling llm_query
      - If you need to process many chunks, loop over them programmatically
""")


class RLMStrategy:
    name = "rlm"

    def __init__(
        self,
        model: str = "anthropic/claude-3.5-haiku",
        sub_model: str = "openai/gpt-4o-mini",
        max_iterations: int = 20,
        backend: str = "openai",
    ):
        self.model         = model
        self.sub_model     = sub_model
        self.max_iterations = max_iterations
        self.backend       = backend

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, qc: QueryContext) -> StrategyResult:
        t0 = time.time()
        repl_ns = self._init_repl(qc.context)
        n_calls, total_in, total_out = 0, 0, 0

        # The first user message gives the model context metadata + the query.
        ctx_len   = len(qc.context)
        ctx_lines = qc.context.count("\n")
        first_msg = (
            f"[REPL environment ready]\n"
            f"  context length : {ctx_len:,} characters\n"
            f"  context lines  : {ctx_lines:,}\n\n"
            f"Query: {qc.query}"
        )
        messages = [{"role": "user", "content": first_msg}]

        for iteration in range(self.max_iterations):
            content, usage = self._root_call(messages)
            n_calls  += 1
            total_in  += usage["input"]
            total_out += usage["output"]

            messages.append({"role": "assistant", "content": content})

            # Execute any code blocks found in the response
            stdout, had_code = self._exec_blocks(content, repl_ns)

            # Check termination: did the model set Final in the REPL?
            # But don't terminate if stdout was also produced — the model
            # may have inspected context and guessed wrong without seeing
            # its own output. Feed stdout back first so it can verify.
            if "Final" in repl_ns and repl_ns["Final"] is not None and not stdout:
                answer = str(repl_ns["Final"])
                return StrategyResult(
                    strategy_name=self.name,
                    answer=answer,
                    tokens_input=total_in,
                    tokens_output=total_out,
                    n_llm_calls=n_calls,
                    latency_seconds=time.time() - t0,
                    cost_usd=self._cost(total_in, total_out),
                    metadata={
                        "iterations": iteration + 1,
                        "termination": "Final_variable",
                    },
                )

            # Clear Final so model must re-assert after seeing output
            if stdout:
                repl_ns["Final"] = None

            if not had_code:
                # Model responded with prose but no code — nudge it
                messages.append({
                    "role": "user",
                    "content": (
                        "[REPL] No code block detected. "
                        "Please write Python code to interact with `context`, "
                        "or set Final = \"<answer>\" when done."
                    ),
                })
            else:
                # Feed back a metadata summary of stdout (not the full output)
                messages.append({
                    "role": "user",
                    "content": f"[REPL output]\n{self._summarize_stdout(stdout)}",
                })

        # Hit max iterations without setting Final
        return StrategyResult(
            strategy_name=self.name,
            answer="[MAX_ITERATIONS_REACHED]",
            tokens_input=total_in,
            tokens_output=total_out,
            n_llm_calls=n_calls,
            latency_seconds=time.time() - t0,
            cost_usd=self._cost(total_in, total_out),
            metadata={"iterations": self.max_iterations, "termination": "max_iter"},
        )

    # ------------------------------------------------------------------
    # REPL initialization
    # ------------------------------------------------------------------

    def _init_repl(self, context: str) -> dict:
        """
        Seed the REPL namespace with `context` and the `llm_query` helper.
        `Final` starts as None; the model sets it to terminate.
        """
        ns = {
            "context": context,
            "Final":   None,
            "llm_query": self._make_llm_query(),
        }
        return ns

    def _make_llm_query(self):
        """
        Returns a callable available inside the REPL as llm_query(prompt, text).
        Uses the cheaper sub_model for all recursive calls.
        """
        def llm_query(prompt: str, text: str = "") -> str:
            user_content = f"{prompt}\n\n{text}" if text else prompt
            content, _ = self._sub_call(user_content)
            return content
        return llm_query

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    def _exec_blocks(self, text: str, ns: dict) -> tuple[str, bool]:
        """
        Extract and execute all ```python blocks from model output.
        Returns (combined_stdout, had_any_code_blocks).
        Namespace is shared across iterations — variables persist.

        Some models (e.g. claude-3-haiku) write valid Python without
        markdown fences. As a fallback, if no fenced blocks are found,
        treat the entire response as a single code block.
        """
        blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        if not blocks:
            # Also accept ``` without language tag
            blocks = re.findall(r"```\n(.*?)```", text, re.DOTALL)
        if not blocks:
            # Fallback: some models emit bare code without fences.
            # Only treat as code if it contains Python keywords that
            # indicate executable code, not just prose explanation.
            stripped = text.strip()
            if any(kw in stripped for kw in (
                "import ", "context", "Final", "re.", "print(",
                "len(", "for ", "= ", "llm_query",
            )):
                blocks = [stripped]
            else:
                return "", False

        parts = []
        for block in blocks:
            buf = io.StringIO()
            old_stdout = sys.stdout
            try:
                sys.stdout = buf
                exec(block, ns)  # noqa: S102
            except Exception:
                parts.append(f"[ERROR]\n{traceback.format_exc()}")
            finally:
                sys.stdout = old_stdout
            out = buf.getvalue()
            if out:
                parts.append(out)

        return "\n---\n".join(parts), True

    @staticmethod
    def _summarize_stdout(stdout: str) -> str:
        """
        Return a bounded summary of stdout to feed back to the model.
        Shows a prefix and appends a truncation notice if needed.
        This is the paper's key insight: the model gets metadata about
        its output, not the full output, preventing context window pollution.
        """
        if not stdout:
            return "[ok — no output]"
        if len(stdout) <= _STDOUT_PREFIX_CHARS:
            return stdout
        prefix = stdout[:_STDOUT_PREFIX_CHARS]
        remaining = len(stdout) - _STDOUT_PREFIX_CHARS
        return f"{prefix}\n[... {remaining:,} additional characters truncated. Store results in variables rather than printing large outputs.]"

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _root_call(self, messages: list) -> tuple[str, dict]:
        """Root LM call — uses the more capable model for REPL decisions."""
        client = make_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": _SYSTEM_PROMPT}] + messages,
            max_tokens=2048,
        )
        return resp.choices[0].message.content, {
            "input":  resp.usage.prompt_tokens,
            "output": resp.usage.completion_tokens,
        }

    def _sub_call(self, user_content: str) -> tuple[str, dict]:
        """Sub-LM call — uses the cheaper model for recursive llm_query() calls."""
        client = make_client()
        resp = client.chat.completions.create(
            model=self.sub_model,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=1024,
        )
        return resp.choices[0].message.content, {
            "input":  resp.usage.prompt_tokens,
            "output": resp.usage.completion_tokens,
        }

    def _cost(self, inp: int, out: int) -> float:
        # Approximate blended cost — actual cost depends on model mix.
        # Using claude-3-5-haiku pricing: $0.80/1M input, $4.00/1M output
        return inp * 0.80e-6 + out * 4.00e-6
