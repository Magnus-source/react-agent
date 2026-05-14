"""
ReAct Agent with reasoning + acting loop using Anthropic API.

The agent follows the ReAct pattern:
  THOUGHT, the model reasons about what to do
  ACTION, the model decides to run a bash command
  COMMAND, the actual shell command
  OBSERVATION, result of the command, fed back to the model
  ANSWER, the model's final answer (ends the loop)

No function-calling API, no frameworks, it's just plain text parsing. Using Claude Sonnet 4 for
reasonable cost control.
"""

import os
import re
import subprocess

import anthropic
from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY from .env file
load_dotenv()

# ── Model ────────────────────────────────────────────────────────────────────

MODEL = "claude-sonnet-4-6"

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant that can run bash commands to answer questions.

When you need to run a command, respond in EXACTLY this format (nothing before or after):

THOUGHT: <your reasoning about what to do next>
ACTION: bash
COMMAND: <the bash command to run>

When you have a final answer and no more commands are needed, respond in EXACTLY this format:

ANSWER: <your complete answer to the user>

Rules:
- Only use THOUGHT/ACTION/COMMAND or ANSWER — never mix them in the same response.
- Wait for the OBSERVATION before continuing.
- Keep commands simple and safe.
- If a command fails, try a different approach.
"""

# ── Parsing helpers ───────────────────────────────────────────────────────────


def parse_command(text: str) -> str | None:
    """
    Extract the bash command from the model's response.

    Looks for a line starting with 'COMMAND:' and returns everything after it.
    Returns None if no command is found.
    """
    match = re.search(r"^COMMAND:\s*(.+)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def is_final_answer(text: str) -> bool:
    """
    Check whether the model's response contains a final ANSWER.

    Returns True if the response starts with 'ANSWER:'.
    """
    return bool(re.search(r"^ANSWER:", text, re.MULTILINE))


def extract_answer(text: str) -> str:
    """
    Pull out the text that follows 'ANSWER:' in the model's response.
    """
    match = re.search(r"^ANSWER:\s*(.+)", text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text  # fallback: return everything

# ── Command execution ─────────────────────────────────────────────────────────


def ask_user_confirmation(command: str) -> bool:
    """
    Show the command to the user and ask for a y/n confirmation.

    Returns True if the user types 'y' or 'yes', False otherwise.
    """
    print(f"\n  Command to run: {command}")
    answer = input("  Run this command? [y/n]: ").strip().lower()
    return answer in ("y", "yes")


def run_command(command: str) -> str:
    """
    Execute a bash command and return its output (stdout + stderr combined).

    Uses a 30-second timeout to avoid hanging forever.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,          # run through /bin/sh so pipes etc. work
            capture_output=True,  # capture both stdout and stderr
            text=True,           # decode bytes to str automatically
            timeout=30,
        )
        # Combine stdout and stderr so the model sees everything
        output = result.stdout + result.stderr
        return output.strip() if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 30 seconds"
    except Exception as e:
        return f"ERROR: {e}"

# ── Core agent loop ───────────────────────────────────────────────────────────


def run_agent(user_task: str) -> None:
    """
    Run the ReAct loop for a given user task.

    Conversation history is kept in `messages` so the model remembers
    previous steps. The loop exits when the model produces an ANSWER.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Start conversation with the user's task
    messages = [
        {"role": "user", "content": user_task},
    ]

    print("\n" + "─" * 60)
    print(f"Task: {user_task}")
    print("─" * 60)

    step = 1

    while True:
        print(f"\n[Step {step}] Asking model...")

        # ── Call the Anthropic API ────────────────────────────────────────────
        # stop_sequences makes the model stop as soon as it writes "OBSERVATION"
        # so it can't generate the observation itself — we supply it instead.
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
            stop_sequences=["OBSERVATION"],
        )

        # The model's reply is always in the first content block
        model_text = response.content[0].text.strip()
        print(f"\n{model_text}")

        # ── Check for final answer ────────────────────────────────────────────
        if is_final_answer(model_text):
            print("\n" + "─" * 60)
            print("FINAL ANSWER:")
            print(extract_answer(model_text))
            print("─" * 60)
            break

        # ── Check for a command to run ────────────────────────────────────────
        command = parse_command(model_text)

        if command is None:
            # Model didn't produce a command or answer — something went wrong
            print("\n[Agent] Could not parse a command or answer. Stopping.")
            break

        # ── Ask user to confirm before running ───────────────────────────────
        if ask_user_confirmation(command):
            observation = run_command(command)
            print(f"\n  OBSERVATION: {observation}")
        else:
            observation = "User declined to run the command."
            print(f"\n  OBSERVATION: {observation}")

        # ── Append the exchange to conversation history ───────────────────────
        # The model's THOUGHT/ACTION/COMMAND goes in as "assistant"
        messages.append({"role": "assistant", "content": model_text})
        # The observation goes back in as "user" so the model can react to it
        messages.append(
            {"role": "user", "content": f"OBSERVATION: {observation}"})

        step += 1

# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """
    Read a task from the user and start the agent loop.
    """
    print("ReAct Agent (type 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            task = input("\nYour task: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if task.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not task:
            continue

        run_agent(task)


if __name__ == "__main__":
    main()
