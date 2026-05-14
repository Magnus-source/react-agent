# ReAct Agent

A minimal ReAct (Reasoning + Acting) agent built with pure Python and the Anthropic API.
No frameworks, no function-calling API — just text parsing and a loop.

## How it works

```
User task
   │
   ▼
[Model] → THOUGHT / ACTION / COMMAND
   │
   ▼
[Python] parses COMMAND, asks user y/n
   │
   ▼
[Shell] runs the command → stdout/stderr
   │
   ▼
[Python] sends OBSERVATION back to model
   │
   ▼
repeat until model responds with ANSWER
```

## Setup

1. **Clone / copy the files**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create your `.env` file**
   ```bash
   cp .env.example .env
   # Open .env and replace the placeholder with your real API key
   ```

4. **Run the agent**
   ```bash
   python agent.py
   ```

## Example session

```
Your task: How many Python files are in the current directory?

[Step 1] Asking model...

THOUGHT: I need to count Python files in the current directory.
ACTION: bash
COMMAND: find . -name "*.py" | wc -l

  Command to run: find . -name "*.py" | wc -l
  Run this command? [y/n]: y

  OBSERVATION: 1

[Step 2] Asking model...

ANSWER: There is 1 Python file in the current directory.
```

## File overview

| File | Purpose |
|------|---------|
| `agent.py` | Main agent loop — all the logic lives here |
| `.env` | Your secret API key (never commit this) |
| `.env.example` | Template showing which variables are needed |
| `requirements.txt` | Python packages to install |

## Key concepts

- **ReAct loop** — the model alternates between reasoning (THOUGHT) and acting (COMMAND) until it can give an ANSWER.
- **No function-calling** — the model is instructed via the system prompt to use a plain-text format; Python parses it with `re.search`.
- **Conversation history** — `messages` grows on each step so the model remembers what happened before.
- **User confirmation** — every command is shown to the user before it runs; type `n` to skip it safely.
