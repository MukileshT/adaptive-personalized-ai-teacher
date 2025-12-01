# Adaptive Personalized AI Teacher

This project is a personalized AI teacher built with Google's Agent Development Kit (ADK). It learns how a student studies and adapts its teaching style in real time. It starts with a short overview, checks understanding with small questions, and then decides whether to go deeper, give another example, or slow down.

The system uses multiple agents (planner, teacher, evaluator, root) plus a persistent memory layer to support adaptive lessons, quizzes, and progress tracking.

## Features

- Short first-time explanations, deeper follow-ups on request.
- Adaptive quizzes that focus on weak topics.
- Persistent student profile: learning style, difficulty, confidence, mastered/weak topics.
- Commands for quick insights: `memory`, `profile`, `weaknesses`, `session`.
- Works both in terminal and through `adk web`.

## Requirements

- Python 3.10+
- A Google API key in environment variable `GOOGLE_API_KEY`
- Python packages from `requirements.txt`:

## Running with ADK (recommended)

From the project folder, install dependencies and start the agent with ADK:

`pip install -r requirements.txt`

run in terminal mode  
`adk run myAgent`

or run the web interface  
`adk web`

Then open the URL shown by ADK in your browser for the web UI.

## Running the standalone CLI

There is also a simple CLI entry point in `agent.py`:

`python agent.py`

This starts an interactive loop in the terminal. You can type commands like:

- `teach me python basics`
- `quiz`
- `memory`
- `profile`
- `weaknesses`
- `session`
- `exit`

Make sure your `GOOGLE_API_KEY` is set before running:

```bash
export GOOGLE_API_KEY="your-key-here" # macOS / Linux
set GOOGLE_API_KEY=your-key-here # Windows CMD
$env:GOOGLE_API_KEY="your-key-here" # PowerShell
```
