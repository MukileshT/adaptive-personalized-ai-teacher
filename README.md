# adaptive-personalized-ai-teacher

This project is a **_personalized AI teacher_** that learns how a student studies and adapts its teaching style in real time. It doesn’t just dump long explanations.

**_It starts with a short overview, checks what the student understood, and then decides whether to go deeper, give another example, or slow down._**

The system uses multiple agents: one plans the topic, one teaches, and one creates quizzes that focus on weak areas. It remembers past sessions, tracks confidence and difficulty level, and gradually moves a learner from basics to more advanced problems. Everything runs in plain text, so it works well in a simple terminal as well as in the web UI.

**What problem does it solve**  
Most online tutorials are static: every student gets the same long lesson, no matter their pace, background, or mood. That makes it easy to get bored or lost.

**This agent tries to fix that by**:

**1.** starting with short, low‑pressure explanations instead of walls of text  
**2.** asking small comprehension questions before moving on  
**3.** adjusting difficulty based on quiz results and past performance  
**4.** remembering weak topics and revisiting them later with targeted practice

**How it works**
Under the hood the project uses Google’s ADK to run a small team of agents:

- A **planner agent** turns the user’s request into a clear topic and suggested teaching style.
- A **teacher agent** builds lessons that follow a set of teaching principles: reduce cognitive load, use progressive disclosure, add micro‑checks, and keep everything in clean Markdown with simple ASCII where needed.
- A **evaluator agent** generates adaptive quizzes from the lesson and past history. It focuses about 40% of questions on topics the student struggled with.
- A **root agent** coordinates everything, passes context between agents, and makes sure the student only sees well‑formatted text (no raw JSON or tool calls).
- A **persistent memory layer** stores a profile for each student: learning style, current difficulty level, confidence estimate, topics mastered, and topics that still need work.
  _This memory is shared across CLI (python agent.py), adk run myagent, and adk web, so progress is not lost between sessions._

---

**_In addition to adaptive lessons and quizzes, agent supports some additional special features:_**

- memory: Instantly view your learning history and progress, including recent topics studied and quiz attempts.

- profile: See your adaptive student profile with current confidence, learning style, difficulty level, and pacing recommendation.

- weaknesses: Identify subjects or concepts you’re struggling with, so you can target practice before moving on.

- session: Quickly review your current session summary: number of lessons, quizzes taken, comprehension checks, and overall performance.

- teach me [topic]: Learn any subject with dynamic depth and pacing.

- quiz: Take an adaptive quiz focused on your recent learning and weak areas.

---

**Key features:**

- The agent’s ability to detect and adapt to your personal learning style (visual/kinesthetic/auditory/analytical/mixed).

- Internal confidence and engagement tracking for each student.

- Automatic difficulty adjustment (lessons and quizzes get harder or easier based on performance history).

- Comprehension checking in real time during lessons—not just quizzes.

- Modular architecture (agents can be extended or customized).

- Everything runs in pure text and Markdown, with optional ASCII diagrams—no images required.
