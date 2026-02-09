# 🧠 Autonomous Technical Blog Generator with Research & Image Planning

An end-to-end **agentic blog generation system** built using **LangGraph**, **LangChain**, and **LLMs** that automatically:

* Decides whether web research is required
* Collects and synthesizes evidence (when needed)
* Plans a high-quality technical blog outline
* Writes each section in parallel
* Decides where diagrams/images add value
* Generates images via an image model
* Produces a **publication-ready Markdown blog**

This project demonstrates **real-world LLM orchestration**, not just prompt chaining.

---

## ✨ Key Features

* **Intelligent routing**

  * Automatically selects between `closed_book`, `hybrid`, and `open_book` modes
* **Optional web research**

  * Uses Tavily search only when freshness or citations are required
* **Structured planning**

  * Generates a multi-section blog plan with goals, bullets, and word targets
* **Parallel writing**

  * Each section is written independently using LangGraph fan-out
* **Image-aware editing**

  * Decides *if* images are needed and *where*
* **Automatic image generation**

  * Generates diagrams and embeds them into Markdown
* **Deterministic reducer**

  * Produces a single, clean `.md` output ready for GitHub / blogs

---

## 🏗️ System Architecture

```text
User Topic
   │
   ▼
Router ──► (needs research?)
   │           │
   │           ▼
   │        Research (Tavily)
   │
   ▼
Orchestrator (Blog Plan)
   │
   ▼
Fan-out
(Parallel Workers)
   │
   ▼
Reducer Subgraph
   ├─ Merge sections
   ├─ Decide images
   └─ Generate & place images
   │
   ▼
Final Markdown Blog
```

This architecture is implemented using **LangGraph**, ensuring:

* clear state transitions
* reproducibility
* debuggability
* extensibility

---

## 🧩 Core Components

### 1️⃣ Router

Determines **upfront** whether the topic:

* is evergreen (no research)
* needs fresh examples/tools
* depends on current events

### 2️⃣ Research Module

* Uses Tavily search
* Deduplicates and normalizes evidence
* Never hallucinates sources

### 3️⃣ Orchestrator (Planner)

Produces a structured blog plan:

* 5–9 sections
* Each with a goal, bullets, word target
* Flags sections needing code or citations

### 4️⃣ Workers (Parallel)

Each worker:

* Writes exactly one section
* Follows strict constraints
* Adds code only when required
* Cites evidence only when allowed

### 5️⃣ Reducer + Image Pipeline

A dedicated **subgraph** that:

* Merges sections deterministically
* Inserts image placeholders (`[[IMAGE_1]]`)
* Generates diagrams via an image model
* Gracefully degrades if image generation fails

---

## 📁 Project Structure

```text
.
├── app.py                 # Main graph + runner
├── images/                # Auto-generated images
│   ├── self_attention_overview.png
│   ├── self_attention_math_flow.png
│   └── self_attention_performance_tradeoffs.png
├── Understanding_Self_Attention_in_Transformer_Architecture.md
├── README.md
```

---

## ⚙️ Installation

### Requirements

* Python 3.9+
* API keys for:

  * OpenAI
  * Tavily
  * Google Gemini (for images)

### Install dependencies

```bash
pip install langgraph langchain langchain-openai pydantic tavily-python google-genai
```

---

## 🔐 Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
GOOGLE_API_KEY=your_google_api_key
```

---

## ▶️ Usage

```python
from app import run

result = run("Self Attention in Transformer Architecture")
```

Output:

* A fully written technical blog
* Images generated and embedded
* Markdown saved to disk

---

## 📝 Example Output

* **Title:** Understanding Self-Attention in Transformer Architecture
* **Sections:** Concepts, math, code, performance, debugging
* **Images:**

  * Self-attention overview diagram
  * QKV computation flow
  * Performance trade-offs visualization

All output is **ready to publish**.

---

## 🧠 Design Principles

* **No hallucinated facts**
* **Research only when necessary**
* **Explicit state, not hidden prompts**
* **Graceful failure handling**
* **Separation of concerns**
* **Production-oriented, not demo code**

---

## 🚀 Why This Project Matters

This is **not** a toy blog generator.

It demonstrates:

* Agent routing
* Conditional tool use
* Parallel execution
* Structured outputs
* Reducer patterns
* Multimodal reasoning

Exactly the skills required for **real-world LLM systems**.

---

## 🙌 Acknowledgements

* LangGraph for stateful LLM orchestration
* LangChain for tool & message abstractions
* Tavily for research grounding
* Gemini for fast image generation


