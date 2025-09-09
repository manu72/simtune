# Simtune Specifications Document

## 1. Purpose

Simtune exists to make personalised AI-powered knowledge experts accessible to everyone.  
It enables non-technical users to fine-tune an LLM with their own knowledge base, and to continuously improve the domain-specific expert via feedback.

This document defines the **vision, principles, architecture, feature scope, and roadmap** of Simtune. It will guide developers, testers, and product managers in delivering a coherent product.

---

## 2. Vision

- Every person can have their own personalised Large Language Model.
- Anyone can create a custom vector database that LLMs can interrogate
  \_ Anyone can finetune their own LLM with the vector knowledge base they provide
- The AI expert improves with feedback and evolves over time.
- The system balances **simplicity** (for non-technical users) with **rigor** (proper fine-tuning, feedback loops, safety checks).
- Simtune is modular: CLI → browser UI → multi-model → multi-user.

---

## 3. Guiding Values

1. **Simplicity for the user**: technical complexity hidden, workflows guided.
2. **Transparency**: the system explains what it is doing, especially during fine-tuning.
3. **Ethical AI**: enforce safe-use constraints, discourage misuse.
4. **Extensibility**: architecture supports multiple LLMs and feedback types.
5. **Trust**: data stays private, clear account and data deletion policies.

---

## 4. Stakeholders

- **End Users**: non-technical individuals who want an AI author.
- **Product Managers**: define feature scope and roadmap.
- **Developers**: implement CLI, adapters, UI, and feedback systems.
- **Testers**: validate dataset correctness, fine-tune pipeline, and feedback loop reliability.

---

## 5. Scope and Features

### Stage 1 Terminal Proof of Concept

1. Simple CLI commands to create an author profile.
2. Guided dataset builder to create 100 - 500 correctly formatted json examples. JSON examples to be stored locally and separately for each author.
3. Fine-tune orchestration for one provider (OpenAI or Gemini).
4. Draft content generation via the tuned author.
5. Feedback collection (ratings + edits).
6. Dataset regeneration and retraining from feedback.

### Stage 1B Knowledge Base Features

- **PDF Knowledge Import**: Upload and process PDF documents to build domain-specific knowledge bases.
- **Vector Database**: Transform documents into searchable embeddings for intelligent content retrieval.
- **Domain Dataset Generation**: Automatically create fine-tuning datasets from knowledge base content.
- **Expert Model Creation**: Fine-tune LLMs to become domain-specific experts rather than writing style mimics.
- **Knowledge Querying**: Query domain experts with context-aware responses backed by source documents.
- **Content Persistence**: All generated content saved with source attribution and metadata tracking.

### Stage 2 Browser-based Interface

- Basic web interface with inline editing.
- Upload PDFs
- Generate vector database
- Generate training dataset
- Run a finetuning job succesfully
- Dashboard showing key metrics
- Feedback and continuous improvement over time

### Stage 3 Multi-Model

- Support for OpenAI, Gemini, and local LLMs.
- LoRA/PEFT fine-tuning for small local models.
- Prompt-tuned mode for models without fine-tuning.

### Stage 4 Multi-User Accounts

- Authentication (passwordless, GitHub, or email).
- Per-user isolated storage of data.
- Bring-your-own-key support.

### Stage 5 Production UI/UX

- Polished dashboard with versioning, rollback, and analytics.
- Job queueing and cost controls.
- Export/delete data features.

---

## 6. Architecture

### High-Level Components

- **CLI / UI Layer**  
  Entry point for user interactions. Typer CLI → later React/Svelte web app.

- **Core Services**

  - **Adapters**: wrap provider APIs (OpenAI, Gemini, Ollama).
  - **Dataset Manager**: builds, validates, and formats JSONL datasets.
  - **Fine-Tune Orchestrator**: launches jobs, tracks status.
  - **Author Runtime**: generates drafts from fine-tuned models.
  - **Feedback Engine**: turns ratings/edits into structured training examples.
  - **Evaluation Module**: style similarity, readability, safety checks.

- **Data Layer**
  - Author profiles (YAML).
  - Datasets (`train.jsonl`, feedback diffs).
  - Model registry (`models.json`).
  - Future: DB and per-user storage buckets.

### Technology

- **Stage 1**: Python, Typer, Pydantic, Rich, dotenv.
- **Stage 2+**: FastAPI/Flask backend, React/Svelte frontend.
- **Stage 3+**: Ollama integration, PEFT libraries.
- **Stage 4+**: Database (Postgres/SQLite), secure storage, auth layer.

---

## 7. Data Model

- **AuthorProfile**

  - name, description, influences, goals.

- **StyleGuide**

  - tone, vocabulary, sentence length, do/don’t list.

- **TrainingExample**

  ```json
  {
    "messages": [
      { "role": "system", "content": "Style guide condensed" },
      { "role": "user", "content": "Prompt text" },
      { "role": "assistant", "content": "Expected output in author’s style" }
    ],
    "tags": ["clarity", "tone"]
  }
  ```

- **FeedbackRecord**
  - rating, original output, edited output, tags.
  - converted into new TrainingExamples.

---

## 8. Workflow

1. User creates an author profile.
2. Simtune builds a training dataset.
3. Fine-tuning job runs via provider adapter.
4. Author generates content.
5. User gives feedback (rating/edit).
6. Feedback converted into new examples.
7. Optional retraining loop.

---

## 9. Quality & Testing

- **Unit Tests**: adapters, dataset validation, feedback conversion.
- **Integration Tests**: CLI flows, fine-tune job orchestration.
- **Acceptance Tests**: user journeys (create author, fine-tune, generate, feedback).
- **Safety Tests**: banned topics, token limits, output length.
- **Performance Checks**: dataset size caps, cost per fine-tune.

---

## 10. Risks & Mitigations

- **Overfitting**: use small style guides + cap dataset size.
- **Cost spikes**: enforce example limits and confirmation steps.
- **Provider changes**: isolate API calls in adapters.
- **Feedback quality**: encourage edits, not just ratings.
- **Data privacy**: .env secrets, never commit user data.

---

## 11. Roadmap Summary

- Stage 1: CLI proof of concept (current).
- Stage 2: Browser UI.
- Stage 3: Multi-model, local models.
- Stage 4: Multi-user accounts.
- Stage 5: Production-ready platform.

---

## 12. Success Criteria

- A non-technical user can fine-tune a personal author in under 30 minutes.
- Drafts from the author reflect measurable similarity to user samples.
- Feedback reliably improves style adherence.
- System is modular and extendable for new providers.

---

This document is **living**: update it whenever major design, scope, or architecture changes occur.
