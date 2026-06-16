# Human notes — research scaffolding

Steps:
1. Research: role-play agent pros and challenges.
   First it is *Character Persona* → through conversation it grows and
   becomes more personalized.

---

## Core papers

| Paper | Focus |
|---|---|
| Role-Playing Agents Driven by LLMs (2025) | Systematic survey: technical evolution, core techniques, evaluation frameworks |
| Identifying and Mitigating Bottlenecks in RPAs (2026) | Personality-axis disentangled diagnostics + FACD decoding strategy |
| A Survey on Role-Playing Language Agents | From Persona to Personalization — survey |
| PsyMem (2025) | Psychological alignment + explicit memory control |
| Advancing RPAs with Role-Aware Reasoning | Role-aware reasoning |
| RoleRAG | Graph-guided retrieval-augmented role-play |

Reference: *A Survey on Role-Playing Language Agents.*

---

## ChatHaruhi vs RoleLLM vs CoSER — RAG implementation comparison

| Dimension | ChatHaruhi (2023) | RoleLLM (2023) | CoSER (2025) |
|---|---|---|---|
| Paper | arXiv:2308.09597 | arXiv:2310.00746 | arXiv:2502.09082 |
| Scale | 32 characters, 54,726 dialogues | 100 characters, 168,093 samples | 17,966 characters, 771 books |
| Source | Anime / film / game scripts | 916 publicly available scripts | Classical literary novels |

---

## 🔍 RAG mechanism comparison

### ChatHaruhi — "memory-retrieval RAG"

```
user input → embedding → retrieve top-k similar scenes from the
                         character's script-memory store
                              ↓
   System Prompt (character setup) + retrieved exemplar dialogues + user input
                              ↓
                              LLM
                              ↓
                            reply
```

Core design:

- **Knowledge base**: every dialogue/scene of the character is extracted
  from the source script and stored as "memory".
- **Retrieval strategy**: semantic similarity (embedding similarity).
  The user's current input drives retrieval of the most relevant past
  dialogue fragments.
- **Prompt construction**: `[character setup] + [retrieved few-shot
  exemplar dialogues] + [current conversation]`.
- **Characteristic**: simple and effective — the LLM mimics the
  retrieved dialogue style via in-context learning.
- **Limitation**: relies purely on semantic similarity; can retrieve
  surface-similar but situationally-mismatched dialogues.

### RoleLLM — "knowledge extraction + fine-tuning instead of RAG"

```
Stage 1: scripts → Context-Instruct → role-knowledge QA pairs (offline)
Stage 2: GPT-4 + retrieval augmentation → RoleGPT generates stylised
         dialogues (data production)
Stage 3: SFT an open-source model on the generated data
         → RoleLLaMA / RoleGLM (no RAG needed at deploy time)
```

Core design:

- **Context-Instruct**: chunks long scripts and uses GPT to extract
  character-specific knowledge from each chunk, generating QA pairs.
  In essence, this converts run-time retrieval (RAG) into train-time
  knowledge injection.
- **RoleGPT (closed-source)**: uses retrieval augmentation — retrieved
  character dialogues serve as few-shot examples.
- **RoleLLaMA (open-source)**: after RoCIT fine-tuning, inference needs
  no RAG — only a system instruction + character description +
  catchphrases.
- **Key finding**: the paper finds the system-instruction approach
  outperforms retrieval augmentation. Once the model has learned the
  character's knowledge through SFT, retrieval actually adds noise.
- **Limitation**: depends on fine-tuning; new characters require
  retraining or extensive prompt engineering.

### CoSER — "situation-driven RAG + character-experience retrieval"

```
parse full novels → structured data:
  - conversation setups
  - character experiences
  - internal thoughts
                    ↓
user input + situation → retrieve the character's experiences /
                         thoughts in similar situations
                    → inject into prompt → LLM → reply
```

Core design:

- **Finer data granularity**: beyond dialogue, the model has access to
  the character's **experiences** and **internal thoughts**.
- **Situation matching**: borrows from Stanislavski's *Given Circumstance
  Acting* — retrieval is keyed off the current situation, not just
  semantic similarity.
- **Multi-layer retrieval**: can pull the character's thoughts /
  behaviour in canonically analogous situations, not only dialogues.
- **Scale advantage**: a knowledge base of 17,966 characters that even
  covers rare ones.
- **Characteristic**: closest to "letting the AI understand the
  character's motivation like an actor before performing".
