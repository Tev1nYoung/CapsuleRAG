# CapsuleRAG

`CapsuleRAG` is a structured Retrieval-Augmented Generation pipeline for multi-hop question answering.

The core idea is simple:

- replace coarse passage-only retrieval with **structured evidence capsules**
- decompose a complex question into **multiple retrieval anchors**
- run **local graph propagation** only around question-related structure
- fuse dense, structural, and lightweight jump signals before final answer generation

This repository is intended to keep the **core CapsuleRAG code only**. Personal interview materials, PPT assets, analysis scripts, figures, local datasets, and reproduction helpers are intentionally excluded from Git tracking.

## What problem this project targets

Naive RAG often struggles on multi-hop QA because it mostly relies on single-shot semantic similarity. In practice, multi-hop QA needs more than "find similar text":

- the evidence unit should be finer than a whole passage
- intermediate hops should be made explicit
- cross-document evidence should be organized, not just retrieved

CapsuleRAG addresses this by introducing a middle layer between raw text and final answer generation: **evidence capsules**.

## Core design

### 1. Evidence capsules

Instead of using only passages or very thin triples, CapsuleRAG extracts structured evidence units that keep:

- predicate / relation
- polarity
- arguments
- modifiers such as time or location
- provenance back to document / passage / sentence

This makes retrieval more compositional while keeping the evidence traceable.

### 2. Query DAG decomposition

A complex question is decomposed into several retrieval-ready sub-questions.

This improves evidence coverage because the system does not rely on a single retrieval path. The current implementation uses the DAG mainly for:

- retrieval-oriented sub-question generation
- dependency-aware candidate selection
- multi-anchor dense + structural fusion

It is **not** a full symbolic execution system.

### 3. Local mini-PPR propagation

The system does **not** run propagation over the whole graph by default.

Instead, it:

1. induces a small `k`-hop local subgraph around question seeds
2. runs sparse personalized propagation inside that subgraph

This keeps computation under control and reduces drift from distant noisy nodes.

## End-to-end pipeline

### Offline stage

- split corpus documents into passages
- extract evidence capsules with an LLM
- canonicalize entities
- canonicalize similar capsules
- build an evidence graph
- build embedding indexes

### Online stage

- decompose the question into retrieval anchors
- retrieve candidate capsules and passages
- run local propagation on the induced subgraph
- fuse dense, capsule, propagation, and jump signals
- assemble evidence passages
- generate the final answer with the LLM

## Current implementation scope

The current codebase is a streamlined implementation centered on:

- structured capsule extraction
- query DAG decomposition
- local mini-PPR propagation
- multi-signal ranking
- evidence assembly + final QA

Important scope notes:

- the public-facing project name is **CapsuleRAG**
- the Python package name is now `src/capsulerag`
- the main runtime classes are now aligned to `CapsuleRAG`
- bridge-style cross-document behavior still exists, but not as a standalone headline module

## Repository layout

The main code you likely care about is:

```text
main.py                                  CLI entrypoint
src/capsulerag/config.py                 runtime configuration
src/capsulerag/capsulerag.py             main pipeline orchestration
src/capsulerag/data/                     dataset loading helpers
src/capsulerag/embed/                    embedding backbones
src/capsulerag/offline/                  indexing / extraction / graph building
src/capsulerag/online/                   query DAG / retrieval / propagation / generation
src/capsulerag/metrics/                  retrieval and QA metrics
src/capsulerag/utils/                    shared utilities
```

Key files:

- `src/capsulerag/offline/passage_splitter.py`
- `src/capsulerag/offline/capsule_extractor.py`
- `src/capsulerag/offline/entity_canonicalizer.py`
- `src/capsulerag/offline/capsule_canonicalizer.py`
- `src/capsulerag/offline/graph_builder.py`
- `src/capsulerag/online/query_dag.py`
- `src/capsulerag/online/retriever.py`
- `src/capsulerag/online/propagation.py`
- `src/capsulerag/online/generator.py`

## Minimal runtime setup

Use your existing Conda environment:

```powershell
conda activate hipporag
```

Install the minimal Python dependencies if needed:

```powershell
pip install -r requirements.txt
```

Notes:

- `faiss` is expected to be installed separately on Windows, typically via Conda
- `llm_key.txt` is read from the project root if present, otherwise `OPENAI_API_KEY` can be used

## Running the pipeline

Retrieval only:

```powershell
python main.py --dataset sample --run_mode retrieval_only --embedding_name facebook/contriever --run_tag smoke_retrieval
```

Full RAG QA:

```powershell
python main.py --dataset sample --run_mode rag_qa --embedding_name facebook/contriever --run_tag smoke_qa
```

Outputs are written under:

```text
outputs/<dataset>/<llm>_<emb>/metrics/runs/<run_tag>/
```

## Expected dataset layout

`main.py` expects local dataset files under:

```text
reproduce/dataset/<dataset>.json
reproduce/dataset/<dataset>_corpus.json
```

This repository may not include those files in Git.

Expected high-level shapes:

- corpus file: a JSON list of documents with at least `title` and `text`
- sample file: a JSON list of QA samples with at least `question`, plus answer / support fields depending on the benchmark format

The loading logic lives in:

- `src/capsulerag/data/dataset_loader.py`

## What is intentionally not tracked here

To keep the repository focused on core code, the following are intentionally excluded from Git tracking in this working setup:

- local PPT / DOCX / PDF interview materials
- local figures and generated PNGs
- analysis notebooks / result scripts
- local reproduction datasets
- temporary preview files
- local tool scripts used only for presentation preparation

If you want to rebuild a full research reproduction workspace, you can add back your local datasets and helper scripts outside this minimal core snapshot.

## Practical summary for another agent

If another agent needs the shortest possible orientation:

- this is a **multi-hop RAG** project
- the main novelty is **structured evidence capsules**
- retrieval is improved by **Query DAG decomposition** and **local mini-PPR**
- the public project name is **CapsuleRAG**
- the current Python package path is **`src/capsulerag`**
- the main entrypoint is **`main.py`**
- the main orchestrator is **`src/capsulerag/capsulerag.py`**
- the most important online logic is in **`src/capsulerag/online/retriever.py`**

## Current limitations

The present implementation still has clear limits:

- capsule extraction quality depends on LLM stability
- Query DAG is retrieval-oriented, not a full explicit reasoning engine
- retrieval gains do not always fully transfer to final QA gains
- evidence assembly and final generation remain bottlenecks on harder datasets

## Naming note

Use **CapsuleRAG** consistently for the project, package path, and main runtime class.

Current aligned names:

- package: `capsulerag`
- config class: `CapsuleRAGConfig`
- main class: `CapsuleRAG`
