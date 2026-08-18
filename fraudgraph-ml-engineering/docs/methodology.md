# Methodology

This document describes the method framework of the project. Concrete
implementations and quantitative results are withheld until the associated
paper is published.

## 1. Problem framing

Fraud detection in financial transactions is naturally *multi-view*: the
same record can be read as a node in a relational graph (who transacts with
whom), as a behavioural sequence (what this entity did over time), and as an
event stream (what happened, when). Each view carries complementary signal,
and a robust detector should exploit all of them.

The framework is organized around three falsifiable research questions:

| Question | Idea |
| --- | --- |
| **RQ1 — structure under heterophily** | A graph encoder that is deliberately designed for heterophily (fraud nodes often connect to innocent nodes) captures signals that a homophily-assuming encoder averages away. |
| **RQ2 — ordered behaviour** | Relation and event sequences add stable signal beyond graph structure alone. |
| **RQ3 — label scarcity** | With a well-designed scheduling and label-reveal mechanism, the useful fraction of a supervised fraud workflow survives when labeled data is scarce. |

## 2. Architecture framework

```text
transaction records
        │
        ├──────────────┬──────────────────┬──────────────────┐
        ▼              ▼                  ▼                  ▼
  relational        relation          event             flat
  graph view        sequences         sequences         features
        │              │                  │                  │
        ▼              ▼                  ▼                  ▼
  graph encoder   seq encoder       seq encoder      (auxiliary)
        └──────────────┴────────┬─────────┘                  │
                                ▼                            │
                         fusion classifier ◄──────────────────┘
                                │
                                ▼
                        fraud decision
```

Training is orchestrated by a federated-style controller (multi-round,
multi-client aggregation) and a dataset scheduler (which datasets and label
regimes to train on, and when).

## 3. Engineering discipline

- **Protocol first**: every claim is attached to a protocol (seeds, splits,
  selection rule, evaluation rule), never to a single headline number.
- **Views are first-class**: graph, relation-sequence, and event-sequence
  views are constructed explicitly, not folded into a flat feature dump.
- **Selection separated from reporting**: validation selects checkpoints and
  thresholds; the held-out test partition is evaluated only after selection.
- **Reproducibility**: runs are recorded with args, seed, timing, metrics,
  diagnostics, and checkpoint paths.
- **Dependency-light validation**: the repository can be validated before the
  graph-learning runtime is installed.

## 4. Evidence boundaries

- This repository distributes the method framework and engineering utilities;
  it does **not** distribute datasets, checkpoints, or results.
- Full reproduction requires the withheld implementation, released with the
  paper.