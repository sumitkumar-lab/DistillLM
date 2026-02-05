# DistillLM
# 🔥 On-Device Intent Classification via LLM Distillation  
**Low-Latency · Low-Cost · Production-Ready AI System**

---

## 🚀 Project Overview

This project demonstrates how **large language models (LLMs)** can be distilled into **small, efficient on-device models** for real-world industry use cases.

The system performs **intent classification** (e.g., vehicle commands, smart device control) with:

- **No internet dependency**
- **CPU-only inference**
- **Low latency**
- **Tiny model size**
- **High accuracy retained from LLM teacher**

A complete **end-to-end AI engineering pipeline** is implemented — from dataset creation and LLM supervision to embedded-ready deployment.

---

## 🎯 Why This Project Matters (Industry Perspective)

Modern industries (automotive, defense, consumer electronics, aerospace) face the same constraints:

| Constraint | Cloud LLMs | This System |
|----------|-----------|------------|
| Latency | High | **Very Low** |
| Cost per request | High | **Near zero** |
| Internet required | Yes | **No** |
| Privacy | Risky | **Safe (on-device)** |
| Determinism | Unstable | **Deterministic** |

This project shows **how to use LLM intelligence without LLM deployment cost**.

---

## 🧠 Core Idea: Knowledge Distillation from LLM

We replace a traditional heavy teacher model with an **LLM acting as a reasoning supervisor**.

### Training Concept
1. LLM generates **intent labels + soft reasoning signals**
2. A **small student model** learns to mimic this behavior
3. Student is exported as a **TorchScript artifact** suitable for embedded systems

The result:  
> *LLM-level intent understanding in a lightweight on-device model.*

---

## 🏗️ System Architecture

                 ┌────────────────────┐
User utterance ─▶│  LLM Teacher       │
                 │  (GPT / Claude)    │
                 │  Intent reasoning  │
                 └─────────┬──────────┘
                           │ logits / probabilities
                           ▼
                ┌────────────────────┐
                │ Student Model      │
                │ (Tiny Transformer) │
                │ On-device inference│
                └────────────────────┘
                           ┬
                           ▼
            ┌──────────────────────────┐
            │ TorchScript / ONNX Model │
            └─────────┬────────────────┘
                      |
                      ▼
        ┌────────────────────────────────┐
        │ Embedded Device / Edge Runtime │
        └────────────────────────────────┘


```bash
pip install -r requirements.txt

streamlit run sre/deploy/embedded_ui.py
```