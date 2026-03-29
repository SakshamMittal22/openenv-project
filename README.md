# 📬 AI Email Triage Environment

> **A production-grade OpenEnv environment that simulates how knowledge workers
> process, classify, and respond to corporate email — built for AI agent
> evaluation.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-green.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)]()
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)]()
[![Tests](https://img.shields.io/badge/tests-16%20passed-brightgreen.svg)]()

---

## 🧑‍⚖️ For Judges — Quick Evaluation Guide

**30-second test:**
```bash
pip install -r requirements.txt
python tests.py       # 16 tests pass
python baseline.py    # See heuristic vs random comparison
python ui.py          # Open http://localhost:7860