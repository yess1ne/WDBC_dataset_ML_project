# Virtual Environment Setup Guide

This guide explains how to create a Python virtual environment, activate it, and install the project dependencies from `requirements.txt`.

---

## 1. Create a Virtual Environment

```bash
python -m venv venv
```

---

## 2. Activate the Virtual Environment

**Windows (PowerShell):**
```bash
venv\\Scripts\\Activate
```

**Windows (CMD):**
```bash
venv\\Scripts\\activate.bat
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

---

## 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Deactivate the Virtual Environment

```bash
deactivate
```

---

## 5. Optional: Upgrade pip

```bash
pip install --upgrade pip

