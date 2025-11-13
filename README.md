# CompL-it Definitions Generation

This repository contains the data and source code used in the experiments described in the paper  
**“Extending the Semantic Layer of the CompL-it Italian Lexicon: Traits, Semantic Types, and Definitions.”**

It provides all necessary steps to reproduce the experiments, including environment setup and model usage.

---

## Requirements

- **GPU:** At least one GPU with **≥16 GB VRAM** (e.g., *NVIDIA RTX 4060 Ti 16 GB* or higher).  
- **Accounts required:**
  - **[Groq](https://groq.com/)** — to run `llama-3.3-70b` and `llama-4-maverick` during evaluation.
  - **[Nebius](https://nebius.ai/)** — to run `Quen2.5-72b` during evaluation.
- **Operating System:** All commands have been tested on **Ubuntu Linux 22.04 LTS** and **24.04 LTS**.

---

## Installation

### 1. Download and unzip the repository
```bash
mkdir definition-generation
cd definition-generation
wget -O definition-generation.zip https://anonymous.4open.science/api/repo/definition-generation-7C48/zip
unzip definition-generation.zip
```

### 2. Create a Python virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file based on the provided `env_stub` template,  
and fill in the required API keys or tokens for external services.
```bash
cp env_stub .env
vi .env
```

### 5. Install **Ollama** (for local LLM inference)
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 6. Pull the required LLMs from Hugging Face
```bash
ollama pull hf.co/microsoft/phi-4-gguf:Q6_K
ollama pull hf.co/unsloth/gemma-3-12b-it-GGUF:Q8_0
ollama pull hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q4_K_S
```

### 7. Verify installation
```bash
ollama list
```
You should see the three downloaded models listed.

---

## Running the Experiments

To reproduce the experiments, follow the commands and instructions provided in the file RUN.txt

---

