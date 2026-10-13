# Multimodal Document & Diagram RAG

A private, 100% offline, open-source alternative to NotebookLM for technical papers, engineering reports, manuals, and textbooks. 

Unlike standard text-only RAG pipelines, this system extracts **both full text and visual diagrams, schematics, charts, and figures**, associates them with their nearest captions and page numbers, and renders the referenced diagrams directly inside an interactive **Streamlit chat interface**.

Generative answers are synthesized locally using **offline Hugging Face GGUF models** running directly on your CPU or GPU — your documents and questions never leave your computer (unless you opt into the Gemini backend).

---

## Key Features

- **Private & Offline:** No documents, embeddings, or prompts are uploaded to cloud servers. The only network access is a one-time download of the embedding/reranker models (cached in `models/`) and any LLM you choose to download; after that it runs fully offline.
- **Multimodal Diagram Capturing:** Detects figure captions, renders the complete figure (raster images, vector drawings, or both) at high resolution, and indexes the captions for visual retrieval.
- **Follow-up Questions:** The web UI remembers the last few turns, so you can ask "what about its units?" after a question.
- **Offline Hugging Face Model Support:** Place any quantized `.gguf` model into the `models/` folder, and the engine runs local generative LLM inference via `llama-cpp-python`.
- **Built-in 1-Click Model Downloader:** Download popular GGUF models directly from Hugging Face through the Streamlit sidebar or the CLI.
- **Fast Local Embeddings:** Employs [FastEmbed](https://github.com/qdrant/fastembed) (`BAAI/bge-small-en-v1.5`) running locally without requiring heavy PyTorch or GPU dependencies.
- **Persistent Vector Database:** Uses [ChromaDB](https://www.trychroma.com/) with incremental indexing: unchanged PDFs are skipped, changed ones re-indexed, and deleted ones removed from the index.
- **Flexible LLM Backends:**
  - **Local Hugging Face GGUF** (Default & recommended: Llama-3.2, Qwen2.5, Mistral)
  - **Local Ollama** (Optional daemon: `ollama run llama3.2`)
  - **Google Gemini API** (Optional cloud API)
- **Interactive Web UI:** Modern Streamlit interface with chat history, document metrics, clickable source citations, and expandable diagram viewports.
- **Headless CLI:** Build indexes, download models, or query your documents directly from the terminal.

---

## Directory Structure

```text
RAG_workflow/
├── knowledge_base/        # Place your target PDF documents here (git-ignored)
├── models/                # Offline GGUF models + embedding model cache (git-ignored)
├── vector_db/             # Local ChromaDB persistent vector storage (auto-generated, git-ignored)
├── extracted_images/      # Extracted figures and schematics (auto-generated, git-ignored)
├── rag_retrieval.py       # Unified RAG engine (Extraction + Vector DB + UI + CLI)
├── requirements.txt       # Python dependencies
├── .env.example           # Optional environment template (e.g. API keys)
├── .gitignore             # Ignores database files, models, and virtual environments
└── README.md              # Documentation and setup guide
```

## Prerequisites & Installation Guides

Before starting, ensure you have both **Git** and **Python** installed on your machine.

---

### Git Installation Guide

If Git is not yet installed on your system, follow the steps for your OS:

#### Windows
You have two quick options:
- **Option A (Installer - Recommended):**
  1. Download the official 64-bit installer from [git-scm.com/download/win](https://git-scm.com/download/win).
  2. Run the `.exe` installer. You can click **Next** through the setup using default settings (ensure the option *"Git from the command line and also from 3rd-party software"* remains checked).
  3. Close and reopen PowerShell or Command Prompt, then verify:
     ```cmd
     git --version
     ```
- **Option B (Windows Package Manager / winget):**
  ```powershell
  winget install --id Git.Git -e --source winget
  ```

#### macOS
- **Option A (Apple Command Line Tools):**
  Open Terminal and run:
  ```bash
  xcode-select --install
  ```
- **Option B (Homebrew):**
  ```bash
  brew install git
  ```
- Verify:
  ```bash
  git --version
  ```

#### Linux
- **Ubuntu / Debian:**
  ```bash
  sudo apt update
  sudo apt install -y git
  ```
- **Fedora / RHEL:**
  ```bash
  sudo dnf install -y git
  ```
- **Arch Linux:**
  ```bash
  sudo pacman -S git
  ```
- Verify:
  ```bash
  git --version
  ```

#### Graphical Alternative: GitHub Desktop (All Platforms)
If you prefer not using the command line for Git operations, you can install **[GitHub Desktop](https://desktop.github.com/)**, which provides an intuitive visual interface to clone, commit, and push repositories.

---

### Python Installation Guide

If Python is not yet installed on your system, follow the instructions for your operating system:

### Windows
1. Download the latest Python 3.10+ installer from the official website: [python.org/downloads](https://www.python.org/downloads/)
2. Run the installer executable.
3. **IMPORTANT:** Check the box at the bottom that says **"Add Python to PATH"** before clicking "Install Now".
4. Open Command Prompt or PowerShell and verify:
   ```cmd
   python --version
   ```

### macOS
1. Open Terminal and install Homebrew (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install Python:
   ```bash
   brew install python
   ```
3. Verify installation:
   ```bash
   python3 --version
   ```

### Linux (Ubuntu / Debian)
1. Open Terminal and update package lists:
   ```bash
   sudo apt update
   sudo apt install -y python3 python3-pip python3-venv
   ```
2. Verify installation:
   ```bash
   python3 --version
   ```

---

## Quickstart Guide

### 1. Clone This Repository
```bash
git clone https://github.com/rajdesai-19/Engineering-rag.git
cd Engineering-rag
```

### 2. Set Up a Virtual Environment (Recommended)
Creating an isolated virtual environment keeps your Python packages organized:

- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate.bat
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note on Hardware Acceleration:** If you have an NVIDIA GPU and want GPU-accelerated inference with `llama-cpp-python`, install with CUDA flags:
> ```bash
> pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
> ```
> On standard CPUs, the default `pip install -r requirements.txt` works out of the box.

### 4. Add Documents
Copy or move any number of PDF files into the `knowledge_base/` folder:
```text
RAG_workflow/
└── knowledge_base/
    ├── engineering_handbook.pdf
    └── system_design_spec.pdf
```

---

## Offline Hugging Face Models (`models/` Folder)

The heart of this system is true local generative synthesis. You can download quantized `.gguf` models from Hugging Face and place them directly into the `models/` directory.

### Recommended Models

| Model | Size | RAM Required | Best For |
| :--- | :--- | :--- | :--- |
| **Llama-3.2-3B-Instruct (Q4_K_M)** | ~2.0 GB | ~3–4 GB | Fast, lightweight general QA & summaries |
| **Qwen2.5-3B-Instruct (Q4_K_M)** | ~2.1 GB | ~3–4 GB | Exceptional technical and scientific reasoning |
| **Qwen2.5-7B-Instruct (Q4_K_M)** | ~4.7 GB | ~6–8 GB | High-detail engineering synthesis & deep explanations |
| **Mistral-7B-Instruct-v0.3 (Q4_K_M)** | ~4.4 GB | ~6–8 GB | Strong general technical comprehension |

### 3 Ways to Get a Model:

1. **Option A: 1-Click Download via Streamlit UI (Easiest)**
   - Open the web interface (`streamlit run rag_retrieval.py`).
   - In the sidebar, select **"Local GGUF (from models/)"** and expand **"⬇️ Download Model from Hugging Face"**.
   - Select your preferred model from the dropdown and click its **"Download …gguf"** button.

2. **Option B: Download via CLI**
   ```bash
   python rag_retrieval.py --download-model "Llama-3.2-3B-Instruct (Q4_K_M, ~2.0 GB)"
   ```

3. **Option C: Manual Download & Drag-and-Drop**
   - Download any `.gguf` file directly from [Hugging Face](https://huggingface.co/models?search=gguf) (for example from `bartowski` or `Qwen`).
   - Place the `.gguf` file directly inside the `models/` folder:
     ```text
     RAG_workflow/
     └── models/
         └── Llama-3.2-3B-Instruct-Q4_K_M.gguf
     ```
   - The application automatically detects all `.gguf` files inside `models/` and lists them in the dropdown!

---

## Launching the Web Application

```bash
streamlit run rag_retrieval.py
```
The application will open in your default browser at `http://localhost:8501`.

### In the Web Interface:
1. **Index Documents:** Click **"🔄 Build / Update Index"** in the sidebar. The system extracts text chunks, isolates diagrams, creates dense embeddings, and stores them in ChromaDB. Only new or changed PDFs are processed.
2. **Select LLM Backend** (the dropdown only lists backends that are available):
   - **Local GGUF (from models/)** — select your model from `models/`.
   - *(Optional)* **Local Ollama** — appears when the Ollama daemon is running.
   - *(Optional)* **Google Gemini** — appears when `GEMINI_API_KEY` is set in `.env`. Excerpts and questions are sent to Google.
3. **Ask Questions:** Enter your question in the chat bar (e.g., *"What causes pressure drops?"* or *"Show me the schematic of the fluid film"*).
4. **Inspect Answers & Figures:**
   - The assistant synthesizes a grounded response citing specific pages.
   - If referenced sections contain diagrams, the **"🖼️ View Referenced Diagrams & Figures"** gallery renders the exact schematics beneath the answer.
   - The **"📑 View Source Citations & Excerpts"** expander reveals the underlying text passages and similarity match scores.

---

## Command-Line (Headless) Usage

You can also run all tasks directly from the terminal without opening a browser:

- **Update the Index** (new, changed, and deleted PDFs only):
  ```bash
  python rag_retrieval.py --index
  ```
- **Rebuild the Whole Index** (e.g. after upgrading this script):
  ```bash
  python rag_retrieval.py --rebuild
  ```
- **Query the Knowledge Base with a Local GGUF Model** (single question; no follow-up memory on the CLI):
  ```bash
  python rag_retrieval.py --query "Explain the discharge cycle" --model-path models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
  ```
- **Download a Model:**
  ```bash
  python rag_retrieval.py --download-model "Qwen2.5-3B-Instruct (Q4_K_M, ~2.1 GB)"
  ```

---

## Publishing to GitHub

To push this project to your own GitHub account (`rajdesai-19`):

> **Keep your documents private:** `.gitignore` excludes `knowledge_base/`, `vector_db/` (which stores the full text of indexed documents), `extracted_images/`, and `models/`. Always publish with `git` or GitHub Desktop as shown below. **Do not use GitHub's "Add files via upload" page**: it ignores `.gitignore` and skips hidden files such as `.gitignore` itself.


1. Open your terminal inside the `RAG_workflow` folder:
   ```bash
   cd RAG_workflow
   ```
2. *(If first time using Git on this machine)* Configure your identity:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```
3. Initialize and commit:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Multimodal Document & Diagram RAG"
   ```
4. Create a new repository on [GitHub](https://github.com/new) named `Engineering-rag`.
5. Link your remote and push:
   ```bash
   git branch -M main
   git remote add origin https://github.com/rajdesai-19/Engineering-rag.git
   git push -u origin main
   ```

### Alternatively: Using GitHub Desktop
1. Open GitHub Desktop and select **File > Add Local Repository...** (or press `Ctrl+O` / `Cmd+O`).
2. Choose your `RAG_workflow` directory and click **Add Repository** (if prompted to create a repository first, click **create a repository**).
3. Type a summary (e.g. `Initial commit: Multimodal Document & Diagram RAG`) and click **Commit to main**.
4. Click **Publish repository** to push it straight to your GitHub account (`rajdesai-19/Engineering-rag`).

---

## License

This project is licensed under the MIT License — free for personal, academic, and commercial use.
