# PyTorch Code Optimizer CLI

A command-line tool to profile PyTorch code, identify bottlenecks, and leverage LLMs to suggest optimizations.

## Setup

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate # or venv\Scripts\activate on Windows
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up your LLM API Key:**
    Set the environment variable for your chosen provider:
    ```bash
    export OPENAI_API_KEY='your_openai_key' 
    # or
    export GOOGLE_API_KEY='your_google_api_key'
    ```

## Usage

```bash
python optimizer_cli.py --script_path <path_to_your_script.py> --target_function <function_to_profile> [options]
```

See `python optimizer_cli.py --help` for more options. 