# python-langgraph-workflow

LangGraph workflow that diagnoses one application log line with local models running through Ollama.

## Install

Install [Ollama](https://ollama.com/download) and download the models:

```bash
ollama pull granite3-dense:2b
```

Create the Python environment and install the requirements:

```bash
python -m venv .venv
```

```text
# Linux and macOS
.venv/bin/python -m pip install -r requirements.txt

# Windows
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

Run the verbose example:

```text
# Linux and macOS
.venv/bin/python main.py example.log --verbose

# Windows
.venv\Scripts\python.exe main.py example.log --verbose
```

Pass a log line directly:

```bash
.venv/bin/python main.py --input "ERROR api - Request rejected: authentication token expired" --verbose
```

## Workflow

```mermaid
flowchart LR
    A[classify_log<br/>Model] -->|category found| B[analyse_log<br/>Model]
    A -->|unknown| E[rejected_response]
    B --> C[validate_analysis<br/>Model]
    C -->|valid| D[final_response<br/>Python]
    C -->|rejected| E[rejected_response]
```

The first error or warning line becomes a work package diagnosis. Pydantic validates each model response, LangGraph shares state between nodes, and Python builds the final response.

The workflow uses IBM Granite 3 Dense 2B for every node. Change it with `OLLAMA_MODEL`.

## Example output

```json
{
  "final_response": {
    "category": "authentication",
    "probable_cause": "The client or application making the API request has not provided a valid or refreshed authentication token.",
    "confidence": 90.0,
    "recommended_next_steps": [
      "Check if the authentication token is being properly stored and refreshed.",
      "Verify that the token is not expired and has sufficient validity period.",
      "Ensure the client application is correctly handling token expiration events.",
      "Contact the API provider or support team if the issue persists."
    ]
  }
}
```

The exact response may vary between runs.

## References

- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangChain Ollama](https://docs.langchain.com/oss/python/integrations/chat/ollama)
- [Ollama](https://ollama.com/)
- [Pydantic](https://docs.pydantic.dev/latest/)
