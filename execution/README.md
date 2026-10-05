# Execution Scripts (Layer 3)

Deterministic, testable Python scripts invoked by directives and orchestrated by the agent.

## Guidelines:
1. **Deterministic & Isolated**: Each script should perform a single well-defined task with predictable outputs.
2. **Environment Variables**: Load secrets and configurations from `.env` (e.g. using `python-dotenv`).
3. **Intermediates vs Deliverables**: Save temporary/scratch processing files to `.tmp/`.
4. **Error Handling**: Return clear exit codes and error logs on stderr/stdout so the orchestration layer can self-anneal.
5. **Well-Commented**: Include clear CLI arguments (using `argparse` or `click`) and usage docstrings.
