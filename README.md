# GPTtest

This repository demonstrates a simple custom GPT wrapper tailored for
consultants at Implement Consulting Group. The script `gpt_wrapper.py` offers a
command-line interface that:

- accepts workflow context (for example, email drafting support),
- calls OpenAI's GPT-4 model with consultancy-specific prompting,
- populates a ready-to-share consultant brief template, and
- logs every interaction to a local SQLite database for lightweight usage
  analytics.

## Prerequisites

- Python 3.11+
- An OpenAI API key exported as `OPENAI_API_KEY`
- The [`openai`](https://pypi.org/project/openai/) Python SDK (install with
  `pip install openai`)

## Usage

```bash
export OPENAI_API_KEY="sk-..."  # Replace with your actual key
python gpt_wrapper.py --workflow-type "Email Draft" --context "Preparing follow-up mail..."
```

If no context is passed on the command line the script will prompt for it
interactively. Usage entries are stored in `usage_logs.db` by default and can be
queried using any SQLite tooling.

