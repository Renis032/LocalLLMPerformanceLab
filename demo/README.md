# Demo data

`lab-demo.sqlite.gz` is the lab's database as it stood when the demo was last rebuilt: every session recorded while this project was built, on an RTX 3060 Ti with Claude Haiku, Sonnet and Opus as the cloud baseline. Nothing in it is invented or tidied up. Sessions that failed or got warnings are kept as they happened, with the reasons.

```bash
uv run python -m llmlab demo
```

That unpacks a fresh copy to a temporary folder, opens the dashboard on it, and deletes the copy when you stop it with Ctrl+C. Starting runs is off in the demo, because a run needs llama.cpp and a model ([docs/setup.md](../docs/setup.md)).

## What was changed before publishing

Only file paths. Paths under the home folder and the folder the runs were started from contain the user name, so they are replaced with `~` and `<repo>`. Everything else (settings, prompts, every run, every telemetry sample, every answer) is as recorded.

To publish your own results the same way:

```bash
uv run python -m llmlab demo --rebuild --db path/to/lab.sqlite
```

The rebuild refuses to write the file if your user name is still anywhere in it, and [`tests/test_demo.py`](../tests/test_demo.py) checks the shipped file contains no home folder at all.
