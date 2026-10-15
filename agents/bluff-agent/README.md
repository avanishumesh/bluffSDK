# Triad Bluff agent

Run `uv run --locked python src/agent.py` from this directory. Input/output use the SDK's
newline-JSON protocol 1.0; stdout is reserved for protocol messages.

The entrypoint loads `src/strategy.py`. It sheds a largest truthful rank
group (one to four current card references), with occasional two-card
semi-bluffs containing one off-rank card when the pile is small and opponents
are not near finishing.
Challenges use logically impossible rank counts, selective-label-shrunk bluff
estimates, and observed opponent challenge decisions to account for clockwise
priority. A large pile increases pickup risk rather than being treated as
evidence of lying. Unchallenged claims are never counted as known truths.

The policy uses only its supplied observation and optional JSON checkpoint.
It now reads deck count for impossibility checks and profiles all configured
player seats. Its Dockerfile follows the runtime 2 resource-label/UID contract.
It does not import the evaluator or benchmark, use game seeds or opponent
identities, access other players' hands, retain card references across turns,
read/write files at runtime, or access the network. Its private action-mixing
seed is generated locally and retained only in the JSON checkpoint. It is
Python standard library only and CPU only. Dependencies are
declared in `pyproject.toml` and locked in `uv.lock` (none beyond Python).

From `~/Bluff`, run the SDK lifecycle, contract, strategy, and paired simulation
checks with:

```bash
python ../BluffSDK-agents/tools/check_competitive_agent.py
python ../BluffSDK-agents/tools/run_matches.py --games 50 --mode mix --seed0 42 --save
```

The included Dockerfile uses Python 3.11 and runs as a non-root user. Local
checks do not certify an official competition submission: an immutable
repository URL, commit SHA and organizer Docker evaluation are still needed.
