# AGENTS.md

## Repository overview

This is a sandbox repository where @AnneBeyer runs agent-driven experiments.

`main` holds only shared infrastructure: these instructions, the journal
renderer in `tools/`, the SSH helper for remote GPU hosts, and the workflows
in `.github/workflows/`.

Each experiment lives on its own branch, in its own top-level directory, and
usually has a pull request. Experiment pull requests are never merged.

```text
<experiment>/
  pixi.toml           # the experiment's own environment
  ...                 # code, scripts, configs
  journal/
    index.md          # entry page, links every iteration
    <iteration>/
      report.md       # one iteration
      *.svg *.png *.json
```

## Environment setup

Give every experiment its own pixi workspace in `<experiment>/pixi.toml`, with
a recent Python. Pick conda-forge or PyPI dependencies depending on what the
experiment needs (conda-forge for compiled/CUDA stacks, PyPI for pure Python).
Commit `pixi.lock`.

The `pixi.toml` at the repository root only provides the shared tools
(Python and pandoc). Do not add experiment dependencies to it.

## Choosing where to run

Decide where an experiment runs from what it needs, cheapest first:

1. **Locally** (the machine the agent runs on) for CPU work and anything that
   fits comfortably in local memory.
2. **A remote GPU host over SSH** when the experiment needs CUDA, more memory
   than the local machine has, or a long run that should not block the local
   machine.

### Remote GPU hosts

Remote hosts are reached with a dedicated SSH key pair that exists only for
agents. Its public key is registered on the AWS hosts and on the jump host.
The hosts are private IPs behind a jump host.

- Run `tools/agent_ssh.sh setup` once per machine or session. It writes
  `~/.ssh/agent-sandbox/config` from the `AGENT_SSH_*` settings and adds one
  `Host agent-<name>` alias per GPU host (ProxyJump is configured).
- Run `tools/agent_ssh.sh check` to see which hosts answer and how much GPU
  memory is free. Then use the aliases directly: `ssh agent-gpu1`,
  `rsync -a exp/ agent-gpu1:exp/`.
- If `tools/agent_ssh.sh` reports missing settings, stop and ask the human
  operator. Never generate, copy, print or commit key material.
- Hosts shut down automatically when idle. If one does not answer, ask the
  human operator to start it rather than retrying in a loop.
- Hosts are shared. Before starting a job, check that nothing else is using the
  GPU. Run long jobs under `tmux` or `nohup` with output going to a log file,
  so a dropped connection does not kill them.
- Copy results back into the experiment's `journal/` before you report them.
  The remote host is not storage.

## Running experiments

Use resources carefully:

- Start with short experiments that give the most information for the least
  compute.
- When benchmarking a user-facing API, focus on calls that take more than about
  1 s each. Those are the cases people actually wait for. Don't report only
  microbenchmarks.
- Where possible, keep individual runs under about 10 s so you can iterate
  quickly. Only run longer experiments when the results justify it.
- Watch host and GPU memory. Plan every run to keep at least 20% of memory
  free, so that memory exhaustion never crashes a run or invalidates its
  results.
- Record the hardware, library versions and exact command for every reported
  number.

## Reporting

Summarize results as Markdown tables in the experiment journal (see the layout
above):

- `journal/index.md` is the entry page. Keep it chronological and link every
  iteration's `report.md`.
- Put each iteration in `journal/<iteration>/report.md`, with its plots and
  raw data (`svg`, `png`, `json`) alongside it.
- Use relative links. Links ending in `report.md` or `index.md` become
  `index.html` when rendered, and links to other `.md` files become `.html`.
- The renderer reads GitHub-flavored Markdown, including tables, fenced code
  and `$...$` math, which it renders as MathML.

Render locally to check the result:

```bash
pixi run render --journal <experiment>/journal --site /tmp/site
```

No workflow file is needed for each experiment. When a pull request changes
any `*/journal/**` file, `.github/workflows/publish-journals.yml` renders every
journal on the branch and publishes it at
`https://annebeyer.github.io/agent-sandbox/pr-<number>/<experiment>/`. It then
posts the links as a pull request comment, which it updates on later runs.
Reports from other pull requests stay online.
