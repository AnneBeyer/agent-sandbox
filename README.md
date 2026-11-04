# agent-sandbox

This is a sandbox repo used by @AnneBeyer to conduct agent-driven experimental
work.

The main branch is expected to hold only the shared setup: agent instructions,
the journal renderer and the report publishing workflow.

Experiments are expected to happen in dedicated branches, possibly with
accompanying PRs, but those PRs are never meant to be merged.

Agent instructions are in [AGENTS.md](AGENTS.md); [CLAUDE.md](CLAUDE.md)
imports them for Claude Code.

## Credits

The setup is modeled on [ogrisel/cursor-sandbox](https://github.com/ogrisel/cursor-sandbox)
by Olivier Grisel: the empty-main / one-branch-per-experiment workflow, the
`AGENTS.md` guidance on resource-conscious experiments, and the idea of a
chronological Markdown journal published per pull request to GitHub Pages.
The tooling here was written independently and adapted for Claude Code, local
and SSH-based GPU runs.

## One-time setup

### GitHub Pages

Rendered journals are pushed to the `gh-pages` branch by
`.github/workflows/publish-journals.yml`. After the first pull request with a
journal has run it, set **Settings → Pages → Build and deployment** to
**Deploy from a branch**, branch `gh-pages`, folder `/ (root)`.

### Dedicated agent SSH key

Agents reach the AWS GPU hosts with their own key pair, never yours. This lets
you revoke the key without touching your own access.

1. Create the key pair on your machine:

   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/agent_sandbox_ed25519 -C agent-sandbox
   ```

   An empty passphrase is reasonable for this key. A passphrase handed to the
   agent alongside the key protects nothing, and a dedicated key that you can
   revoke at any time is what protects your account. Ideally, register the key
   for a dedicated, unprivileged user on each host.

2. Append `~/.ssh/agent_sandbox_ed25519.pub` to `~/.ssh/authorized_keys` on
   the jump host and on every GPU host.

3. Describe the hosts in `~/.config/agent-sandbox/env`. This file stays out of
   the repository, because the repository is public:

   ```bash
   # alias=user@private-ip, separated by spaces
   AGENT_SSH_HOSTS="gpu1=ubuntu@10.0.1.12 gpu2=ubuntu@10.0.1.13"
   # jump host, user@host[:port]
   AGENT_SSH_JUMP="ec2-user@bastion.example.com"
   # optional, defaults to ~/.ssh/agent_sandbox_ed25519
   # AGENT_SSH_KEY_FILE=~/.ssh/agent_sandbox_ed25519
   ```

4. Run `tools/agent_ssh.sh setup`, then `tools/agent_ssh.sh check`.

### Where agents can use SSH

| Where the agent runs | Local experiments | SSH to GPU hosts |
|---|---|---|
| Local session (Claude desktop app or CLI) | yes | yes, with the setup above |
| Self-hosted cloud environment (your own runner) | on the runner | yes: export the `AGENT_SSH_*` settings and the key in `AGENT_SSH_KEY` on the runner |
| Anthropic-hosted cloud session | in the cloud VM | no: only outbound HTTP(S) is allowed |

For a runner, `AGENT_SSH_KEY` holds the private key itself. The setup script
writes it to `~/.ssh/agent-sandbox/id_ed25519`. Do not put the key in the
environment variables of an Anthropic-hosted environment, because anyone who
uses the environment can read them, and the session could not use the key
anyway.

## Tools

```bash
pixi run render --journal <experiment>/journal --site /tmp/site
```

```bash
pixi run test-tools
```
