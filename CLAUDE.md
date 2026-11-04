@AGENTS.md

## Claude Code specific instructions

- Start every experiment on a new branch off `main` and push it. Open a draft
  pull request once there is a first journal entry, so the report is published.
- To tell where you are running, check `CLAUDE_CODE_REMOTE`. It is `true` in
  cloud sessions and unset in local ones.
- Anthropic-hosted cloud sessions can make only outbound HTTP(S) connections,
  through a proxy, so they cannot reach the GPU hosts over SSH. If an
  experiment needs a GPU and you are in such a session, say so and hand the
  remote part to the operator instead of trying workarounds.
- Local sessions and self-hosted cloud environments can use
  `tools/agent_ssh.sh` as described above.
