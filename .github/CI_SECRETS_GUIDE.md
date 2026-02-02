# CI & Secrets Guide

This guide shows how to rotate a compromised DeepSeek API key and store the new key securely for GitHub Actions.

1) Rotate the compromised key (DeepSeek dashboard)
- Log in to your DeepSeek account and revoke the compromised key immediately.
- Create a new API key and copy it.

2) Set the new key as a GitHub Actions secret (recommended)
- Using `gh` (GitHub CLI):

```bash
# Authenticate first: gh auth login
gh secret set DEEPSEEK_API_KEY --body "NEW_KEY_HERE" --repo OWNER/REPO
```

- Using the web UI: Repository → Settings → Secrets and variables → Actions → New repository secret. Name: `DEEPSEEK_API_KEY`, Value: (new key).

3) Update workflows to read the secret
- In your GitHub Actions workflow, reference the secret as an environment variable:

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run agent
        env:
          DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
        run: |
          python tools/deepseek_scaffold/agent/agent_swe.py
```

4) Local development
- Use a `.env` file (add `.env` to `.gitignore`) and load it with `python-dotenv` or `dotenv` for Node. Do NOT commit `.env`.

5) Rotate procedure checklist
- Revoke the old key in DeepSeek.
- Create new key.
- Update GitHub Actions secret (`gh secret set ...`).
- Inform team members and update any local `.env` copies.
- Run a quick smoke test of workflows.

6) Quick GH Actions troubleshooting
- If the workflow cannot access the secret, ensure repository-level secrets exist (environment or repo scope). For organization secrets, ensure the repo has access.

7) Optional: Automatic rotation with HashiCorp Vault / Azure Key Vault
- For production, use a secrets manager and retrieve secrets at runtime instead of storing long-lived keys in CI.
