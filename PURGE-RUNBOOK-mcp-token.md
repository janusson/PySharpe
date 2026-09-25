# Purge runbook — leaked AgentRQ MCP bearer token (`.mcp.json`)

**Incident:** `.mcp.json` was committed to the public repo `janusson/PySharpe` and
contains a live bearer token for the AgentRQ MCP endpoint.

| Fact | Value |
| --- | --- |
| Exposed file | `.mcp.json` (repo root) |
| Repo visibility | PUBLIC |
| Introduced | commit `8666514` — 2026-07-14 ("Add agent skills, validation modules, and brokerage export") |
| Discovered | 2026-09-23 |
| Still at HEAD of `main` before fix | yes (`git log --oneline --all -- .mcp.json` → `8666514`) |
| Independently confirmed public | `GET https://raw.githubusercontent.com/janusson/PySharpe/main/.mcp.json` → HTTP 200, 315 bytes, token present |
| Token length | 173 chars (`?token=…` query parameter, MCP server key `agentrq-0f4YqKnJ5Ob`) |

Remediation already applied in the working tree (uncommitted):

- `git rm --cached .mcp.json` + `rm .mcp.json` (deleted from index and disk)
- `.gitignore` now ignores `.mcp.json`, `.codex/`, and `.claude/settings*.json`
  with a comment explaining why

### Other local files that carry the same token

A full-history blob scan found exactly **one** reachable blob containing
`token=eyJ…`: `.mcp.json`, reachable from the tip commit (`28e16c3`). No other
tracked file has ever contained it — so the `--path` list in Step 2 needs one
entry, not several. But two *untracked* local files do hold it:

| File | Tracked? | Ignored? | Action |
| --- | --- | --- | --- |
| `.mcp.json` | was (leak) | now yes | removed + history purge |
| `.codex/config.toml` | no | **was not** → now yes | rotate token; never track; keep local-only |
| `repomix-output.xml` | no | yes (`.gitignore:74`) | delete and regenerate after rotation — it packs the whole tree, so it mirrors any secret in the working dir |

Verification command used:

```bash
git rev-list --all --objects | awk '{print $1}' | sort -u \
  | git cat-file --batch-check='%(objecttype) %(objectname)' \
  | awk '$1=="blob"{print $2}' | git cat-file --batch | grep -c 'token=eyJ'
# → 1  (the .mcp.json blob)
```

**This is not sufficient.** `git rm` only removes the file from the tip; the blob
stays reachable in history and in GitHub's cached views forever. Two things must
happen: (1) rotate the credential, (2) rewrite history.

> ⚠️ Order matters: **rotate first**. Purge is defense-in-depth, not the fix. A
> public repo means assume the token is already in someone's dataset.

> 🚫 **Do not commit this file.** It is an operational runbook, not project
> documentation. Keep it untracked and delete it once every box below is ticked.

### Sequencing against the local changes already made

The working tree already has the `.mcp.json` deletion staged and `.gitignore`
updated. Land those *before* the rewrite, so the purge applies to a history that
already reflects the intent:

1. Rotate the token (Step 0).
2. Branch (`fix/remove-leaked-mcp-json`), commit `.gitignore` + the `.mcp.json`
   deletion, open a PR, merge. Pointless-looking history is fine — filter-repo
   strips the path from *every* commit, including the one that deletes it.
3. Run Steps 1–4 (rewrite on a fresh mirror clone, forced mirror push, purge).
4. Back in the working clone: `git fetch --all --prune` → `git reset --hard
   origin/main` → `git reflog expire --expire=now --all` → `git gc --prune=now
   --aggressive`.
5. Everyone else re-clones. Any old clone that pushes again re-introduces the blob.

---

## Step 0 — Rotate the AgentRQ token (do this before anything else)

1. In the AgentRQ dashboard, revoke the MCP server key `agentrq-0f4YqKnJ5Ob` and
   issue a new one.
2. Confirm the old token no longer authenticates (`curl` the endpoint with the old
   token → expect 401/403). Do this from a throwaway shell so the command is not in
   your history file.
3. Decide the replacement plumbing: credentials from an environment variable
   (`AGENTRQ_MCP_TOKEN`), read by `.mcp.json` at run time — not a literal in a
   tracked file. See Step 5.

## Step 1 — Check for forks (they keep the blob alive)
```bash
gh api repos/janusson/PySharpe/forks --jq '.[].full_name'
```

- **No forks** → history rewrite on the origin is sufficient.
- **Forks exist** → each fork retains the leaked blob until its owner rewrites or
  GitHub support purges it; open an issue/direct contact per fork, and include this
  in the disclosure note. (As of 2026-09-23: 0 forks.)

## Step 2 — Rewrite history with `git-filter-repo`

`git-filter-repo` is the maintained replacement for `git filter-branch`
(BFG is a Java wrapper for the old, deprecated, slow path — use filter-repo).

```bash
# Install (pick one)
brew install git-filter-repo
# or: python3 -m pip install --user git-filter-repo

# Work from a fresh MIRROR clone — never rewrite the clone you work in.
cd ~/Programming
git clone --mirror https://github.com/janusson/PySharpe.git pysharpe-purge.git
cd pysharpe-purge.git

# Remove the path from every commit on every ref
git filter-repo --path .mcp.json --invert-paths --force
```

`--invert-paths` means "drop this path, keep everything else". Run it once for
`.mcp.json`; if a full secret scan (Step 4) turns up more paths, add one
`--path <file> --invert-paths` pair for each.

### Before force-pushing, verify locally

```bash
# 1. The path must be gone from every commit object ever written
git log --all --oneline -- .mcp.json          # expect: empty
git rev-list --all --objects | grep -F '.mcp.json'   # expect: empty

# 2. The token string must not appear in any reachable blob
git rev-list --all --objects | awk '{print $1}' | sort -u \
  | git cat-file --batch-check='%(objecttype) %(objectname)' \
  | awk '$1=="blob"{print $2}' \
  | git cat-file --batch | grep -c 'agentrq'   # expect: 0

# 3. Sanity: commit count and tip content are otherwise unchanged
git log --oneline -5
```

## Step 3 — Force-push and clean up

```bash
# The mirror clone's remote is the original URL. Confirm before pushing:
git remote -v

# Overwrite every remote ref with the rewritten history
git push --force --mirror https://github.com/janusson/PySharpe.git
```

A mirror push replaces remote refs with exactly what the mirror has. **Check the
ref list before and after** so you don't delete branches you meant to keep:

```bash
git ls-remote --heads https://github.com/janusson/PySharpe.git > /tmp/before.txt   # run BEFORE the push
git ls-remote --heads https://github.com/janusson/PySharpe.git > /tmp/after.txt    # after
diff /tmp/before.txt /tmp/after.txt
```

In the working repo (`~/Programming/PySharpe`):

```bash
git fetch --all --prune
git reset --hard origin/main          # or rebase your local work onto the rewritten main
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

**Everyone with a clone must re-clone** — old local clones will re-push the leaked
blob on their next push if they are merged back in.

## Step 4 — Purge GitHub's cached views and scan for other secrets

- Dangling/old blobs stay fetchable by SHA on GitHub until support purges cached
  views. Open a request at <https://support.github.com/contact> (category: "Remove
  data") describing the leaked token SHA range, and ask them to run a garbage
  collection / purge of unreachable objects for the repo.
- Re-check after the rewrite (expect 404 once caches expire):

```bash
curl -s -o /dev/null -w "%{http_code}\n" \
  https://raw.githubusercontent.com/janusson/PySharpe/<old-commit-sha>/.mcp.json
```

- Scan the *entire* history for anything else, and schedule it in CI:

```bash
brew install gitleaks
gitleaks detect --source . --log-opts="--all" -v      # working clone, full history
gitleaks protect --staged                            # optional pre-commit use
```

Any other finding → repeat Steps 2–3 with its path. A `gitleaks` job in
`.github/workflows/` is a tracked roadmap item (PS-1.7).

## Step 5 — Prevent recurrence

```bash
# In the working repo: replace the file with a template that has no secret
cat > .mcp.json.example <<'JSON'
{
  "mcpServers": {
    "agentrq": {
      "type": "http",
      "url": "https://<your-endpoint>.mcp.agentrq.com?token=${AGENTRQ_MCP_TOKEN}"
    }
  }
}
JSON
git add .mcp.json.example
```

Then:

- `.gitignore` — `.mcp.json` ignored (done).
- `AGENTS.md` §5 currently instructs agents to push notifications to the AgentRQ
  dashboard, which is what dragged the credential into the repo. Reword it to
  "read the endpoint and token from the environment (`AGENTRQ_MCP_TOKEN`); never
  commit a populated `.mcp.json`", and note the token must be supplied by the
  operator.
- Enable **secret scanning + push protection** on the repo (Settings → Code security)
  so the next commit like this is blocked at push time.
- Keep the CI `gitleaks` job from Step 4 as the standing tripwire.

## Post-incident checklist

- [ ] AgentRQ token rotated and old token rejected by the endpoint
- [ ] Forks checked (0 = nothing further)
- [ ] `.mcp.json` removed from index and disk; `.gitignore` updated
- [ ] History rewritten on a mirror clone and verified locally (3 checks)
- [ ] `git push --force --mirror` done; remote ref list diffed before/after
- [ ] Local clones re-cloned or reflog-expired + gc'd
- [ ] GitHub support asked to purge cached/ unreachable objects
- [ ] Old-SHA raw fetch returns 404
- [ ] `gitleaks detect --all` clean (and added as a CI job)
- [ ] `.mcp.json.example` committed; `AGENTS.md` §5 reworded to env-var credentials
- [ ] Repo secret scanning + push protection enabled
