# aa-bigco-demo

## Setup

Dependencies are installed from JFrog Artifactory only (public PyPI is disabled). Configure auth via `UV_INDEX_JFROG_USERNAME` / `UV_INDEX_JFROG_PASSWORD` or `~/.netrc` for `fsaiartifact.jfrog.io`.

### Xray blocking (not network timeout)

Failures that look like "operation timed out" when fetching a wheel are often **JFrog Xray blocking** the download:

- **Unscanned artifacts**: If the artifact is not yet indexed in Xray, "Block unscanned artifacts" can block the download; the client may see a timeout instead of a clear error ([JFrog doc](https://jfrog.com/help/r/xray-how-to-prevent-pypi-client-or-other-clients-timeout-when-block-unscanned-artifacts-is-enabled)).
- **Vulnerable artifacts**: If the artifact was scanned and has violations (e.g. High severity), Xray blocks the download.

To verify a specific artifact (e.g. virtualenv) use the JFrog CLI:

```bash
# Find the artifact and its sha256
jf rt search "fsai-private-py-AUS-cache/**/virtualenv-<version>.whl"

# Check Xray: scanned and issues, or "not indexed/cached"
jf xr curl -s -XPOST "/api/v1/summary/artifact" -H "Content-Type: application/json" -d '{"checksums":["<sha256>"]}'
```

If a version is blocked or unscanned, pin a version that is scanned and allowed by your policy, or ask your JFrog/Xray admin to scan the repo or adjust block-unscanned behavior.

**watchdog (streamlit):** The only `watchdog` 6.0.0 cp311 macos wheel in the cache is **unscanned** in Xray, so `uv sync` / `uv add streamlit` can time out on it. There is no older cp311 macos version to pin to. See below for triggering scans as non-admin; otherwise ask your JFrog/Xray admin to scan the cache or add it to Xray indexing. Streamlit works without watchdog (falls back to polling).

#### What you can do as non-admin (Options 1 & 2)

**Option 1: `jf scan` / `jf audit` (local, self-service)**  
These bypass server-side indexing: they run the Xray indexer locally and scan against Xray’s vulnerability DB. You need a scoped access token with Xray permissions (not just Artifactory).

```bash
# Scan a specific wheel (e.g. after downloading it from public PyPI)
jf scan /path/to/watchdog-6.0.0-cp311-cp311-macosx_10_9_universal2.whl

# Or audit your project’s Python dependency tree
jf audit --python
```

Docs: [LaunchPad – how to run jf scan / jf audit locally](https://futuresecureai.atlassian.net/wiki/spaces/LaunchPad/pages/1144914071).

**Option 2: `POST /api/v1/scanArtifact` (when repo is indexed)**  
Requires “Manage Components” permission (not full admin), but **only works for artifacts in repos that are already in Xray’s Indexed Resources**. If the PyPI cache repo is not indexed (typical), this returns `{"error":"Failed to scan component"}`.

```bash
jf xr curl -s -XPOST "/api/v1/scanArtifact" -H "Content-Type: application/json" \
  -d '{"checksums":["6eb11feb5a0d452ee41f824e271ca311a09e250441c262ca2fd7ebcf2461a06c"]}'
```

**Sustainable fix (one-time admin request)**  
Server-side scanning requires: (1) repository indexing (admin-only), (2) a security policy, (3) a watch linking the policy to the repo. Ask your admin to add the PyPI cache repo to **Xray → Settings → Indexed Resources**. After that, new artifacts are indexed and scanned automatically.

- Raise on FPS board: <https://futuresecureai.atlassian.net/jira/software/c/projects/FPS/boards/25>
- Confluence: [indexing + policy + watch](https://futuresecureai.atlassian.net/wiki/spaces/DT/pages/1030520835), [access request process](https://futuresecureai.atlassian.net/wiki/spaces/DS/pages/694943772)

#### Workaround: use streamlit now (local watchdog wheel)

So that `uv add streamlit` (or `uv sync` with streamlit in `pyproject.toml`) works without waiting for the cache to be indexed, satisfy `watchdog` from a **local wheel** downloaded from public PyPI. Then the resolver uses that instead of the blocked artifact in JFrog.

```bash
# 1. Download the watchdog wheel from public PyPI (one-time)
mkdir -p local_wheels
curl -L -o local_wheels/watchdog-6.0.0-cp311-cp311-macosx_10_9_universal2.whl \
  "https://files.pythonhosted.org/packages/e0/24/d9be5cd6642a6aa68352ded4b4b10fb0d7889cb7f45814fb92cecd35f101/watchdog-6.0.0-cp311-cp311-macosx_10_9_universal2.whl"

# 2. Add the local wheel so the resolver uses it (cp311 macos only; adjust path for other platforms)
uv add ./local_wheels/watchdog-6.0.0-cp311-cp311-macosx_10_9_universal2.whl

# 3. Add streamlit (or run uv sync if streamlit is already in pyproject.toml)
uv add streamlit
```

`local_wheels/` is in `.gitignore` so the wheel is not committed. On other platforms (e.g. Linux, Windows or different Python), download the matching wheel from PyPI and add it the same way.

**Fail fast:** Timeouts are usually Xray blocks, not slow networks. Use a short HTTP timeout so blocked fetches fail quickly. uv retries 3× per request; with a 5s timeout that’s ~15s max per package (default 30s timeout × 3 ≈ 90s):

```bash
export UV_HTTP_TIMEOUT=5   # use 'export', not 'set' (zsh/mac)
uv sync
```

If you see "Failed to extract archive" with "current value: 1s", the env var wasn’t set (e.g. you used `set` instead of `export`). Run `export UV_HTTP_TIMEOUT=5` then retry.

### Pre-commit (install via Homebrew)

Pre-commit is not in `pyproject.toml` because its dependency `virtualenv` is blocked by Xray (403) or unscanned (timeout) in JFrog. Install pre-commit via Homebrew so you don't pull it from the private index:

```bash
brew install pre-commit
uv sync
```

Then run `pre-commit install` in the repo as usual.
