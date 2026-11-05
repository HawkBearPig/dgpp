# DGPP packaged release

This package serves supported checkpoints on Linux/aarch64 DGX Spark (GB10).
It includes the server and CUDA runtime libraries, but not model weights.
Run the following commands from the extracted package root on rank 0.
Use a head with working DNS and outbound access for package/Python/model
downloads, or prepare them offline first. A reachable RoCE address does not
imply internet access. Peers receive checkpoints from rank 0 over SSH.

## Getting started

### 1. Install the dependencies

1. On every node, check `nvidia-smi` works and Python is at least 3.10.
   Use the NVIDIA driver supplied with your DGX OS installation.
2. Install the runtime and transfer tools on every node:

   ```bash
   sudo apt-get update
   sudo apt-get install python3 python3-venv rdma-core libibverbs1 \
     ibverbs-providers libnl-3-200 libnl-route-3-200 libstdc++6 \
     openssh-client openssh-server rsync curl jq zstd iproute2 \
     libavformat60 libavcodec60 libswscale7
   ```

The package bundles cudart, cuBLAS and cuBLASLt; no compiler or CUDA toolkit is needed
to run it. FFmpeg runtime libraries are required for the default video-enabled
build. Install `poppler-utils` on rank 0 to enable PDF file inputs. PCRE2
is built statically when system development files are absent; a build using
system shared PCRE2 also requires that runtime library on every node. The
PCRE2 license is included under `docs/licenses/`.

Keep libibverbs installed even for a single node: the server links
it. Multi-node production uses it for RoCE communication. The build baseline
is Ubuntu 24.04; older distributions may not provide compatible glibc/libstdc++.

### 2. Verify the package and choose a deployment

1. Run `sha256sum -c MANIFEST.sha256`. Direct CMake installs have no manifest.
2. Choose a model-specific filename and copy its matching template. For example:

   ```bash
   # Four Sparks, GLM-5.3-Flash hybrid NVFP4/FP8:
   CONFIG=deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json
   test -f "$CONFIG" || cp "${CONFIG%.json}.example.json" "$CONFIG"
   ```

   For two Sparks, set `CONFIG=deploy/cluster_qwen-3.8-flash-next_fp8_w2.json`
   before copying; for one, use
   `CONFIG=deploy/cluster_qwen-3.8-flash-next_nvfp4_w1.json`.
   See [deployment filenames](../deploy/README.md) for the templates and the knobs that vary them.
3. Check the JSON's `model` and `world_size`. Leave its engine settings
   unchanged for the first run. Pass this file explicitly with `--config`.
   `CONFIG` is a shell variable for brevity; set it again in a new terminal.

### 3. Configure the nodes and fabric

1. Run `test -f .env || cp .env.example .env`.
2. Edit `DGPP_NODES` in `.env`: use space-separated addresses, rank 0 first,
   or `127.0.0.1` for a single node. Set `DGPP_SSH_USER` for peers.
3. From **rank 0**, run `ssh USER@PEER_ADDRESS hostname` for each peer. Verify
   its host-key fingerprint through a trusted console or administrator before
   accepting it. Install your public key with
   `ssh-copy-id -i /path/to/key.pub USER@PEER_ADDRESS`; do not overwrite existing
   keys or disable host-key checking. Load a passphrase-protected key into your
   SSH agent. Confirm `ssh -o BatchMode=yes USER@PEER_ADDRESS hostname` succeeds
   for every peer before continuing. Incoming SSH access to rank 0 does not
   establish outgoing access to peers.
4. For multiple nodes, run:

   ```bash
   python3 scripts/discover_roce.py --config "$CONFIG"
   ```

5. Copy the appropriate device names and GID indices into `.env`. Match
   lane order by subnet across nodes; choose one or two lanes. Merge per-node
   differences into `DGPP_NODE_OVERRIDES`. See [networking](networking.md).
   Discovery shows the deployment's selected lane order and local candidates;
   neither proves RDMA connectivity. Failed probes retain other nodes' results.

Do not source `.env`; scripts parse it as data. Leave cache paths unset
to use the standard `~/.cache/huggingface/hub` on each node.

### 4. Download once and sync

Check free disk space on every node first (`df -h` on each cache filesystem).
The default GLM-5.3-Flash weights occupy about 181.7 GiB; allow roughly 250 GiB
per node for that checkpoint, one four-rank resident image and transfer headroom.
Other models and extra cache variants need their own budget. Doctor reports
free space but does not predict the combined future disk footprint.

On rank 0:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-download.txt
python scripts/download_model.py --config "$CONFIG"
```

Use `hf auth login` first if authentication is required. Only rank 0 downloads
from Hugging Face; it syncs the selected snapshot and blobs to peers
sequentially using rsync/SSH. Credentials and other cached revisions are not
copied. Peers need space for the full checkpoint plus their resident cache.

If rank 0 already has the checkpoint, use `--sync-only` instead. Use
`--verify-only` to check active snapshots on all nodes without downloads or
writes. Add `--local-only` to download or verify without contacting peers.
Sync-only and verify-only do not require Hugging Face packages or internet
access. For a disconnected head with an empty cache, see the source project's
[offline preparation guide](https://github.com/HawkBearPig/dgpp/blob/master/docs/getting-started.md#offline-preparation).

### 5. Install and start

For a **single node**, use the extracted binary:

```bash
python3 scripts/dgpp-cluster doctor --config "$CONFIG" --bin bin/dgpp-serve
python3 scripts/dgpp-cluster up --config "$CONFIG" --bin bin/dgpp-serve
```

For **multiple nodes**, install the same tarball on each node using the
launcher. Replace `VERSION` with the value in `MANIFEST`:

```bash
python3 scripts/dgpp-cluster install /path/to/dgpp-VERSION.tar.zst --config "$CONFIG"
python3 scripts/dgpp-cluster doctor --config "$CONFIG" --release VERSION
python3 scripts/dgpp-cluster up --config "$CONFIG" --release VERSION
```

Fix failed doctor checks before starting. Installation uses `DGPP_RELEASE_DIR`
and refuses to replace an existing version. Named releases use each node's
installed binary and bundled libraries; `--bin` stages only the executable,
so peers need compatible runtime libraries separately.

### 6. Check and stop the service

```bash
curl --fail http://127.0.0.1:18080/v1/models
python3 scripts/serve_api_check.py 127.0.0.1 18080
python3 scripts/dgpp-cluster status --config "$CONFIG"
python3 scripts/dgpp-cluster down --config "$CONFIG"
```

HTTP defaults to all IPv4 interfaces (`0.0.0.0`); deployment `http.bind_host` and `http.port` select
another address or port. Use the same config and any `--log-dir` override for
start, status and stop. Use an SSH tunnel or authenticated TLS proxy for remote
clients; DGPP has no authentication or TLS.

Run `python3 scripts/dgpp-cluster paths --config "$CONFIG"` to locate
logs. A running deployment must be stopped before another `up`, or explicitly
replaced with `up --replace`. Stop jobs from older launchers with their
original launcher; unrecorded processes are not adopted or killed.

Source-only benchmarks and reference generators are not included. Resident
weight caches are built locally at first load; allow additional NVMe space.

## Retain an NVMe prefix cache across upgrades

If the deployment uses persistent `nvme_cache`, automatic cache selection
changes with the server build. Before upgrading, record the existing
16-character binding from the startup log or its directory under
`nvme_cache.path`. Add `"reuse_binding": "<existing binding>"` inside the
deployment's `nvme_cache` block and keep its capacity unchanged. Leave this
setting in place on subsequent upgrades to continue using that cache.

This explicitly asserts that the model artifacts and cached-state
interpretation remain compatible. File formats, geometry, page checksums
and agreement across ranks are still checked; rejected reuse leaves the
selected cache intact. The package includes the full guide at
`docs/nvme-cache.md` and the configuration reference at `docs/configuration.md`.
The [online example](https://github.com/HawkBearPig/dgpp/blob/master/docs/nvme-cache.md#across-restarts)
and [upgrade/rollback operations](https://github.com/HawkBearPig/dgpp/blob/master/docs/operations.md#install-upgrade-roll-back)
also cover this procedure.
