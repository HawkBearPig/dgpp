# Getting started

Run these steps on the first Spark in your cluster (rank 0).
A supported deployment uses one GB10 per node
on Linux/aarch64; choose a model and node count from the
[supported configurations](../README.md#supported-models-and-configurations).
For the abbreviated command sequence, see the [quickstart](../README.md#quickstart).

For compilation on an x86 Linux workstation, see [cross-compiling for Spark](cross-compiling.md).

## 1. Install the dependencies

1. Choose a rank-0 machine with working DNS and outbound access to GitHub,
   your OS package repositories, PyPI and Hugging Face (including its download
   hosts). Peers need cluster access; they do not fetch models from the Hub.
   Check the chosen head before installing anything:

   ```bash
   ip route
   getent hosts github.com pypi.org huggingface.co
   curl --fail --head --location --connect-timeout 10 https://huggingface.co
   ```

   A RoCE address does not imply internet access. Fix DNS/routing with your
   administrator if these fail, or use the [offline preparation](#offline-preparation)
   below. Success here does not test every download endpoint.

2. On **every node**, check `nvidia-smi` works. Use the NVIDIA driver supplied
   with your DGX OS installation. For source builds, peers need compatible CUDA
   13 cudart and cuBLASLt runtime libraries, but do not need `nvcc` or a compiler.
   Keep OS and runtime versions compatible across nodes.

3. On **every node**, install the runtime and transfer tools:

   ```bash
   sudo apt-get update
   sudo apt-get install python3 rdma-core libibverbs1 ibverbs-providers \
     libnl-3-200 libnl-route-3-200 libstdc++6 openssh-client openssh-server \
     rsync curl jq zstd iproute2
   ```

4. On **rank 0 only**, install build and downloader tools:

   ```bash
   sudo apt-get install git build-essential cmake pkg-config python3-venv libibverbs-dev poppler-utils
   ```

   PDF file inputs use `pdftotext` from `poppler-utils` on rank 0. String
   constraints use PCRE2: CMake uses an installed development package when
   available, otherwise downloads and builds the pinned static library.
   For an offline build, prefetch PCRE2 10.45 and set
   `-DFETCHCONTENT_SOURCE_DIR_PCRE2=/path/to/pcre2-10.45` during configuration.

   Check the CUDA 13 compiler with `nvcc --version`. If that command is missing,
   try `/usr/local/cuda/bin/nvcc --version`: the toolkit may already be installed
   outside your shell's `PATH`. See [build setup](#5-build-the-server-on-rank-0)
   for selecting it. Install the CUDA 13 toolkit through your DGX OS setup only
   if no suitable installation exists. Ninja is not required by these commands.

5. Check `cmake --version` is at least 3.25 and `python3 --version` is at least
   3.10. GCC 13, CMake 3.28.3 and CUDA 13.0.88 on Ubuntu 24.04 are the tested
   baseline.

6. On rank 0, clone the repository if you have not already done so:

   ```bash
   git clone https://github.com/HawkBearPig/dgpp.git
   cd dgpp
   ```

   Run the remaining commands from this directory in the same shell.

Keep libibverbs installed even for one node: the current server links it.
Multi-node production uses it for RoCE communication. Turning
`DGPP_ENABLE_IBV` off omits the server; it is not a single-node build option.
`rsync` is required on both ends of checkpoint transfers.

## 2. Copy a deployment template

1. Run **one** assignment matching your hardware. `CONFIG` is just a shell
   variable for the filename; each command below still passes `--config` explicitly.

   ```bash
   # One Spark: Qwen NVFP4 with FP8 dense projections.
   CONFIG=deploy/cluster_qwen-3.8-flash-next_nvfp4_w1.json
   ```

   ```bash
   # Two Sparks: Qwen NVFP4 with its n-gram table mapped from NVMe.
   CONFIG=deploy/cluster_qwen-3.8-flash-next_nvfp4_w2.json
   ```

   ```bash
   # Four Sparks: GLM-5.3-Flash hybrid NVFP4/FP8.
   CONFIG=deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json
   ```

2. Copy the matching template without replacing an existing local file:

   ```bash
   test -f "$CONFIG" || cp "${CONFIG%.json}.example.json" "$CONFIG"
   ```

3. Open the file named by `CONFIG`. Confirm `model` and `world_size` match your
   choice. Leave the engine settings unchanged for the first run. In a new
   terminal, set `CONFIG` again or pass the full filename to `--config`.

Names carry the model, the checkpoint quant and the world size — one template
per combination, each with MTP on and the slot count and cache budget that
measured best. The other shapes (plain decode with `--no-mtp`, another draft
depth, slot count or cache budget) are boot knobs; see
[deployment templates](../deploy/README.md) for the list and the knobs that
reproduce every retired variant.

## 3. Set your node addresses and SSH user

1. Create the site file if it does not already exist:

   ```bash
   test -f .env || cp .env.example .env
   ```

2. Open `.env`. For **one Spark**, set `DGPP_NODES="127.0.0.1"`.
   For **multiple Sparks**, replace the example addresses with your machines'
   addresses, separated by spaces. Put the machine running these commands
   first. Every node must be able to reach the first address.
3. Set `DGPP_SSH_USER` to the login used on the other nodes. That account needs
   write access to its cache and staging directories.
4. From **rank 0**, connect to each peer with `ssh USER@PEER_ADDRESS hostname`.
   Replace the placeholders with your chosen account and peer address. Compare
   the displayed host-key fingerprint against a trusted console or your
   administrator before accepting it. On the peer's console,
   `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub` prints its Ed25519 fingerprint.
   An SSH connection in the other direction does not establish this trust.
5. If rank 0 has no suitable SSH key, create one with `ssh-keygen -t ed25519`.
   Do not overwrite an existing key. For a passphrase-protected key, load it into
   your SSH agent with `ssh-add /path/to/private_key` before running the scripts.
   If no agent is running, start one with `eval "$(ssh-agent -s)"` first.
   Install **only the public key** on each peer:

   ```bash
   ssh-copy-id -i /path/to/key.pub USER@PEER_ADDRESS
   ssh -o BatchMode=yes -o ConnectTimeout=5 USER@PEER_ADDRESS hostname
   ```

   Repeat the batch-mode check for every peer. `Host key verification failed`
   means trust is missing or changed; verify the host's identity before updating
   known hosts. `Permission denied (publickey)` means the user/key or agent setup
   is wrong. A timeout or refused connection means routing or the peer's SSH
   server needs attention. Do not disable host-key checking to bypass an error.

Keep any existing credentials in `.env`. Do not run `source .env`: scripts
read the allowed settings as data, without executing shell expressions.
The first `world_size` entries in `DGPP_NODES` participate in the deployment.

## 4. Discover the RoCE devices (multiple nodes only)

1. Run:

   ```bash
   python3 scripts/discover_roce.py --config "$CONFIG"
   ```

2. Read the device-to-interface mapping, IP addresses, MTUs and GID indices for
   each node. DGX Spark verbs names may look like `rocep1s0f0` and
   `roceP2p1s0f0`; copy your discovered names, not these examples.
3. Read the configured or automatic **deployment lane order** above the inventory.
   This shows which devices serving would select, not which cables reach peers.
   Copy candidate `DGPP_ROCE_DEVICES` and, where provided, `DGPP_ROCE_GID_INDICES`
   into `.env` only after identifying the intended fabric. Choose one or two
   locally eligible devices; an IP or active port alone does not prove connectivity.
4. Match the **subnet order** across nodes. Lane 0 on each node must reach
   the other nodes' lane 0, and likewise for lane 1. Use the printed
   `DGPP_NODE_OVERRIDES` when device names or GID indices differ.

Discovery does not change networking or test RDMA traffic. Failed nodes are
reported individually; other inventories are retained and the command exits
nonzero. Fix SSH access and rerun, or omit `--config` to inspect only the local
host (without deployment settings). `--json` returns `inventories`, `selections`
and `errors` objects keyed by node address; failed nodes appear in `errors`, not
as empty inventories. If no locally eligible device appears, check cabling,
interface IP assignments and the RDMA driver. If more
than two appear, select the pair connected to the intended fabric. With these
settings omitted, serving auto-selects active Ethernet RDMA devices; explicit
selection avoids choosing an unintended network.

## 5. Build the server on rank 0

```bash
cmake --preset release
cmake --build --preset release -j 4
```

This creates `build-release/dgpp-serve` with `-O3` optimization and no debug
symbols. The launcher copies it to peers; their CUDA and verbs runtime
libraries must already be installed. Leave `DGPP_BUILD_DIR` unset unless you
use a custom build directory. The separate `ci` preset retains debug symbols
in `build-ci/` for [testing](testing.md); deploy that build explicitly with
`--bin build-ci/dgpp-serve` when diagnosing a problem.

CMake searches `PATH` and the conventional `/usr/local/cuda/bin` location.
For a different installation, or after a failed configure, select the compiler
explicitly and clear the old configure cache:

```bash
CUDACXX=/path/to/cuda/bin/nvcc cmake --fresh --preset release
cmake --build --preset release -j 4
```

Replace the path with your CUDA 13 installation. `--fresh` resets CMake's
configuration, not your source or model cache.

## 6. Download once and sync to peers

Before downloading, check storage on **every node**, for example
`df -h "$HOME/.cache"` and `ssh USER@PEER_ADDRESS 'df -h "$HOME/.cache"'`.
If that directory does not exist, check its existing parent; if you selected
different cache paths in `.env`, check those disks instead.
The default GLM-5.3-Flash checkpoint has about **181.7 GiB of weights**.
Budget roughly **250 GiB per node** for this checkpoint, one four-rank resident
image and transfer headroom. This is a planning estimate, not an enforced quota;
other models, revisions and resident-cache variants need their own budget.
`doctor` reports free space and writability, but does not calculate the combined
future download/resident-cache footprint. Existing model files still occupy disk.

1. On rank 0, prepare the downloader:

   ```bash
   python3 -m venv .venv
   . .venv/bin/activate
   python -m pip install -r requirements-download.txt
   ```

2. If the model requires authentication, run `hf auth login` on rank 0.
   Public checkpoints do not require a login.
3. Run **one** command on rank 0:

   ```bash
   python scripts/download_model.py --config "$CONFIG"
   ```

The script downloads the complete checkpoint to rank 0, then syncs its selected
snapshot and referenced blobs to each peer sequentially over rsync/SSH. Peers
do not download from Hugging Face. Matching files are checksum-checked and
reused; interrupted transfers can be resumed by rerunning the command.
Other models, other cached revisions, `.env`, and HF login tokens are not copied.

The default on each node is the standard `~/.cache/huggingface/hub` directory.
Leave the cache settings unset to use it. `HF_HUB_CACHE`, `HF_HOME`, and
per-node cache overrides are supported if you need another disk. Each node
needs space for the **full checkpoint**, its resident image cache, and temporary
transfer files—not just its tensor-parallel share of the download.

If the checkpoint is already downloaded on rank 0, skip the Hub entirely:

```bash
python scripts/download_model.py --config "$CONFIG" --sync-only
```

Check the active snapshot on every node without modifying files:

```bash
python scripts/download_model.py --config "$CONFIG" --verify-only
```

Use `--local-only` to verify or download on rank 0 without contacting peers.
For a pinned revision, use `--revision COMMIT --activate` with the download
command. Sync selects the same revision on peers after verification. Stop
deployments using the checkpoint before changing it. Structural verification
checks metadata and shard lengths; rsync checks file contents during sync.

### Offline preparation

The quickstart assumes an internet-connected head. For a disconnected head,
first provision compatible system packages, drivers and CUDA through your
site's offline package process; a Git bundle and Python wheels do not replace
those dependencies.

1. On a connected machine, clone the source and create a transferable bundle:

   ```bash
   git clone https://github.com/HawkBearPig/dgpp.git dgpp-source
   git -C dgpp-source bundle create dgpp.bundle --all
   ```

2. On a connected Linux/aarch64 machine with the same Python version as rank 0,
   download Python packages without installing them:

   ```bash
   python3 -m pip download -r dgpp-source/requirements-download.txt -d dgpp-wheels
   ```

3. Transfer `dgpp-source/dgpp.bundle` and `dgpp-wheels/` to the disconnected head
   with SCP or your site's approved transfer method. In the receiving directory:

   ```bash
   git clone dgpp.bundle dgpp
   cd dgpp
   python3 -m venv .venv
   . .venv/bin/activate
   python -m pip install --no-index --find-links ../dgpp-wheels -r requirements-download.txt
   ```

4. Follow the configuration and build steps above. If the full checkpoint is
   already cached on the head, `python scripts/download_model.py --config "$CONFIG" --sync-only`
   distributes it without Hub access. `--sync-only` and `--verify-only` use the
   standard library and do not require the downloader's Python packages.
   For an empty cache, download on a connected machine first, then transfer the
   selected `models--ORG--NAME` cache directory to the head, preserving its
   `snapshots/`, referenced `blobs/`, `refs/` and relative symlinks. Do not copy
   HF credentials. Verify it with `--verify-only --local-only` before syncing.

## 7. Check the deployment and start it

```bash
python3 scripts/dgpp-cluster doctor --config "$CONFIG"
python3 scripts/dgpp-cluster up --config "$CONFIG"
```

Fix any failed preflight checks before starting. `doctor` checks GPU/platform,
libraries, cache files, writable paths, ports and RoCE selection without
starting ranks. Add `--local-only` to check rank 0 without SSH. It does not
prove end-to-end RDMA connectivity; startup checks the model's memory plan.

Wait for `READY`. A cold load takes longer than a resident-cache boot.
HTTP defaults to `127.0.0.1:18080`. To change it, add or edit the deployment's
`http` object, for example `"http": {"bind_host": "127.0.0.1", "port": 18081}`.
Keep localhost for now: the server has no TLS or authentication.

## 8. Send a request, then stop

1. On rank 0, check the endpoint and ask a question. The short example requests
   low reasoning effort because `max_tokens` includes reasoning tokens; without
   it, the model may reach the limit before producing answer content.

   ```bash
   curl --fail http://127.0.0.1:18080/v1/models
   MODEL=$(curl --fail -s http://127.0.0.1:18080/v1/models | jq -r '.data[0].id')
   jq -n --arg model "$MODEL" \
     '{model:$model,max_tokens:160,reasoning_effort:"low",messages:[{role:"user",content:"Name three primary colors."}]}' \
     | curl --fail http://127.0.0.1:18080/v1/chat/completions \
       -H 'Content-Type: application/json' --data-binary @-
   ```

2. Inspect or stop the deployment with the same config:

   ```bash
   python3 scripts/dgpp-cluster status --config "$CONFIG"
   python3 scripts/dgpp-cluster down --config "$CONFIG"
   ```

Use the URL printed at startup if you changed the HTTP port or bind address.
For access from another machine, run
`ssh -N -L 18080:127.0.0.1:18080 USER@HEAD_ADDRESS` on that machine, then
use its localhost endpoint. Shared access needs an authenticated TLS proxy;
see [networking](networking.md).

To find logs, run `python3 scripts/dgpp-cluster paths --config "$CONFIG"`.
Use the same config path and any `--log-dir` override for start, status and stop;
`down` and `status` without `--config` cover every recorded deployment.
`up` refuses an already running deployment; `up --replace` explicitly stops it
first. Stop processes created by an older launcher with that launcher before
upgrading—the current launcher does not adopt unrecorded processes.

## Optional development and evaluation tools

These are not needed to serve a model:

| Task | Additional setup |
|---|---|
| Host/Python tests | Build the test targets, then select the `host`/`python` CTest labels; see [testing](testing.md). `ci-local.sh` also runs GPU/RDMA suites, so use idle test hardware. |
| Tokenizer/template goldens and token-ID preparation | Install `requirements-tools.txt` in a venv. PyTorch reference modes need a compatible PyTorch installation separately. |
| Evaluation datasets | Run `python3 scripts/prepare_data.py download`. Token fixtures: `python3 scripts/prepare_data.py tokens --config "$CONFIG" --text /path/to/long-prompt.txt`. |
| HumanEval | Use an isolated evaluation environment and explicitly pass `--allow-code-execution`. Generated Python runs without a security sandbox. |
| Profiling | Install Nsight Systems (`nsys`). |
| Formatting | Install clang-format to enable the CMake formatting targets. |
| Release packaging | `tar`, `zstd`, `sha256sum`, Git and the CUDA toolkit on the packaging host. The package bundles cudart/cuBLASLt; peers still need the driver, verbs providers, libnl, libstdc++ and glibc. |
