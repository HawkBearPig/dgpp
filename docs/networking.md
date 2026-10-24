# Network and cache configuration

## SSH and control addresses

`DGPP_NODES` identifies hosts for SSH and TCP coordination, in rank order.
These addresses may be management IPs, fabric IPs, or a mixture. SSH/control
traffic and RDMA can share the same interfaces; a separate management network
is optional. The address list and RoCE device/GID selection configure those
roles independently.

Rank 0 runs locally and must reach every peer's listed address over SSH using
`DGPP_SSH_USER` without an interactive prompt. Peers must resolve and reach
rank 0's listed address for TCP coordination. The addresses can be on different
subnets when routing provides that connectivity.

A supported layout has a separate management IP only on rank 0, with nodes
1, 2 and 3 addressed solely by their fabric IPs. For example, suppose rank 0's
operator-facing management IP is `192.0.2.10` and the four nodes' fabric IPs
are `198.51.100.11` through `198.51.100.14` (illustrative addresses):

```dotenv
DGPP_NODES="198.51.100.11 198.51.100.12 198.51.100.13 198.51.100.14"
```

You can log into rank 0 at `192.0.2.10` and run setup there. Peers use its
fabric address, `198.51.100.11`, for coordination; rank 0 uses each peer's
fabric address for SSH, discovery, binary staging and checkpoint transfers.
Rank 0's separate management connection can provide internet access for
downloads; peers receive their checkpoints from rank 0.

Alternatively, put rank 0's management address first if every peer can reach
it, while keeping the peers' fabric addresses in the remaining entries.
Setup accepts either layout. The choice depends on actual routing, not on
whether an address is labelled management or fabric. Select RoCE devices and
GIDs afterward, including the interfaces carrying these addresses when desired.

Keep fabric/journal TCP ports restricted to cluster members. The defaults
are 29970 and 29971; set them in `.env`. These internal protocols have no
authentication and must not be exposed to an untrusted network.

## RoCE lanes

For multi-node runs, set `DGPP_ROCE_DEVICES` to a space-separated ordered
list of local verbs devices. Discover the actual names on your machines:

```bash
python3 scripts/discover_roce.py --config "$CONFIG"
```

Set `CONFIG` to your deployment filename. Run this on rank 0 after setting
node addresses and SSH access. Without
`--config`, it inspects only the local machine. It prints the verbs-to-Linux
interface mapping, addresses, MTUs, locally eligible GID indices, and candidate `.env`
settings without changing files or networking. On Spark, names can look like
`"rocep1s0f0 roceP2p1s0f0"`; use the discovered values for your hardware.
Match corresponding lanes by subnet before copying the suggestions. If
several GIDs or more than two devices are locally eligible, choose the intended fabric.

The deployment lane order shows the configured device order (including
per-node overrides), or the automatic selection when none is configured.
It does not identify a proven "primary" fabric: an active device with an IP
may still be disconnected from the other ranks. Failed SSH probes leave their
node inventory unknown; discovery prints the other results and exits nonzero.
`--json` returns separate `inventories`, `selections` and `errors` objects keyed
by host, rather than treating an unreachable node as a host with no devices.

If `DGPP_ROCE_DEVICES` is omitted,
DGPP discovers active Ethernet RDMA devices and sorts their names. Explicit
selection is preferable on hosts with several fabrics. The native transport
uses port 1 of each selected device; multi-port HCA selection is not supported.

Use one or two lanes, with the same count and corresponding network order on each rank. Device
names need not match between hosts. Without explicit GID indices, the verbs
layer selects a non-link-local RoCE-v2 GID. `DGPP_ROCE_GID_INDICES="3 3"`
pins one index per explicitly named device. Check link state, IP/GID
assignment, routing and consistent MTU on the selected network interfaces.
Discovery does not prove that corresponding lanes can exchange RDMA traffic.

A DGX Spark's one ConnectX-7 port appears as two verbs devices, one per
PCIe function (`rocep1s0f0` and `roceP2p1s0f0` here), each behind its own
x4 link, and both carry traffic over the one cable. Name both as lanes:
the bus stripes its bulk class (the prefill boundary folds) across the
lanes, keeps the decode-class messages on the first, and derives each
queue pair's pace from the port rate over all inbound queue pairs, so the
port's cap is unchanged while the host-side link doubles. Measured on
DeepSeek-V4.1-Flash-1017 on four nodes (2026-10-18, the same binary):
cold prefill 0.514 / 1.572 / 4.715 / 17.916 s at 512 / 2K / 8K / 32K
tokens with both functions against 0.523 / 1.619 / 5.152 / 19.591 on one
(−8.5 % at 8K and 32K), decode unchanged. Give the second function an
address on its own subnet on every node and pin a GID index per device.

Read-only starting points are `rdma link`, `ip -br address`, `ip link`, and
`scripts/dgpp-cluster doctor --config "$CONFIG"`.
Only run the fabric probes on an idle test allocation;
they open QPs, use GPU memory and generate traffic.

## Per-node differences

Shared defaults and overrides live in `.env`, not each deployment JSON:

```dotenv
DGPP_ROCE_DEVICES="rocep1s0f0 roceP2p1s0f0"
HF_HUB_CACHE="~/models/hub"
DGPP_RESIDENT_CACHE_DIR="~/dgpp/resident-cache"
DGPP_NODE_OVERRIDES='{"192.0.2.12":{"DGPP_ROCE_DEVICES":"rocep1s0f0 roceP2p1s0f0","HF_HUB_CACHE":"/srv/models/hub"}}'
```

Override keys must match a host in `DGPP_NODES` exactly. Accepted settings
include RoCE devices, GID indices, HF hub cache, resident cache directory,
`DGPP_LOG_LEVEL` and `DGPP_MLOCK` (the L2 weight prefetcher's knobs are the
engine's `l2_prefetch*` keys since 2026-10-05, not node settings). `~/`
expands on the destination host; `$VARIABLE` substitution
and shell commands are not supported. `HF_HUB_CACHE` takes precedence over
`HF_HOME/hub`. The launcher forwards only these allowlisted site values, not
`.env` or its credentials. Direct native programs do not parse `.env`; use
the wrappers, a resolved config, or explicitly export their settings.

## Memory registration failures

`dgpp-serve` raises its soft `RLIMIT_MEMLOCK` to the existing hard limit before
CUDA/RDMA setup. It does this even with `DGPP_MLOCK=off`, which only disables
the later optional `mlockall(MCL_CURRENT)` call. The bus starts before resident
model construction and before that optional pin.

Set `DGPP_LOG_LEVEL=debug` in `.env` or export it before launching to log the
limits before and after preparation, staging geometry, and each registration.
Both staging-block and bus-slab registration errors include the device, byte
count, access flags, CUDA pointer type/device, and the soft/hard limits and
`VmPin` captured immediately before the failed attempt, even at INFO level.

An `ENOMEM` registration failure can reflect a cumulative process memlock
limit. Registering one shared buffer on two devices charges its pages twice.
Compare the failing rank's logged limits with the successful rank and with any
standalone probe; rank 0 inherits the launcher's limits while peers inherit
their SSH session's limits. A 5 MiB allocation succeeding once does not prove
that a second registration fits. If the hard limit is insufficient, adjust the
launching session/service's memlock limit and verify the next startup log.
If the limits already permit the registrations, retain these diagnostics and
collect the kernel log for the same attempt to investigate pinning, DMA mapping
or driver resource failures.

### systemd launchers

A systemd unit's single-value `LimitMEMLOCK=` sets both the soft and hard
limits. With `LimitMEMLOCK=8M`, the server's soft-limit preparation leaves
both at 8 MiB; it cannot raise the hard limit. For a launcher running as a
system service, add this drop-in (replace `dgpp-launcher.service` with the
unit that starts your launcher or server):

```sh
sudo systemctl edit dgpp-launcher.service
```

```ini
[Service]
LimitMEMLOCK=infinity
```

Stop the existing DGPP deployment, then reload systemd and restart the
launcher service so newly started ranks inherit the new limit:

```sh
sudo systemctl daemon-reload
sudo systemctl restart dgpp-launcher.service
systemctl show dgpp-launcher.service -p LimitMEMLOCK -p LimitMEMLOCKSoft
```

The unit should report `infinity` for both limits. With
`DGPP_LOG_LEVEL=debug`, verify that each rank's startup log reports
`RLIMIT_MEMLOCK soft unlimited, hard unlimited`. Peer ranks inherit their
SSH session's limits, so check those separately if a peer still reports a
finite hard limit. Changing the unit does not update an already-running
rank's limits.

For a user service (`systemctl --user`), `LimitMEMLOCK=infinity` cannot
exceed the user manager's inherited hard limit. An administrator must first
raise that limit (for example, on the user's `user@.service` instance), and
the user manager must be restarted before relaunching the deployment. See
the resource-limit settings in
[systemd.exec](https://www.freedesktop.org/software/systemd/man/latest/systemd.exec.html).

## HTTP exposure

HTTP defaults to `0.0.0.0:18080`, accepting traffic on every IPv4 interface.
Setup prompts for the bind address; `scripts/setup.sh --http-bind ADDRESS`
sets it unattended. Choose `127.0.0.1` for local-only access or a specific
local LAN IPv4 address for one interface. Set `http.bind_host` and `http.port`
in the deployment JSON to override site defaults. The server does not enforce
API keys and does not implement TLS.

For individual remote access, use an SSH tunnel. For shared access, keep
DGPP on localhost behind an authenticated TLS reverse proxy. A starting
configuration is [deploy/nginx.example.conf](../deploy/nginx.example.conf).
Replace its hostname, certificate paths and password file before installation;
it is not installed or enabled automatically. Disable proxy response buffering
so SSE tokens arrive as produced, and set timeouts for long prefill requests.
See the NGINX documentation for [proxy buffering](https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_buffering)
and [basic authentication](https://nginx.org/en/docs/http/ngx_http_auth_basic_module.html).

The bundled diagnostic clients use plain HTTP to a host/port. Run them on
rank 0 or through a tunnel; they do not implement proxy authentication or TLS.
