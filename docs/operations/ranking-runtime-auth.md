# Ranking Runtime Auth, Drain Control, and Network Policy

The ranking coordinator (TCP `8013`) and ranking runners (TCP `8014`) speak a
small length-prefixed protocol: `R` rank, `B` batch rank (runner only), `H`
health, `M` metrics, `D` drain and `U` undrain (runner only). This page covers
how those connections are authenticated, who may change runner drain state,
and the Kubernetes NetworkPolicies around both ports.

## Connection Auth

Every client connection (recommendation-service and ranking-service to the
coordinator, coordinator to runners, `runner_healthcheck` probes) may start with
an `A` frame:

```json
{"v": 1, "ts": 1800000000, "nonce": "<32 hex chars>", "mac": "<hex>"}
```

- `K = HMAC-SHA256(SECURITY_INTERNAL_SERVICE_KEY, "ranking-runtime-v1")`
- `mac = HMAC-SHA256(K, "ranking-auth|v1|<ts>|<nonce>")`, compared with
  `hmac.compare_digest`.
- The server accepts `ts` within ±60 seconds of its clock and rejects nonces it
  has already seen (bounded in-memory LRU, 65,536 entries per process).
- After a successful handshake the rest of the connection is trusted; frames are
  not signed one by one.

What this does and does not protect:

- It stops pods without the internal key from ranking, reading metrics, or
  probing health over the protocol.
- It does not encrypt traffic or bind the handshake to later frames. An on-path
  attacker could still observe or hijack an authenticated connection. Use the
  NetworkPolicies below (and a mesh/mTLS if your threat model needs it).
- The replay cache is per process. A captured handshake replayed to a different
  runner within the 60 second window is accepted by that runner. Under a flood
  of valid handshakes, older nonces can also be evicted while still inside the
  window. Both cases need the internal key or a captured handshake first.
- Keep node clocks synchronized (NTP). Skew beyond 60 seconds rejects
  otherwise valid handshakes with reason `expired`.

### Modes

`RANKING_RUNTIME_AUTH_MODE` (Helm: `appConfig.ranking.runtimeAuthMode`) is read
by every ranking client and server.

| Mode | Client | Server |
| --- | --- | --- |
| `disabled` | never sends `A` | treats `A` as an unknown op (400) |
| `permissive` (default) | sends `A` when a key is set; keeps using a legacy server that answers `A` with 400 | accepts unauthenticated connections and counts them; still rejects bad handshakes |
| `required` | fails the connection unless `A` returns 200 | answers any other first frame with 401 and closes the connection |

- A handshake that is present but wrong (bad MAC, expired, replayed, malformed)
  is always rejected with 401 and the connection is closed, in both
  `permissive` and `required`.
- Without `SECURITY_INTERNAL_SERVICE_KEY`, clients send nothing and servers
  cannot verify handshakes. `required` therefore refuses to start without a key.
- With `ENVIRONMENT=production`, `disabled` is rejected at startup. The internal
  key is already mandatory there.

### Metric

`video_commerce_ranking_runtime_auth_total{role, result}` is recorded by the
servers. `role` is `coordinator` or `runner`. `result` is one of:

- `authenticated`: valid handshake.
- `unauthenticated`: connection used without `A` and accepted in permissive
  mode. Counted once per connection.
- `rejected`: connection used without `A` in required mode.
- `invalid_mac`, `expired`, `replayed`, `malformed`, `unsupported_version`:
  a failed handshake.

Coordinator metrics are exported through ranking-service `/metrics`, which
proxies the coordinator's `M` frame. Runners have no scrape target by default,
so runner-side results are visible only if you scrape runners via `M`.

## Flipping To `required`

Clients send `A` in permissive mode as soon as the key is set, so the
transition needs no code change, only configuration and observation.

1. Confirm every ranking pod has the same `SECURITY_INTERNAL_SERVICE_KEY`. Use
   the chart Secret or `existingSecret`. In Compose, set it in `.env` or
   `SECURITY_INTERNAL_SERVICE_KEY_FILE`.
2. Roll out this release with `runtimeAuthMode: permissive`. Wait until every
   recommendation-service, ranking-service, coordinator, and runner pod runs it.
3. Watch the coordinator counters for at least one full traffic cycle:

   ```promql
   sum by (result) (increase(video_commerce_ranking_runtime_auth_total{role="coordinator"}[15m]))
   ```

   Expect only `authenticated`. Any `unauthenticated` means an old client or a
   client without the key is still connecting. Any `invalid_mac` means a key
   mismatch. Fix the source before continuing.
4. Optional for runners: scrape one runner with
   `python -m video_commerce.ranking_runtime.runner_healthcheck` from inside the
   pod, or temporarily scrape runners via `M`. Runner clients are coordinators,
   so step 3 being clean usually implies the runners are clean too.
5. Set `runtimeAuthMode: required` (Compose: `RANKING_RUNTIME_AUTH_MODE=required`)
   and roll out again. Readiness probes and the runner preStop hook use
   `runner_healthcheck`, which reads the same environment and authenticates,
   so probes keep working.
6. After the rollout, `rejected` should stay at zero. A non-zero `rejected`
   rate after the flip means something is still connecting without auth.

Rollback: set the mode back to `permissive` and roll out. No data migration is
involved.

Key rotation: the protocol accepts a single key per process. Rotating
`SECURITY_INTERNAL_SERVICE_KEY` needs a coordinated restart. In `required`
mode, pods on different keys reject each other until the rollout finishes. To
avoid an outage, drop back to `permissive` for the rotation, roll every ranking
workload, then return to `required`. Note that permissive mode still rejects a
*wrong* key: it only tolerates a *missing* handshake.

## Runner Drain And Undrain

- `D` (drain) and `U` (undrain) are accepted only from loopback peers
  (`127.0.0.0/8`, `::1`, IPv4-mapped loopback). A remote peer gets 403
  `ranking_runner_lifecycle_loopback_only`, even when it is authenticated.
- The Helm `preStop` hook already runs
  `runner_healthcheck --host 127.0.0.1 --operation drain` inside the pod.
- To return a drained runner to service without restarting it, exec into the
  pod and run:

  ```bash
  python -m video_commerce.ranking_runtime.runner_healthcheck --host 127.0.0.1 --port 8014 --operation undrain
  ```

- The coordinator marks a runner protocol-drained when the runner answers a
  batch with `ranking_runner_draining`. The coordinator clears that state only
  after the runner returns an explicit HTTP-200 health response with status
  `ready` or `degraded`. The coordinator health check (its readiness probe
  every 10 seconds) issues these. Seeing the runner in DNS again does not clear
  the drain.

## NetworkPolicy

`networkPolicy.rankingRuntime.enabled` (default `true`) renders two ingress
policies:

- `ranking-coordinator`: TCP `8013` only from `recommendation-service` and
  `ranking-service` pods of the same release.
- `ranking-runner`: TCP `8014` only from `ranking-coordinator` pods of the same
  release.
- `networkPolicy.rankingRuntime.metricsFrom.namespaceSelector` and
  `.podSelector` add one more peer for the protocol port, for a scraper that
  speaks the `M` frame. Both selectors set means pods matching `podSelector` in
  namespaces matching `namespaceSelector`. Leaving both empty adds no rule.

Notes:

- NetworkPolicy needs an enforcing CNI (Calico, Cilium, AWS VPC CNI with
  network policy, etc.). Without one, the objects are accepted but do nothing.
- Exec probes and the preStop drain connect over `127.0.0.1` and are not
  affected. The TCP liveness probe comes from the kubelet on the node. Common
  CNIs allow node-to-local-pod traffic, but verify this on your CNI before
  enabling the policies in production.
- If a new workload needs to call the coordinator directly, add it to the
  template rather than disabling the policy.
