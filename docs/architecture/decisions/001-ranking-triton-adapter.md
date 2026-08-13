# ADR 001: Ranking adapter with Triton-owned batching

- Status: Accepted, opt-in
- Date: 2026-08-12

## Context

The legacy ranking path centralizes request batching in a coordinator and sends
batches to PyTorch runner processes. Adding Triton behind that path would create
two independent queues and two batchers, obscuring deadlines and amplifying
overload. Ranking also contains stateful business behavior that does not belong
inside a stateless model server: feature assembly, DIN index expansion, value
calibration, recommendation construction, and top-k selection.

## Decision

`ranking-service` is the adapter and remains the only HTTP ranking interface.
In Triton mode it assembles fixed-contract tensors, performs one async gRPC
request for ranking requests within the configured model batch limit, and
applies business scoring and top-k after the response. A larger legal candidate
set is split into ordered, non-retried chunks under the same overall deadline;
outputs are concatenated before business scoring so candidates are not silently
truncated. Triton owns the inference queue, dynamic batching, ONNX Runtime model
execution, and only those concerns. The coordinator and runners are bypassed.

The ONNX model is exported offline from one exact, trained, verified checkpoint.
Checkpoint, ONNX, schema, DIN sidecar, and parity metadata are published in one
artifact record. A pod init step resolves the exact business model version,
verifies checksums and lineage, derives the numeric Triton version from the ONNX
checksum, and atomically creates an immutable model repository.

There is no request-level retry and no automatic fallback after a request is
sent to Triton. Capacity rejection is 429; model, artifact, transport, or tensor
contract failure is 503. Legacy remains the deployment-level rollback backend.

## Consequences

- Cross-process requests can form one dynamic inference batch without a second
  coordinator queue.
- Adapter preprocessing remains bounded and independently observable.
- Triton mode requires an exact model version and disables direct coordinator
  calls from recommendation serving.
- The first release supports DCN and DCN plus DIN on CPU. Trimodal, GPU,
  TensorRT, quantization, multi-replica failover, and automatic fallback remain
  out of scope.
- Enabling staging requires matched dedicated-Linux A/B evidence. Ranking-only
  improvement does not imply full recommendation-path improvement.
