# ring-gpu

Experimental project combining `io_uring` and GPU compute (`vulkano`) to calculate dot products for large vectors, with GPU-to-CPU completion synchronized via a Vulkan Timeline Semaphore.

The project models an async pipeline where data "arrives" from multiple sources (producer threads), is passed between threads via `IORING_OP_MSG_RING`, processed on the GPU, and sent back to the main `io_uring`.

## What the program does

1. Initializes Vulkan device/queue and compute pipelines.
2. Starts producer threads that generate random `f32` vectors.
3. Producers send vector pointers to the `listener ring` using `MsgRingData`.
4. The listener collects vector pairs and starts a GPU task (dot product).
5. After the GPU signals the `Timeline Semaphore`, the result is sent to the `main ring`.
6. The main thread reads completion events and prints the result.

---

## Components

- `src/main.rs`
  - Initializes GPU context.
  - Creates the main `io_uring` (`main ring`).
  - Starts `data_flow`.
  - Listens on `CQ` and receives GPU results.

- `src/data_flow.rs`
  - Creates the `listener ring`.
  - Starts multiple producer threads.
  - Receives vectors from `listener ring`, groups them in pairs, and calls `run_gpu`.

- `src/gpu.rs`
  - Initializes Vulkan (`vulkano`), compiles WGSL with `naga`.
  - Creates two compute pipelines:
    - `map_reduce` (element-wise dot for `vec4<f32>` + partial reduction),
    - `final_reduce` (further reduction of partial sums).
  - Creates a `Timeline Semaphore`.
  - After semaphore signaling, reads the final value and sends it to `main ring` via `MsgRingData`.

- `src/shader.wgsl`
  - WGSL shaders `map_reduce` and `final_reduce`.

---

## Threads and roles

The runtime has these roles:

1. `Main thread`
   - Waits for events in `main ring`.
   - Receives final `f32` values from the GPU path.

2. `Listener thread`
   - Waits for messages in `listener ring`.
   - Receives vector pointers from producers.
   - Launches a GPU task for each vector pair.

3. `Producer threads` (5 in current code)
   - Generate vectors with size `16_777_216`.
   - Send vector pointers to `listener ring`.

4. `GPU completion thread` (spawned inside `run_gpu`)
   - Waits for `timeline semaphore` to reach value `1`.
   - Reads the final output buffer.
   - Sends the result to `main ring`.

---

## ASCII data-flow diagrams

### 1) Global flow

```text
+-------------------+      MsgRingData(ptr to Box<[f32]>)      +--------------------+
| Producer threads  | ---------------------------------------> | Listener io_uring  |
| (vector generate) |                                          | (collect 2 vectors)|
+-------------------+                                          +--------------------+
                                                                       |
                                                                       | pair (A, B)
                                                                       v
                                                             +----------------------+
                                                             | GPU task (run_gpu)   |
                                                             | map_reduce + reduce  |
                                                             +----------------------+
                                                                       |
                                                                       | wait timeline semaphore
                                                                       v
                                                             +----------------------+
                                                             | Read result f32      |
                                                             | MsgRingData(ptr f32) |
                                                             +----------------------+
                                                                       |
                                                                       v
                                                             +----------------------+
                                                             | Main io_uring        |
                                                             | print result         |
                                                             +----------------------+
```

### 2) `io_uring` ring interaction

```text
                 (A) data pointers                            (B) final result pointer
+-------------------------------+                         +-------------------------------+
| listener ring (in data_flow)  |                         | main ring (in main)           |
| receives ptr -> Box<[f32]>    |                         | receives ptr -> Box<f32>      |
+-------------------------------+                         +-------------------------------+
          ^                |                                              ^
          |                |                                              |
          |                +---- consumed in pairs ----+                  |
          |                                             |                  |
+----------------------+                      +----------------------+     |
| producer local rings |                      | gpu completion ring  |-----+
+----------------------+                      +----------------------+
```

---

## Why `Timeline Semaphore` is used

In `run_gpu`, compute work is submitted to the GPU queue. The CPU cannot read the result immediately after submit, because the GPU may still be writing into the output buffer.

This project uses a `Timeline Semaphore`:

1. Create semaphore with `initial_value = 0`.
2. Submit command buffer with signal `value = 1`.
3. A separate thread waits with `semaphore.wait(value = 1)`.
4. Only after the wait succeeds, read the staging buffer and send the value to `main ring`.

### Why this matters

- Guarantees correct ordering: `GPU write -> CPU read`.
- Waits for an exact submission point instead of blocking globally.
- Scales naturally to multiple submissions via monotonic timeline values.

---

## Single-batch lifecycle

```text
1) Producer creates vector A/B
2) Producer sends pointer via MsgRingData -> listener ring
3) Listener receives two vectors (A, B)
4) Listener starts run_gpu(A, B)
5) GPU command buffer executes map_reduce + final_reduce
6) GPU signals timeline semaphore value=1
7) CPU wait thread unblocks on semaphore wait(value=1)
8) CPU reads final f32 from staging buffer
9) CPU sends pointer via MsgRingData -> main ring
10) Main thread receives and prints result
```

---

## Build and run

```bash
cargo run --release
```

Requirements:

- Linux with `io_uring` support.
- Vulkan runtime and driver.
- Rust toolchain (edition 2024).

---

## Dependencies

- `io-uring` - message passing between rings (`MsgRingData`), timeouts, CQ/SQ.
- `vulkano` - Vulkan compute pipeline and synchronization.
- `naga` - runtime WGSL to SPIR-V compilation.
- `rand` - test data generation.
