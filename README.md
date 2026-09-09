## llama-manager

Run your models using optimal configurations that degrade gracefully as context expands. `llama-manager` ensures that you are always serving the most capable version of your model possible by reactively adjusting model configuration as your context window grows.

**Note:** llama-manager uses a custom fork of llama.cpp with the following changes:
* TODO elaborate

**Note:** llama-manager is in beta and may not be tested with your hardware/model. Please open an issue if you find any bugs, and include your OS, what model you were using, and any relevant logs!

**Note:** llama-manager assumes a single-GPU setup, and defaults to running models entirely on GPU. It will not work for pure-CPU inference, and probably will not work for split inference. If this is something you want, please open an issue.

### Build

Requires Node.js and npm.

```sh
git clone --recursive https://github.com/wadealexc/llama-manager.git
cd llama-manager
npm install
npm run build
```

#### llama.cpp

`llama-manager` drives `llama-server` in router mode, which requires a custom fork of llama.cpp (included as a git submodule). To build the server binary, cd into the llama.cpp submodule and build llama-server for your platform. `llama.cpp` has detailed build instructions [here](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

If you're on Linux+CUDA like me, you can use this script:

```sh
./scripts/build-server.sh --gpu
```

### Run

* **Single model**: Serve a single model using an existing `llama-server` invocation
* **Router mode**: Serve multiple models using a config.yaml file

#### Single Model

Use an existing `llama-server` invocation and run via `llama-manager` instead. Most arguments are passed directly through to `llama-server`:

```sh
node dist/index.js \
  --model "/home/models/Qwen3.8-27B-UD-Q4_K_XL.gguf" \
  --mmproj "/home/models/mmproj-BF16.gguf" \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  -ngl 999
```

Arguments are interpreted the same way `llama-server` would interpret them, with a few
changes:
- The model starts at its **best-possible configuration**: full-precision kvcache, spec decoding, mmproj on GPU. A breakpoint table showing performance degradation is printed at startup, then the model is loaded and served.
- `-c` is ignored - context is sized reactively.
- kvcache arguments are treated as a "lower bound" on kvcache precision. e.g. supplying `--cache-type-k q8_0 --cache-type-v q8_0` will allow kvcache to be quantized to a _minimum_ of q8_0 during inference. If no kvcache precision is supplied, llama-manager will quantize down to q4_0, if needed.
- Models unload after 10 minutes of idle by default, which also resets
degradation. Pass `--sleep-idle-seconds <n>` to change it (`0` disables).
  - NOTE: Currently, idle acts as a 'reset' for strategy application, so running with a timeout is recommended.

See `--help` for the full flag list. All other arguments pass through to `llama-server` untouched.

#### Router Mode

For multi-model setups, use a config file (`config.yaml` by default; override with `--config <path>` or the `MANAGER_CONFIG` env var):

```sh
node dist/index.js --config ./config.yaml --serve
```

Example config (or see this repo's config.yaml):

```yaml
host: 127.0.0.1
port: 8080
sleep-idle-seconds: 300

models:
    qwen3.8-27b:
        model: /home/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
        mmproj: /home/models/mmproj-BF16.gguf
        spec-type: draft-mtp
        n-gpu-layers: 99
        aliases: [qwen3.8-dense]
        ladder: [disable-spec, mmproj-to-cpu, quantize-kv-q8]
    qwen3.5-9b:
        model: /home/models/Qwen3.5-9B-Q4_K_M.gguf
        n-gpu-layers: 99
        ladder: []

default-models: [qwen3.8-27b]
```

Each model's entry is just llama-server arguments (passed through verbatim) plus a `ladder` — an ordered list of degradation strategies the manager may apply as that model's context grows. `default-models` are pre-loaded as weights at startup (when run with `--serve`).



## About

### Reactive context growth

When serving a model, `llama-manager` reactively applies strategies to free up device space, expanding the context window on-demand. We can see which strategies it uses by adding `--show-breakpoints` to our startup command. The result looks like this:

```sh
════════════════════════════════════════════════════════════════════════════
 qwen3.8-27b        baseline: 150,784 tokens                device: 31 GiB
════════════════════════════════════════════════════════════════════════════
  i  strategy        ctx (tokens)    gain (tokens)       weights / ctx GiB
 ──────────────────────────────────────────────────────────────────────────
  0  baseline             150,784                            17.13 / 12.01
  1  disable-spec         182,784        (+32,000)           17.13 / 12.03
  2  mmproj-to-cpu        200,960        (+18,176)           16.02 / 13.16
  3  quantize-kv-q8       262,144        (+61,184)           16.02 / 10.41
 ──────────────────────────────────────────────────────────────────────────
 final ctx:      262,144 tokens
```

For this model, `llama-manager` first serves the model at ~150k context and full kvcache precision. When a request comes in that exceeds this context window, `llama-manager` disables speculative decoding, then expands the model's context window to ~182k tokens. When the context window is exceeded again, `llama-manager` moves the mmproj to the CPU and continues serving at ~200k context.

Finally, when a request comes in that requires more than 200k tokens of context, `llama-manager` quantizes the kvcache to q8, serving the model at its maximum possible context.

`llama-manager` exposes an OpenAI-compatible interface and manages strategy applications gracefully, so your favorite harness or frontend doesn't need to do anything special to use this.

<!-- ### How it works -->

<!-- TODO -->

<!-- llama.cpp router mode, custom fork, and a sprinkle of movie magic -->

### Caveats

Assumes single GPU. Early stage project; expect breaking changes.

<!-- ### Roadmap

Strategies:
- Reactively move layers between CPU/GPU
- True mmproj-on-demand

Multi-device support -->
