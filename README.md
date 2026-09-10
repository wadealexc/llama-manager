## llama-manager

llama-manager runs your models using optimal configurations that degrade gracefully as context expands. This ensures that you are always serving the most capable version of your model possible by reactively adjusting model configuration as your context window grows.

**Notes:**
- **llama-manager is in beta** and may not be tested with your hardware/model. Please open an issue if you find any bugs, and include your OS, what model you were using, and any relevant logs!
- **llama-manager assumes inference uses a single GPU**. It will not work for pure-CPU inference, and probably will not work for split inference. If this is something you want, please open an issue.
- llama-manager uses a custom fork of llama.cpp. See (TODO) for details on what the fork introduces.

---

### Build

Requires Node.js and npm.

```sh
git clone --recursive https://github.com/wadealexc/llama-manager.git
cd llama-manager
npm install
npm run build
```

#### llama.cpp

llama-manager drives `llama-server` via a custom fork of llama.cpp. To build the server binary, cd into the llama.cpp submodule and build llama-server for your platform. llama.cpp has detailed build instructions [here](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

If you're on Linux+CUDA like me, you can use this script:

```sh
./scripts/build-server.sh --gpu
```

---

### Run

Supply args/config as either/both CLI args, or YAML.

#### Single Model

Serve a model directly from the command line using the same syntax as `llama-server`:

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

#### Multiple Models

Serve multiple models and swap between them automatically using a `config.yaml` file:

```sh
node dist/index.js --config './my-config.yaml'
```

See the [example config file](./config.example.yaml) for an example.

#### Notes

- Models are served using the **best-possible configuration**. If you pass the flag `cache-type-k/v: q8_0`, the model will initially be served at `f16` precision, and will degrade _to a minimum_ of `q8_0`. 
  - If you do not specify a precision argument, the minimum is set to `q4_0` (this will not be used unless space is needed).
- When a model is loaded for the first time, llama-manager calculates strategy breakpoints and displays them as a printed table. Use `--calc-breakpoints` to do this on startup, instead.
- Models unload after 10 minutes of idle by default, which also resets kvcache and strategies. Pass `--sleep-idle-seconds <n>` to change it (`0` disables).
  - NOTE: Currently, idle acts as a 'reset' for strategy application, so running with a timeout is recommended.
- `-c` is ignored: context is sized reactively as strategies are applied
- CLI args take priority over YAML
- Unknown YAML fields/CLI args are passed to llama-server

See `--help` for the full flag list.

---

## About

### Reactive context growth

When serving a model, llama-manager reactively applies strategies to free up device space, expanding the context window on-demand. llama-manager recognizes the following strategies:
- `disable-spec`: Disable speculative decoding
- `mmproj-to-cpu`: Move mmproj to CPU
- `quantize-kv-q8`: Quantize kvcache to q8_0
- `quantize-kv-q4`: Quantize kvcache to q4_0

By default, strategies are applied in the following order: [`disable-spec`, `mmproj-to-cpu`, `quantize-kv-q8`, `quantize-kv-q8`]. This order can be changed via the CLI flag `--ladder` (or by editing your config.yaml. See [the example](./config.example.yaml)).

Strategies are displayed when a model is loaded for the first time:

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

### llama.cpp changes

llama-manager uses [my fork of llama.cpp](https://github.com/wadealexc/llama.cpp/tree/feat/reload-runtime). This version has a few notable changes:
- New: `common_init_result::reinit_context`
  - This method factors out some common model initialization logic from `common_init_result`'s constructor, and defines a method `reinit_context` to reset a model's existing context and reinit using the factored logic.
- New: `server_context_impl::reload_model`
  - Acts as the reload analogue to `server_context_impl::load_model`. This method performs similar steps to `load_model`, except that it assumes model weights have already been loaded, and instead calls `reinit_context` rather than `common_init_from_params`.
  - Note that if `n_ctx: 0` is passed in, `reload_model` performs a fit calculation to reload to the max possible ctx (similar to `load_model` fit).
- New HTTP endpoint: `POST /reload`
  - Exposes `reload_model` as an HTTP endpoint, returning the new `n_ctx` after reloading. `POST /reload` accepts input in the form `ReloadParams` (see [the definition in types.ts](`src/client/types.ts`)).
- New HTTP endpoint: `GET /memory`
  - Query the amount of space a currently-loaded model occupies on each backend device, broken down by component. Outputs `MemoryResponse` (see [the definition in types.ts](`src/client/types.ts`)).
- Modified: `POST /slots/:id-slot` (fix prompt reuse for swa/hybrid/recurrent models)
  - `?action=save`: adds an additional 'sidecar' save file that saves prompt checkpoints
  - `?action=restore`: reads the aforementioned sidecar to restore prompt checkpoints
  - (Here, I adapted a solution from [this issue](https://github.com/ggml-org/llama.cpp/issues/25913))
- Modified: `POST /slots/:id-slot` (convert kvcache precision)
  - `?action=restore`: when restoring a slot, automatically convert between f16 / q8_0 / q4_0 precision, rather than rejecting.

The llama.cpp work is admittedly a little messy in places. I'm still working on cleaning/polishing it, as I think these features are genuinely useful and would like to contribute upstream. I'm releasing it now to get feedback from the community, as getting llama.cpp maintainer eyes on PRs has proved quite challenging so far!

### Known Issues

- Expects inference to be performed on a single GPU; does not support CPU inference. If you want support for CPU/multi-device, please open an issue.
- Hadamard rotation is disabled to simplify llama.cpp-side slot restore code.
- I've only tested this extensively on my server. YMMV; please open an issue if you find bugs or crashes. My specs:
  - Ubuntu Server
  - RTX 5090
  - CUDA v13.1
- Models I've tested:
  - Qwen3
  - Qwen3.8
  - Gemma4

### Future Work

- Smarter kv growth and model swapping (to reduce swap time and allow inference to multiple models at once)
- Better kvcache management:
  - Attach kvc to consumer via api key
  - Serve model based on input tokens / cache hit rate (each request is served at precisely the config it needs, rather than having model-wide config)
- Additional degradation strategies:
  - Move layers between GPU/CPU
  - Mmproj "on demand" (bring mmproj to GPU temporarily/as-needed to process an image, then evict it and carry result into prompt processing)
  - kvu/batching/parallelization strategies