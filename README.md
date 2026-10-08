## llama-manager

llama-manager runs your models using optimal configurations that evolve with context length.

Models are initially served at the highest speed and precision available. As context expands, llama-manager applies strategies that scale model performance down to make room on your machine. This ensures you are always serving the most capable version of your model.

llama-manager is built using a custom fork of llama.cpp. See ([llama.cpp changes](#llamacpp-changes)) for details on what the fork introduces.

#### Features

llama-manager exposes the features below over via a standard OpenAI chat completions interface. You can use it with your favorite harness or frontend.

- **Always serve max context**: model configuration and context window are updated during inference, favoring performance at low context, and capacity at high context. llama-manager ensures you can always run your models up to their max supported context without impacting quality until it's absolutely necessary.
- **Auto-swap models:** models are loaded/unloaded/swapped on demand, allowing you to serve multiple models from the same server.
- **Session storage:** slots are cached between model swaps and reloads, keeping recent sessions ready to serve at any time.
- **Configurable strategies:** specify what strategies you prefer and how they should be applied (see [Customizing](#customizing)). 

Strategies are applied automatically as needed, sacrificing either _speed_ or _quality_ in exchange for a larger context window. All strategies maintain a persistent kvcache, meaning existing prefill does not need to be repeated:
- **Speed:** 
  - toggle speculative decoding on/off to increase context
  - move the mmproj off the GPU in exchange for 
- **Quality:**
  - quantize your model's kvcache to q8 or q4
  - quantize your model's weights

---

### Build & Run

#### Build llama-manager

Requires Node.js and npm.

```sh
git clone --recursive https://github.com/wadealexc/llama-manager.git
cd llama-manager
npm install
npm run build
```

#### Build llama.cpp

llama-manager drives `llama-server` via a custom fork of llama.cpp. To build the server binary, cd into the llama.cpp submodule and build llama-server for your platform. llama.cpp has detailed build instructions [here](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

If you're on Linux+CUDA like me, you can use this script:

```sh
./scripts/build-server.sh --gpu
```

#### Quick Start

After building, serve a model directly from the command line using the same syntax as `llama-server`. llama-manager will apply default performance strategies and serve your model at the best possible speed and precision.

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

Note that this requires models on your machine already; llama-manager does not resolve huggingface or other remote options.

#### Customizing

See [config.example.yaml](./config.example.yaml) for an example.

Run with a config file to:
- Serve multiple models (llama-manager will swap between them as needed)
- Specify which strategies should/should not be applied
- Customize strategy order
- Use strategies like `swap-model` to quantize model weights on demand
- Configure other server parameters (endpoint, idle timeout, disk cache usage, ...)

```sh
node dist/index.js --config './my-config.yaml'
```

---

## About

### Strategies

llama-manager reactively applies strategies to free up device space, expanding the context window on-demand. By default, strategies are applied in the following order:
- `disable-spec`: Disable speculative decoding
- `mmproj-to-cpu`: Move mmproj to CPU
- `quantize-kv-q8`: Quantize kvcache to q8_0
- `quantize-kv-q4`: Quantize kvcache to q4_0

Strategies are applied only at measured context thresholds, so including `quantize-kv-q4` will not actually quantize your kvcache until the corresponding threshold is reached. Additionally, strategies are only applied if they actually increase your context capacity.

These thresholds are printed out when a model is loaded for the first time. For example:

```sh
════════════════════════════════════════════════════════════════════════════
 qwen3.8-27b        baseline: 167,680 tokens                device: 31 GiB
════════════════════════════════════════════════════════════════════════════
  i  strategy        ctx (tokens)    gain (tokens)       weights / ctx GiB
 ──────────────────────────────────────────────────────────────────────────
  0  baseline             167,680                            17.13 / 13.14
  1  disable-spec         200,960        (+33,280)           17.13 / 13.16
  2  mmproj-to-cpu        218,880        (+17,920)           16.02 / 14.27
  3  quantize-kv-q8       262,144        (+43,264)           16.02 / 10.40
 ──────────────────────────────────────────────────────────────────────────
 final ctx:      262,144 tokens
```

llama-manager also supports swapping between model quantizations mid-generation. By including a `model-variants` field, you can define quantizations for use with the `swap-model` strategy. 

For example, the following config uses Qwen3.8-27B `UD-Q6_K_XL` at low context, then swaps to `UD-Q4_K_XL` at medium context:

```yaml
# (from config.example.yaml)
models:
    qwen-dynamic-model:
        # main model, loaded first (Q6_K_XL)
        model: /home/models/qwen3.8-27b/Qwen3.8-27B-UD-Q6_K_XL.gguf
        mmproj: /home/models/qwen3.8-27b/mmproj-BF16.gguf

        # lower-precision quant(s) to swap in as context grows
        model-variants:
            q4: /home/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf

        spec-type: draft-mtp
        spec-draft-n-max: 2

        fit-target: 512
        n-gpu-layers: 99

        # model variants can be referenced in ladder via swap-model:${VARIANT_NAME}
        # I chose to degrade the Q6 model a little before swapping in the smaller model
        ladder:
            - quantize-kv-q8
            - mmproj-to-cpu
            - swap-model:q4  # swap to Q4_K_XL
            - disable-spec
            - quantize-kv-q8 # (swapping model quant resets kv precision to f16)
```

### llama.cpp changes

llama-manager uses [my fork of llama.cpp](https://github.com/wadealexc/llama.cpp/tree/feat/reload-runtime). This version has a few notable changes:
- New: `common_init_result::reinit_context`
  - This method factors out some common model initialization logic from `common_init_result`'s constructor, and defines a method `reinit_context` to reset a model's existing context and reinit using the factored logic.
- New: `fit.cpp::common_fit_for_reload`
  - Performs fit calculations while assuming the model's weights are already in place. (Needs work)
- New: `server_context_impl::reload_model`
  - Acts as the reload analogue to `server_context_impl::load_model`. This method performs similar steps to `load_model`, except that it assumes model weights have already been loaded, and instead calls `reinit_context` rather than `common_init_from_params`.
  - Note that if `n_ctx: 0` is passed in, `reload_model` performs a fit calculation to reload to the max possible ctx (similar to `load_model` fit).
- New HTTP endpoint: `POST /reload`
  - Exposes `reload_model` as an HTTP endpoint, returning the new `n_ctx` after reloading. `POST /reload` accepts input in the form `ReloadParams` (see [the type definition in types.ts](`src/client/types.ts`)).
- New HTTP endpoint: `GET /memory`
  - Query the amount of space a currently-loaded model occupies on each backend device, broken down by component. Outputs `MemoryResponse` (see [the type definition in types.ts](`src/client/types.ts`)).
- Modified: `POST /slots/:id-slot` (fix prompt reuse for swa/hybrid/recurrent models)
  - `?action=save`: adds an additional 'sidecar' save file that saves prompt checkpoints
  - `?action=restore`: reads the aforementioned sidecar to restore prompt checkpoints
  - (Here, I adapted a solution from [this issue](https://github.com/ggml-org/llama.cpp/issues/25913))
- Modified: `POST /v1/chat/completions/input_tokens` and slot-save responses
  - Always include rendered token IDs and media chunk identities/spans for manager-side prefix matching.
- Modified: `POST /slots/:id-slot` (convert kvcache precision)
  - `?action=restore`: when restoring a slot, automatically convert between f16 / q8_0 / q4_0 precision, rather than rejecting. (Needs work)

The llama.cpp work is admittedly a little messy in places. I'm still working on cleaning/polishing it, as I think these features are genuinely useful and would like to contribute upstream. I'm releasing it now to get feedback from the community, as getting llama.cpp maintainer eyes on PRs has proved quite challenging so far!

### Limitations and Known Issues

I built llama-manager for my personal workflows and hardware. I did not go out of my way to support 'every possible workflow' because I only have so much time. I'm continuing to prioritize my needs - but if you want to use llama-manager and find it doesn't work for you, please open an issue!

Here are the major limitations I'm aware of:

- *Single-GPU only:* llama-manager does not support running on the CPU (or running across multiple GPUs).
- *Only one active model:* while llama-manager will swap between models as needed, it is currently geared towards a "one big model" workflow. It does not support parallel generation from multiple models (even if they fit on your GPU at the same time). Requests to different models are queued and served roughly in order of request.
    - (Note that parallel generation from the _same_ model is supported!)
- *Session storage:* is somewhat brittle and is not highly optimized. The current version does a decent job of keeping repeated-prefill minimal even with multiple users/workflows, but there's lots of room for improvement.
- *Sluggish kvcache save/restore at very high context:* "Sluggish" is relative. On my machine, applying a strategy at 200k+ context can take upwards of 20-30 seconds (while strategies < 150k are hardly noticable). This is still much faster than having the model repeat prefill (~80 seconds at 200k), but there is lots of room for optimization.

### LLM Usage Disclosure

I use a mixture of human coding and LLM coding + human review. In particular, there are a few parts of the codebase that are more LLM-heavy:
- Stream/SSE handling in the API routes
- Config parsing/building
- Tests/scripts