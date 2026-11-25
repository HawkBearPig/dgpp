# Inkling 2 Mini on DGPP

This documents how DGPP, a C++/CUDA inference engine for DGX Spark, serves
[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini).
It is not a republished checkpoint. The engine loads the release as shipped,
with no conversion or re-quantization pass.

## The checkpoint, as loaded

The release is one safetensors file of 18.2 GiB, which holds the language
model and the vision tower, and a second file of 0.6 GiB with the bundled
draft model. The routed experts carry the release's two formats, named
per tensor in `hf_quant_config.json`:

- Layers 0 to 19 are NVFP4. Each weight is an E2M1 code under an E4M3 scale
  per 16 values, and each expert has one FP32 factor.
- Layers 20 to 23 are MXFP8. Each weight is an E4M3 code under a power-of-two
  scale per 32 values.

Every other tensor is BF16 or FP32: the attention projections, the norms, the
routers, the mixing queries, the loop controllers, the convolution taps, the
embedding and the output head. The engine decodes every code exactly. The
embedding and head tensors store 201,024 rows and the model reads the first
200,058.

## Architecture

Inkling 2 Mini is a looped transformer. It stores 24 layers of weights and runs
layers 6 to 17 three times, so a token passes 48 layer visits. Each visit has
its own key and value cache and its own convolution state. The hidden width is
2,304 and the position limit is 1,048,576 tokens.

**No residual stream.** The layers are grouped in blocks of six. The model
keeps the state at the end of every finished block and the partial state of
the block in progress. Each sublayer reads a learned mix of those states: a
query scores every state, a softmax weighs them, and the sublayer's norm
follows. A sublayer's output is added to the partial state of its block, and
the first layer of a block starts a fresh one.

**The loop.** Every pass over the core layers starts from the state at the
core's entry. The second and third passes also read messages from the pass
before: four queries per layer mix the previous pass's states, and a low-rank
controller per stream decides how much of the message to add and how much of
the layer's own input to keep.

**Attention.** Each layer has 24 query heads over 8 key and value heads of
width 128. There are no rotary positions. A row projects a small relative
vector per head, and a learned table turns it into a bias over the last 1,024
distances. Queries and keys are clipped in L2 norm, each row adds a sink
column to its softmax, and a per-dimension gate scales the result. Twenty
layers attend over a 1,024-token window. The last layer of every block attends
over the whole context and scales its queries with the logarithm of the
position past 8,192.

**Short convolutions.** Four streams of every visit pass through a causal
convolution over the last four positions: the keys, the values, the attention
output and the MLP output. On the full-attention layers the key and value taps
depend on the token.

**Experts.** Every layer routes each token to 8 of 128 experts by sigmoid
score with a selection bias. An expert is a soft-capped SwiGLU whose routing
weight is applied inside the activation. The final logits are soft-capped at
30.

## Expert activations

`engine.activations` selects what the experts read.

- `native` is the checkpoint's own recipe and the setting the templates ship.
  NVFP4 experts read NVFP4 inputs: each row is scaled by its maximum, and each
  16-value block takes whichever of two block scales reconstructs it with the
  smaller error. MXFP8 experts read MXFP8 inputs. The quantizers match the
  released implementation bit for bit.
- `bf16` feeds the experts BF16 inputs, which is how the reference
  implementation in transformers runs the model.

In both settings an expert's outputs are rounded to BF16 at their row scale
and a token's expert outputs are summed in a fixed order.

The native recipe is discontinuous: a difference of one unit in the last place
of an input can move a block onto the other scale. Two runs that differ only
in how many rows share a launch therefore agree closely on a trained
checkpoint but not bit for bit. Decode rows at a fixed batch shape are
deterministic.

## Serving

**Tensor parallel.** Two and four nodes split every layer the same way. A rank
holds whole key and value heads with their three query heads, the same slice
of every expert's intermediate width, and its share of the vocabulary in the
output head. The norms, routers, mixing queries, controllers and convolution
taps are replicated. A visit folds twice across ranks, once after attention
and once after the experts, so a token crosses 96 folds. Under the native
recipe each rank quantizes its own slice of the expert activation.

**Memory.** The weights take about 18 GiB on one node as the templates load
them, with the attention matrices in BF16 beside their 12-bit form. The draft
model adds 0.7 GiB and the vision tower 0.9 GiB with its workspace. The key
and value cache holds one BF16 plane per visit and one per draft layer, which
is 204 KiB per token on one node, 102 on two and 51 on four per rank. The
templates size the cache at 393,216, 786,432 and 1,572,864 tokens. Only the
four-node template seats a request at the model's position limit.

**Convolution state.** Each request keeps the last sixteen positions of every
convolution input in a ring indexed by position. A rolled-back speculative row
leaves nothing to undo, because the next row at that position overwrites its
slot. A prefix-cache snapshot carries the three positions before its cut, so
a cached prefix resumes with the same bits as a cold one.

**Prefix cache.** The convolution state exists only where a snapshot was
taken, so a request attaches at a snapshot's own position and not inside an
entry. The template writes no generation prompt, and the model opens its
turn with its own marker. The cache therefore cuts one token before every
change of speaker and at the last token of a prompt. A repeated prompt and
the next turn of a conversation attach there, and a turn that returns the
previous answer with its reasoning attaches at the end of that answer. See
[prefix reuse](../prefix-cache.md#templates-without-a-generation-prompt).

**Decode.** A decode step is one CUDA graph. The attention projections read the
BF16 matrices in their lossless 12-bit form, the attention splits the cached
keys across blocks and merges them, and the experts read each selected
expert's weights once. On one node the step is bound by memory bandwidth. A
step also verifies the draft model's proposals and drafts the next ones, as
the next section describes.

**Prefill.** Prompts are processed in chunks of up to 4,096 tokens. Attention
runs on the tensor cores with the relative bias, the sink and the gate inside
the kernel. Under the native recipe the NVFP4 experts run on the block-scaled
tensor-core instruction with both operands in four bits.

## Text interface

The tokenizer is the release's byte-level BPE. The prompt is rendered by the
checkpoint's own `chat_template.jinja`.

- **Reasoning effort.** The template writes a thinking effort level into the
  prompt. `reasoning_effort` takes `none`, `minimal`, `low`, `medium`, `high`,
  `xhigh` and `max`, and an omitted value is `high`. `none` is the lowest
  level. It shortens the reasoning and does not switch it off.
- **Replies.** The model writes typed blocks. A thinking block becomes
  `reasoning_content`, a text block becomes `content`, and a tool block
  carries one call as a JSON object with the tool's name and arguments.
- **Tool calls.** Tools are declared in the prompt as the template writes
  them. `tool_choice` and `response_format` are enforced by constrained
  decoding over the same blocks. A call's arguments follow the tool's schema,
  a required call cannot be put off with an answer in text, and a JSON
  response is one text block that holds the value.

## The draft model

The release bundles a three-layer draft model for speculative decoding, and
the templates run it (`engine.mtp`). After every step it proposes the next
seven tokens in one pass. The next step verifies some of them in the same
pass as the token the model just chose, and commits every proposal that is
the model's own choice. The output is therefore the model's. The draft only
changes how many tokens a step yields.

**What it reads.** The draft keeps four states of every committed token: the
block state at the entry of visits 5, 23 and 41, and the final output. Each
draft layer mixes them with its own weights into keys and values, which live
in one cache plane per draft layer beside the model's 48. A prefix-cache
snapshot and a rolled-back row cover them like the model's own planes.

**A block.** The draft runs eight rows at the next eight positions: the
newest token, then seven copies of a learned mask row. Every row attends over
the block and the context, and two of the three layers only over the last
1,024 positions. Each mask row predicts the token at its own position. A
recurrent head then corrects the predictions one after another: each step
reads the proposal before it and biases the logits of the whole vocabulary.
The arithmetic follows the released kernels, with FP32 inside and BF16 where
they round.

**How many proposals a step verifies.** Every verify row reads its own
experts, so a row costs real time: about 3.9 ms on one node, 2.2 on two and
1.4 on four. The engine verifies a proposal only while it is worth its row
(`engine.mtp_schedule`), and none at all when the first is not. The draft has
no confidence output, so the engine estimates each proposal's chance of
acceptance from the probability the draft gave it. That probability
overstates acceptance, and a two-parameter fit measured on the release
corrects it. Code is accepted far more often than prose, and the schedule
follows: it verifies about four proposals a step on code and about one on
prose. The recurrent head uses the same estimate to stop early: once no later
proposal could be verified, it skips its remaining steps.

**Several requests in one pass.** With the draft on, the verify rows of
several requests share a pass. The templates keep the attention matrices in
BF16 beside their 12-bit form (`engine.bf16_weights` set to `bf12+bf16`), so
a pass of more than eight rows reads each matrix once. The vocabulary head
is read once in its 12-bit form for up to 32 rows.

**The shared head screen.** `engine.draft_head` set to `screen` has no effect
on this model. The recurrent head moves tokens that a shortlist taken from
the uncorrected logits leaves out, so the draft reads the exact head.

## Loop guidance

`engine.loop_guidance` is an optional decoding mode for this model, off in the
templates that carry no `_guided` suffix. It follows the contrastive decoding
of looped transformers described in arXiv 2610.02185.

**What it does.** Each pass over the core layers leaves the states the last
six layers read. The model's answer comes from the third pass. Under guidance
the last six layers run a second time on the first pass's states, which gives
a weaker reading of the same position, and the request's logits move away
from that reading: `(1 + s) * logits - s * readout` at strength `s`, before
penalties, bias, grammar and temperature. A guided output is not the
checkpoint's own. At strength 0 it is.

**What a request controls.** `loop_guidance_strength` on a chat request sets
`s` for that request, from 0 to 2. An omitted field takes the deployment's
`engine.loop_guidance_strength`. Requests at different strengths share a pass.

**What stays as it is.** The second run reads and writes K/V planes and
convolution state of its own, six of each, after the draft model's. The
model's own keys, values and convolution state and the draft model's inputs
are bit for bit what they are without guidance. A prefix-cache entry
therefore serves every strength, and the draft model's proposals are checked
against the guided choice, so the output with the draft model on is the
output without it.

**What it costs.** The second run happens for every prompt and generated
token of a guided deployment, whatever a request's strength: six more layer
visits on top of 48, about 12 percent more K/V memory per token, and one more
read of the vocabulary head per row. See [benchmarks](../benchmarks.md) for
the measured rates of the guided template.

**Fewer passes.** Running the core fewer than three times is not offered. The
checkpoint is trained at exactly three passes and stops completing its
reasoning at two.

## Images

The vision tower is served. An image part of a chat request is kept at its
own size, without resizing, and costs one token per 40 x 40 pixel cell: a
640 x 480 image is 192 tokens. The image is padded on the right and bottom to
whole cells, and the tower sees the padding as the processor's padding value.
Transparency is composited on the processor's gray checkerboard. An image
that would need more than 1,024 tokens is scaled down to the largest size
that fits, keeping its aspect.

The tower is a 23-layer vision transformer of width 1,024 over 20-pixel
patches, with a learned position grid resampled to the image and a rotary
embedding along rows and columns. The four patches of a cell are projected
together into one row of the language model's width, and that row takes the
place of the placeholder token's embedding. The tower runs once per image on
every rank during prefill, and an image seen before is taken from a small
cache keyed on its pixels.

The chat template writes the image block itself, so the request's image parts
go to the template as they are and the engine expands each placeholder to the
image's tokens. Images take part in the prefix cache: a repeated image prompt
and the next turn of an image conversation attach like text.

## How correctness is established

A NumPy reference in `tools/inkling2_reference_dump.py` restates the forward
pass from the checkpoint's own tensors. The engine is compared with it visit
by visit, each visit starting from the engine's own state, on a small
synthetic checkpoint and on the release. The comparison covers both activation
settings and worlds of one, two and four ranks. The activation quantizers are
also compared bit for bit with the released implementation's kernels, and the
transformers implementation is the end-to-end reference for the `bf16`
setting.

The draft model has its own restatement in the same tool. The context keys
and values the engine stores, the block's rows and all proposals are compared
with it at one, two and four ranks on the synthetic checkpoint, and on the
release, where every proposal matches.

The vision tower is compared stage by stage with a NumPy reference in
`tools/inkling2_vision_reference.py`, and on the release with the image rows
transformers computes. The engine's rows sit 2 to 3 percent from
transformers' FP32 tower, which is where transformers' own BF16 tower sits.
See [testing](../testing.md).

## Measured

See [benchmarks](../benchmarks.md) for the current numbers.

## License and attribution

The model, its weights, tokenizer and chat template are the upstream
release's. See the upstream model card for its terms.
