# V-JEPA 2: How Meta's Video World Model Actually Works

**A ground-up explanation for developers who are new to AI/ML**

> *"A major challenge for modern AI is to learn to understand the world and learn to act largely by observation."*
> — Opening line of the V-JEPA 2 paper (Meta FAIR, June 2025)

V-JEPA 2 is a video model that learns to understand the physical world by watching over a million hours of internet video — and then uses that understanding to control a robot arm in environments it has never seen before. No labels. No rewards. No task-specific training.

That sentence probably sounds like magic if you don't have a machine learning background. This tutorial will take you from zero to understanding exactly how it works, building every concept from scratch.

---

## Part 1: Prerequisites — The Building Blocks

### What Is a Neural Network?

Forget the brain metaphors for a moment. A neural network is a **mathematical function that you can tune**.

Imagine you have a function like this:

```
output = f(input)
```

For a regular function, the behavior is fixed. `sin(x)` always returns the sine of x. But a neural network is more like:

```
output = f(input, knobs)
```

Where `knobs` are millions (or billions) of adjustable numbers. By turning these knobs, you can make `f` do almost anything — recognize cats, translate languages, predict what happens next in a video.

Each "knob" is called a **parameter** or **weight**. A neural network is structured as layers, where each layer takes the output of the previous layer, multiplies it by a matrix of weights, adds a bias, and then applies a simple non-linear function (like "if the number is negative, make it zero"). Stack enough of these layers, and the network can approximate incredibly complex functions.

```
                    ┌─────────┐
   Input  ────────▶ │ Layer 1 │ ──▶ (some numbers)
                    └─────────┘
                         │
                    ┌─────────┐
                    │ Layer 2 │ ──▶ (different numbers)
                    └─────────┘
                         │
                        ...
                         │
                    ┌─────────┐
                    │ Layer N │ ──▶ Output
                    └─────────┘
```

The key insight: **the network doesn't "know" anything innately**. All its intelligence comes from the specific values of its weights, which are learned during training.

### Tensors, Vectors, and Embeddings

Neural networks don't work with images, words, or videos directly. They work with **numbers arranged in grids**.

- A **scalar** is a single number: `42`
- A **vector** is a 1D list of numbers: `[0.3, -1.2, 0.8, 0.1]`
- A **matrix** is a 2D grid of numbers (like a spreadsheet)
- A **tensor** is the general term for any of these — an n-dimensional grid of numbers

An image with 256×256 pixels and 3 color channels (RGB) is a tensor with shape `[3, 256, 256]` — that's 196,608 numbers. A video is a tensor with an extra time dimension: `[frames, 3, height, width]`.

Now here's the really important concept: an **embedding**. Imagine you want to represent the *meaning* of a video clip as a list of numbers. Not the raw pixels, but something more abstract — whether there's a hand reaching for a cup, whether an object is falling, whether something is being pushed left.

An embedding is a **learned vector that captures the essence of something**. The neural network learns to produce these during training. For V-JEPA 2, a single video patch might be represented as a vector of 1024 numbers — its embedding. These 1024 numbers encode everything the model thinks is important about that patch in context.

Think of it like this: a photo of a cat is millions of pixel values. But the *meaning* of "there's a cat here" can be captured in a much smaller embedding vector. Similar cats will have similar embeddings. Dogs will have different embeddings. The network learns which features matter.

### How Models Learn: Training, Loss, and Gradient Descent

Training is the process of finding good values for all those millions of knobs. Here's the loop:

1. **Forward pass**: Feed some data through the network, get a prediction
2. **Compute loss**: Compare the prediction to what you wanted — how wrong was it?
3. **Backward pass**: Figure out which knobs to turn and in which direction to reduce the error
4. **Update**: Turn the knobs slightly

The **loss function** is just a number that says "how bad is this prediction?" Lower is better. If the model predicted a cat but it was a dog, the loss is high. If it predicted cat and it was a cat, the loss is low.

**Gradient descent** is the algorithm for step 3. "Gradient" tells you the slope — which direction makes the loss go up or down for each knob. You nudge each knob a tiny bit in the direction that reduces the loss. Repeat this millions of times, and the knobs converge to values that make the network good at its task.

```
  Loss
   ▲
   │  ╲
   │    ╲         ← gradient tells us: "go this way"
   │      ╲
   │        ╲  ╱
   │          ╲╱   ← we want to find this valley (minimum)
   │
   └──────────────▶  Weight value
```

**Learning rate** is how big each nudge is. Too big and you overshoot the valley. Too small and training takes forever.

### Supervised vs. Self-Supervised Learning

In **supervised learning**, you need labeled data. Someone has manually annotated each example: "this video shows a person cooking," "this image contains a dog." The model learns to match inputs to labels. This works great but has a fatal flaw: **labeling is expensive and doesn't scale**. You can't manually label a million hours of video.

In **self-supervised learning**, the model creates its own training signal from the data itself. The classic example: take a sentence, hide a word, and make the model predict the hidden word. No human labels needed — the structure of the data IS the supervision.

For V-JEPA 2, the self-supervised task is: **take a video, hide large chunks of it, and make the model predict what's in the hidden parts** — but not at the pixel level (we'll get to why shortly). This means you can train on essentially unlimited video data from the internet without a single human annotator.

This is why self-supervised learning is such a big deal. It lets you learn from the internet's entire video library.

---

## Part 2: The Transformer — The Architecture Behind Everything

### The Attention Mechanism

The Transformer is an architecture (a specific way of organizing a neural network) that revolutionized AI starting in 2017. Its key innovation is **attention**.

Consider this sentence: "The cat sat on the mat because **it** was tired." What does "it" refer to? The cat, not the mat. Your brain figured that out by *attending* to the relevant earlier word.

Attention works similarly in a neural network. For each element in a sequence, the network computes:
- **Query**: "What am I looking for?"
- **Key**: "What do I contain?"
- **Value**: "What information do I actually have?"

Every element sends out a query and compares it against every other element's key. High similarity = "pay attention to this." The network then takes a weighted combination of the values, where the weights are based on those query-key similarities.

```
  Token A          Token B          Token C
    │                 │                │
  ┌─┴─┐           ┌──┴─┐          ┌──┴─┐
  │ Q │           │ Q  │          │ Q  │
  │ K │           │ K  │          │ K  │
  │ V │           │ V  │          │ V  │
  └───┘           └────┘          └────┘
    │                 │                │
    └────── compare all Q×K pairs ─────┘
                     │
              weighted sum of V's
```

The magic: **attention is learned**. The network discovers through training which things are relevant to each other. In a video, it might learn that a moving hand is highly relevant to the cup it's approaching, but irrelevant to the bookshelf in the background.

**Multi-head attention** means you run several independent attention computations in parallel. One head might focus on spatial relationships, another on motion, another on color patterns. The results are combined.

### Vision Transformer (ViT) — How Images Become Tokens

Transformers were originally designed for text sequences. To use them for images, we need to convert a 2D image into a sequence of tokens. The Vision Transformer (ViT) does this in a delightfully simple way:

1. **Cut the image into patches** (e.g., 16×16 pixel squares)
2. **Flatten each patch** into a vector
3. **Project each vector** through a linear layer to get an embedding
4. Feed this sequence of patch embeddings into a standard Transformer

```
  Original Image (256×256)
  ┌──┬──┬──┬──┬──┬──┬──┬──┐
  │  │  │  │  │  │  │  │  │
  ├──┼──┼──┼──┼──┼──┼──┼──┤    256 / 16 = 16 patches per side
  │  │  │  │  │  │  │  │  │    16 × 16 = 256 patches total
  ├──┼──┼──┼──┼──┼──┼──┼──┤
  │  │  │  │  │  │  │  │  │    Each patch → embedding vector
  ├──┼──┼──┼──┼──┼──┼──┼──┤
  ...                           Result: sequence of 256 tokens
  └──┴──┴──┴──┴──┴──┴──┴──┘
```

Now the Transformer can work its attention magic across patches. A patch showing an eye can attend to a patch showing a nose to understand "this is a face."

### From Images to Video: Tubelets and the Temporal Dimension

Video adds time. V-JEPA 2 extends ViT to video by cutting the video into **tubelets** — 3D patches that span 2 frames × 16 pixels × 16 pixels.

Think of it like cutting a loaf of bread into cubes instead of slices. Each tubelet captures a tiny chunk of space AND time.

```
  Video: 64 frames at 256×256

  Time ──────▶
  ┌──┬──┬──┬──┐
  │  │  │  │  │  Each tubelet = 2 frames × 16px × 16px
  ├──┼──┼──┼──┤
  │  │  │  │  │  Spatial: (256/16)² = 256 patches per frame-pair
  ├──┼──┼──┼──┤  Temporal: 64/2 = 32 time steps
  │  │  │  │  │
  └──┴──┴──┴──┘  Total: 256 × 32 = 8,192 tokens!

  (Simplified — actual grid is 16×16×32)
```

With 64 frames of 256×256 video, V-JEPA 2 produces **8,192 tokens**, each represented by a 1024-dimensional embedding (for the ViT-L model). That's a long sequence, which is one reason these models need so much compute.

### Positional Embeddings: How the Model Knows Where Things Are

Once you cut an image into patches and feed them as a sequence, the Transformer has no idea which patch came from where. The top-left patch and the bottom-right patch are both just vectors in a sequence. We need to inject position information.

**Positional embeddings** are vectors added to (or combined with) each token to encode its position. The original ViT used fixed, pre-computed embeddings. V-JEPA 2 uses something more sophisticated called **3D Rotary Position Embeddings (3D-RoPE)**, which we'll cover in Part 3.

---

## Part 3: V-JEPA 2 — The Actual Model

### The JEPA Philosophy: Predict Representations, Not Pixels

This is the single most important idea in V-JEPA 2, so let's spend some time on it.

Most video models (like Sora, Cosmos, etc.) try to **predict future pixels**. Given some frames of a video, generate the next frames. This seems intuitive — but it has deep problems.

**The leaf problem**: Imagine a video of a tree swaying in the wind. To predict the next frame at the pixel level, the model needs to figure out exactly where every single leaf will be. That's chaotic and essentially unpredictable. But what's *actually* important? The tree is still there, it's still swaying, nothing meaningful changed.

Pixel prediction forces the model to waste capacity on irrelevant, unpredictable details (exact leaf positions, precise water ripple patterns, subtle lighting changes). It's like asking someone to predict the exact position of every grain of sand on a beach instead of "the beach is still there."

**JEPA (Joint-Embedding Predictive Architecture)** takes a radically different approach:

> **Don't predict pixels. Predict the *abstract representation* of what comes next.**

```
  ┌─────────────────────────────────────────────────────┐
  │  PIXEL PREDICTION (e.g., Sora)                      │
  │                                                     │
  │  Video frames ──▶ Model ──▶ Future pixel values     │
  │                              (must get every        │
  │                               detail right)         │
  └─────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │  JEPA (V-JEPA 2)                                    │
  │                                                     │
  │  Video frames ──▶ Encoder ──▶ Abstract embedding    │
  │                                (captures meaning,   │
  │                                 ignores noise)      │
  └─────────────────────────────────────────────────────┘
```

The model learns a representation space where similar situations map to nearby points. The exact pixel arrangement of leaves doesn't matter — what matters is "tree swaying in wind," and that concept has a stable representation even as the pixels change chaotically.

This is why V-JEPA 2 is better at understanding physics: it's forced to learn what's *predictable* about the world (object trajectories, gravity, cause-and-effect) and ignore what's not (noise, texture details, lighting fluctuations).

### The Training Architecture: Three Components

V-JEPA 2's pre-training involves three interacting components:

```
                          ┌─────────────────┐
  Video ──(mask)──────▶   │    ENCODER       │ ──▶ Visible patch
  (visible parts only)    │    (ViT, ~1B     │     embeddings
                          │     params)      │
                          └─────────────────┘
                                  │
                                  ▼
                          ┌─────────────────┐
  Mask tokens ──────────▶ │   PREDICTOR      │ ──▶ Predicted
  (position info for      │   (small ViT,    │     embeddings for
   hidden patches)        │    ~22M params)  │     hidden patches
                          └─────────────────┘
                                  │
                          Compare (L1 loss)
                                  │
                          ┌─────────────────┐
  Video ──────────────▶   │  TARGET ENCODER  │ ──▶ Target
  (full, unmasked)        │  (EMA copy of    │     embeddings for
                          │   encoder)       │     hidden patches
                          └─────────────────┘
```

Let's walk through each piece.

### The Encoder: Video → Abstract Features

The encoder is the main model — a large Vision Transformer (up to 1 billion parameters in the ViT-g variant). Its job:

1. Take a video clip (e.g., 64 frames at 256×256 or 384×384 resolution)
2. Patchify it into tubelets (2 frames × 16px × 16px each)
3. **Mask out** a large portion of the tubelets (randomly drop ~75% of them)
4. Process only the **visible** tubelets through the Transformer
5. Output an embedding vector for each visible tubelet

The masking is crucial — this is the self-supervised training signal. The model only sees fragments of the video and must figure out what's happening in the hidden parts.

V-JEPA 2 uses a **multi-block masking strategy**: rather than randomly removing individual patches, it masks out large contiguous blocks. This makes the prediction task harder and more meaningful — the model can't just interpolate from nearby visible patches.

### The Predictor: Learning Through Masked Prediction

The predictor is a smaller, lighter-weight ViT (~22M parameters) that's **only used during training** (it's discarded at inference time).

Here's what it does:

1. Takes the encoder's output (embeddings for visible patches)
2. Takes **learnable mask tokens** — placeholder tokens that carry position information for the masked patches
3. Concatenates them and processes everything through its Transformer layers
4. Outputs predicted embeddings for the positions that were masked

The predictor's job is essentially: "Given what you can see, what do the hidden parts look like *in representation space*?"

### EMA Target Encoder: The Stable Teaching Signal

Here's a subtle problem: if the encoder generates both the input AND the target for prediction, the network could find a shortcut — it could learn to "collapse" all representations to the same value. If every patch maps to the same embedding, prediction becomes trivially easy (just predict that same embedding), but the representations are useless.

V-JEPA 2 prevents this with the **EMA (Exponential Moving Average) target encoder**:

- The target encoder is a **copy** of the main encoder
- Its weights are NOT trained by gradient descent
- Instead, after each training step, the target encoder's weights are updated as a slow-moving average of the main encoder's weights:

```
target_weights = 0.999 × target_weights + 0.001 × encoder_weights
```

This means the target encoder evolves slowly and smoothly, providing a **stable prediction target**. The main encoder can learn quickly, but it's chasing a target that moves gently. This prevents collapse — if the main encoder tried to produce collapsed representations, the target encoder (which still has good representations from recent history) would provide targets that push it back.

Think of it like a dance partner who's always half a beat behind you — similar enough to provide useful guidance, but different enough that you can't cheat by standing still.

### 3D-RoPE: Rotary Position Embeddings in 3D

Remember the positional embedding problem? The model needs to know where each token is in space and time. V-JEPA 2 uses **3D Rotary Position Embeddings (3D-RoPE)**, and it was a critical factor in making training stable at the billion-parameter scale.

Traditional positional embeddings are **added** to the token embeddings — they're fixed vectors that encode position. RoPE takes a different approach: it **rotates** the query and key vectors in the attention mechanism based on position.

Here's the intuition: imagine each embedding dimension as a clock hand. For each spatial/temporal position, the hands are rotated by a specific angle. When two tokens compute their attention score (query × key), the rotation naturally encodes the *relative* distance between them. Nearby tokens in space/time will have small angular differences; distant ones will have large differences.

For video, we need position encoding in 3 dimensions: **time** (which frame), **height** (which row), and **width** (which column). V-JEPA 2 splits the embedding dimensions into three roughly equal groups and applies 1D rotations to each group for its respective axis:

```
Embedding dimensions: [──── time ────│── height ──│── width ──]
                       rotated by     rotated by   rotated by
                       temporal       vertical     horizontal
                       position       position     position
```

Why is this better than fixed positional embeddings?

1. **Relative positions**: RoPE naturally encodes relative distance, which is more useful than absolute position
2. **Generalization**: The model can handle different video lengths and resolutions at inference time, even if it wasn't trained on them
3. **Stability**: This turned out to be crucial for training the largest models without instabilities

### The Training Objective: L1 Loss in Representation Space

The loss function is straightforward: for each masked patch, compute the **L1 distance** (sum of absolute differences) between the predictor's output and the target encoder's output.

```
Loss = |predicted_embedding - target_embedding|

(averaged across all masked patches in the batch)
```

L1 loss (also called Mean Absolute Error) was chosen over the more common L2 loss (Mean Squared Error). L1 is more robust to outliers — it doesn't excessively penalize occasional large errors, leading to more stable training.

The loss is only computed for the **masked** patches. The visible patches already have known representations — the whole point is learning to predict the hidden ones.

### The Complete Pre-Training Picture

Let's trace through one training step:

1. **Sample** a video clip from the 22M-video dataset (VideoMix22M)
2. **Patchify** it into ~8,192 tubelets
3. **Mask** out large contiguous blocks (~75% of tubelets)
4. **Encode** the visible ~25% through the main encoder → visible embeddings
5. **Predict** the hidden patches: predictor(visible embeddings + mask tokens) → predicted embeddings
6. **Generate targets**: pass the FULL unmasked video through the target encoder → target embeddings
7. **Compute L1 loss** between predicted and target embeddings (only at masked positions)
8. **Backpropagate** through predictor and encoder (NOT through target encoder)
9. **Update** encoder and predictor weights via gradient descent
10. **EMA update** the target encoder's weights

Repeat this ~252,000 times across 1 million+ hours of video. The result: an encoder that deeply understands video content — motion, physics, object interactions, spatial relationships.

---

## Part 4: From Understanding to Robotics (V-JEPA 2-AC)

### Stage 1: Learning Physics from Internet Video

Everything described in Part 3 is Stage 1. After pre-training on 1M+ hours of internet video, the V-JEPA 2 encoder has learned:

- How objects move through space (trajectories, gravity, momentum)
- Object permanence (things still exist when occluded)
- Spatial relationships (above, below, inside, next to)
- Temporal dynamics (actions have consequences, motion has patterns)
- Appearance understanding (what things look like from different angles)

The evidence: V-JEPA 2 achieves 77.3% top-1 accuracy on Something-Something v2 (a dataset specifically designed to test motion understanding — tasks like "pushing something from left to right") and 39.7% recall-at-5 on Epic-Kitchens-100 for action anticipation (predicting what a person will do next in a kitchen). That's a **44% relative improvement** over the previous best model on action anticipation.

**After Stage 1, the encoder weights are frozen.** All this knowledge is locked in and won't be modified.

### Stage 2: Learning Cause-and-Effect from Robot Data (V-JEPA 2-AC)

Stage 1 gives the model passive understanding — it watches but never acts. Stage 2 adds the ability to understand **"if I do this action, then this state change happens."**

This is V-JEPA 2-AC (Action-Conditioned). Here's what changes:

- The **encoder stays frozen** — its internet-learned knowledge is preserved
- A **new action-conditioned predictor** is trained on top of the frozen encoder
- Training data: **less than 62 hours** of robot manipulation videos from the DROID dataset (a public dataset of Franka robot arms performing various tasks)
- The robot videos include **action commands** (joint velocities, end-effector positions) paired with the video frames

The action-conditioned predictor is a 300M-parameter Transformer with **block-causal attention** — meaning each timestep can only attend to current and past timesteps (no peeking at the future). It predicts:

> "Given the current video representation + this action → what will the next video representation look like?"

```
  ┌──────────────────────────────────────────────────┐
  │          ACTION-CONDITIONED PREDICTOR             │
  │                                                   │
  │  Input at each timestep:                          │
  │    • Frozen encoder features (from current frame) │
  │    • Action taken (robot joint commands)           │
  │    • End-effector state (gripper position, etc.)   │
  │                                                   │
  │  Output:                                          │
  │    • Predicted encoder features for next frame     │
  │                                                   │
  │  Attention pattern (block-causal):                │
  │    t=1  t=2  t=3  t=4                             │
  │    ✓    ✗    ✗    ✗     ← t=1 sees only itself    │
  │    ✓    ✓    ✗    ✗     ← t=2 sees t=1 and t=2   │
  │    ✓    ✓    ✓    ✗     ← t=3 sees t=1,2,3       │
  │    ✓    ✓    ✓    ✓     ← t=4 sees everything     │
  └──────────────────────────────────────────────────┘
```

The brilliant part: because the encoder already understands physics from watching millions of hours of video, the action-conditioned predictor only needs to learn the **mapping between actions and their effects** — a much simpler task. That's why 62 hours of robot data is enough.

### Model-Predictive Control: How the Robot Actually Plans

V-JEPA 2-AC doesn't learn a fixed "policy" (a direct mapping from observation to action). Instead, it uses its world model for **Model-Predictive Control (MPC)** — actively planning by simulating different possible futures.

Here's how the robot completes a task like "pick up the cup":

**Step 1: Goal Specification**
The robot is given a **goal image** — a picture of what success looks like (e.g., the cup in the robot's gripper). This goal image is encoded through the frozen encoder to produce a goal embedding.

**Step 2: Imagine Possible Futures**
The robot samples many random action sequences. For each sequence, it uses V-JEPA 2-AC to predict what would happen:

```
Current state → [action 1] → predicted state 1 → [action 2] → predicted state 2 → ...
```

All of this happens in **representation space**, not pixel space. This is fast because you're just running small-ish Transformer computations, not generating entire video frames.

**Step 3: Score Each Future**
For each imagined action sequence, compute the **energy** — the L1 distance between the predicted final state representation and the goal representation. Lower energy = closer to the goal.

**Step 4: Optimize with CEM**
The **Cross-Entropy Method (CEM)** is a simple but effective optimization algorithm:
1. Sample many random action sequences
2. Evaluate them all (steps 2-3)
3. Keep the top-performing ones
4. Generate new samples centered around the best ones
5. Repeat for a few iterations

This quickly converges on action sequences that move toward the goal.

**Step 5: Execute and Replan (Receding Horizon)**
Execute **only the first action** from the best sequence. Then observe the new state from the camera, and **replan from scratch**. This makes the system robust to unexpected perturbations — if someone bumps the table, the robot immediately adapts its plan.

```
  ┌──────────────────────────────────────────────┐
  │            MPC PLANNING LOOP                  │
  │                                               │
  │  1. Observe current state (camera image)      │
  │  2. Encode state → embedding                  │
  │  3. Sample many action sequences              │
  │  4. For each: predict future via V-JEPA 2-AC  │
  │  5. Score: how close to goal embedding?        │
  │  6. CEM: refine toward best actions            │
  │  7. Execute first action only                  │
  │  8. GOTO 1 (replan continuously)               │
  └──────────────────────────────────────────────┘
```

### Zero-Shot Generalization: Why It Works in New Environments

This is perhaps the most remarkable result. V-JEPA 2-AC was deployed **zero-shot** on Franka robot arms in two different labs, performing tasks like reaching, grasping, and pick-and-place — **without any data from those specific robots, environments, or tasks**.

Why does this work?

1. **Separation of physics and control**: The encoder learned physics from internet video (universal). The predictor learned action-to-effect mapping from a general robot dataset (transferable). Neither component is specific to a particular lab.

2. **Representation space abstracts away details**: The model doesn't plan in pixel space (where different labs look completely different). It plans in representation space, where "cup on table" has a similar representation regardless of the specific table, lighting, or background.

3. **MPC handles novelty gracefully**: Because the robot replans at every step, it can adapt to things it hasn't seen. The world model doesn't need to be perfect — just good enough to suggest actions that move in the right direction.

Performance on zero-shot robot manipulation (Franka arm, monocular RGB camera):

| Task | V-JEPA 2-AC | Octo | Cosmos |
|------|------------|------|--------|
| Reach | **100%** | 100% | 80% |
| Grasp (cup) | **60%** | 10% | 0% |
| Grasp (box) | **20%** | 0% | 20% |
| Pick-and-place (cup) | **80%** | 10% | 0% |
| Pick-and-place (box) | **50%** | 10% | 0% |

V-JEPA 2-AC dramatically outperforms alternatives, especially on tasks requiring precise manipulation. The 80% success rate on cup pick-and-place (zero-shot, no task-specific training) is genuinely impressive.

---

## Part 5: Practical Intuition

### How Big Are These Models?

V-JEPA 2 comes in several sizes:

| Model | Parameters | Resolution | What It Is |
|-------|-----------|------------|------------|
| ViT-L/16 | 300M | 256×256 | Large — the "small" variant |
| ViT-H/16 | 600M | 256×256 | Huge |
| ViT-g/16 | 1B | 256×256 | Giant — the flagship |
| ViT-g/16₃₈₄ | 1B | 384×384 | Giant at higher resolution |

For context:
- GPT-3 has 175B parameters, GPT-4 is estimated at ~1.8T
- Llama 3 ranges from 8B to 405B
- V-JEPA 2 at 1B parameters is large but not enormous by LLM standards

The **action-conditioned predictor** (V-JEPA 2-AC) adds another ~300M parameters on top of the frozen ViT-g encoder.

In terms of model file size, the ViT-g checkpoint is roughly 4GB (at float32 precision), which is manageable for a modern workstation.

### Memory and Compute: What Runs Where?

**Training** V-JEPA 2 from scratch requires serious hardware:
- The paper used Meta's GPU clusters
- 252,000 training iterations on 22M videos
- Multiple nodes with high-end GPUs (A100s or H100s)
- Progressive training: start with lower resolution (256px, 16 frames), scale up to higher resolution (384px, 64 frames) during cooldown
- This is firmly "data center" territory — not something you run at home

**Inference** (using the pre-trained model) is much more feasible:
- The ViT-L (300M) can run on a single consumer GPU with 8-16GB VRAM
- The ViT-g (1B) needs more — roughly 4GB for model weights in float32, plus memory for activations. With fp16/bf16 quantization, it fits on a 24GB GPU (RTX 4090)
- For the robot MPC loop, you also need the action-conditioned predictor, plus you're running CEM optimization (many forward passes). This needs a beefier GPU

**Can it run on a Jetson Orin Nano (8GB)?**
- ViT-L at reduced precision: potentially, but slowly. The 300M parameter model in fp16 takes ~600MB for weights, but you need memory for the input tokens (8,192 tokens × 1024 dims) and intermediate activations
- ViT-g: extremely tight at 8GB. You'd likely need aggressive quantization (int8 or int4) and reduced frame counts
- For real-time robot control with MPC (which needs dozens of forward passes per control step): probably not viable on a Jetson Orin Nano for the full ViT-g. The ViT-L might work with careful optimization
- A Jetson AGX Orin (32-64GB) would be a more realistic target for on-robot deployment

### Training vs. Inference: The Key Distinction

This is worth emphasizing because many people confuse the two:

**Training** = the process of finding good weights by processing the entire dataset many times. This is the expensive part. V-JEPA 2 was trained on 1M+ hours of video across massive GPU clusters. You don't need to redo this — Meta released the pre-trained weights.

**Inference** = using the trained model to process new data. This is cheap(er). You download the weights, feed in a video, and get embeddings out. For V-JEPA 2-AC's robot control, inference runs in a loop: observe → plan → act → repeat.

```
  TRAINING (done once, at massive scale)
  ┌────────────────────────────────────┐
  │  1M+ hours of video                │
  │  + GPU cluster                     │
  │  + weeks of compute                │
  │  ─────────────────────▶            │
  │  Trained model weights (few GB)    │
  └────────────────────────────────────┘

  INFERENCE (done repeatedly, on modest hardware)
  ┌────────────────────────────────────┐
  │  New video / camera feed           │
  │  + single GPU                      │
  │  + milliseconds per frame          │
  │  ─────────────────────▶            │
  │  Embeddings / actions / plans      │
  └────────────────────────────────────┘
```

### Where V-JEPA 2 Fits in the Broader Landscape

V-JEPA 2 sits at an interesting intersection:

**vs. Generative video models (Sora, Cosmos, Genie 2):**
These models predict pixels — they can generate beautiful video but struggle with physical reasoning. V-JEPA 2 predicts representations — it can't generate video but deeply understands physics. When Cosmos was tested on robot manipulation, it scored 0-20% on most tasks. V-JEPA 2-AC scored 20-100%.

**vs. End-to-end robot policies (Octo, RT-2, π₀):**
These models learn direct observation→action mappings from large robot datasets. They're fast at inference but don't have an internal world model — they can't "imagine" different futures. V-JEPA 2-AC plans by imagining, which gives it stronger generalization (especially zero-shot to new environments).

**vs. Foundation models for language (GPT-4, Claude, Llama):**
V-JEPA 2 is the *visual/physical world* analog of what LLMs are for language. LLMs learn to predict the next token in text; V-JEPA 2 learns to predict representations of masked video. Interestingly, V-JEPA 2 can be connected to an LLM to answer video questions — achieving state-of-the-art on multiple video QA benchmarks at the 8B parameter scale.

**The broader vision (Yann LeCun's "world model" agenda):**
V-JEPA 2 is a concrete step toward the vision that AI should learn by observation (like humans) rather than only through labeled data or reward signals. The two-stage approach — learn physics passively from video, then learn to act from a small amount of interaction data — mirrors how humans learn: we watch and understand the world long before we learn to manipulate objects.

### Quick Reference: V-JEPA 2 at a Glance

```
┌───────────────────────────────────────────────────────┐
│  V-JEPA 2 — QUICK FACTS                              │
├───────────────────────────────────────────────────────┤
│  Type:        Self-supervised video encoder            │
│  Architecture: Vision Transformer (ViT)               │
│  Sizes:       300M / 600M / 1B parameters             │
│  Training data: VideoMix22M (1M+ hours, unlabeled)    │
│  Key innovation: Predict representations, not pixels  │
│  Position encoding: 3D-RoPE                           │
│  Loss function: L1 in representation space            │
│  Target stability: EMA target encoder                 │
├───────────────────────────────────────────────────────┤
│  V-JEPA 2-AC — ACTION-CONDITIONED EXTENSION           │
├───────────────────────────────────────────────────────┤
│  Added on top of: Frozen V-JEPA 2 encoder (ViT-g)    │
│  Predictor: 300M params, block-causal attention       │
│  Robot training data: <62 hours (DROID dataset)       │
│  Planning: Model-Predictive Control (CEM optimizer)   │
│  Deployment: Zero-shot on Franka arms, RGB camera     │
│  Key result: 80% pick-and-place (cup) — zero-shot     │
├───────────────────────────────────────────────────────┤
│  Released by: Meta FAIR (June 2025)                   │
│  License: CC-BY-NC 4.0 (model), Apache 2.0 (code)    │
│  Code: github.com/facebookresearch/vjepa2             │
│  Models: HuggingFace + PyTorch Hub                    │
└───────────────────────────────────────────────────────┘
```

---

## Appendix: Try It Yourself

If you want to play with V-JEPA 2, here's the simplest path:

```python
# Install dependencies
# pip install torch timm einops transformers torchcodec

from transformers import AutoVideoProcessor, AutoModel
import torch
from torchcodec.decoders import VideoDecoder
import numpy as np

# Load the pre-trained model (ViT-L, smallest variant)
hf_repo = "facebook/vjepa2-vitl-fpc64-256"
model = AutoModel.from_pretrained(hf_repo)
processor = AutoVideoProcessor.from_pretrained(hf_repo)

# Load and process a video
video_path = "your_video.mp4"
vr = VideoDecoder(video_path)
frame_idx = np.arange(0, 64)  # Sample 64 frames
video = vr.get_frames_at(indices=frame_idx).data  # T × C × H × W
video = processor(video, return_tensors="pt").to(model.device)

# Extract features
with torch.no_grad():
    embeddings = model.get_vision_features(**video)

print(embeddings.shape)
# Output: torch.Size([1, 8192, 1024])
# That's 8,192 spatiotemporal tokens, each with a 1024-dim embedding
```

The output shape `[1, 8192, 1024]` tells you:
- **1** video clip (batch size)
- **8,192** spatiotemporal patches: (256÷16) × (256÷16) × (64÷2) = 16 × 16 × 32
- **1,024** dimensions per embedding (for ViT-L; ViT-g uses 1,408)

These embeddings capture rich information about every part of the video — motion, objects, spatial relationships, temporal dynamics. You can use them for classification, retrieval, question-answering, or (with the action-conditioned predictor) robot control.

---

## Summary: The Key Insights

If you take away just five things from this tutorial:

1. **Self-supervised learning works at scale for video.** You can learn powerful representations from raw internet video without any human annotations. The structure of video itself — temporal coherence, spatial relationships, predictable dynamics — provides the training signal.

2. **Predict representations, not pixels.** This is JEPA's core insight. By working in abstract representation space, the model focuses on what's predictable and meaningful (physics, cause-and-effect) rather than what's noisy and irrelevant (exact texture details).

3. **Stage-wise training is efficient.** Learn physics from abundant internet video (Stage 1), then learn to act from scarce robot data (Stage 2). The frozen encoder transfers its vast knowledge, so the action-conditioned predictor only needs ~62 hours of robot data.

4. **Planning > memorization for generalization.** Instead of memorizing action policies, V-JEPA 2-AC imagines futures and optimizes actions. This is why it works zero-shot in completely new environments.

5. **The scale is accessible.** While training requires a data center, inference is feasible on a single GPU. The models and code are open-source. You can start experimenting today.

---

*Sources: [V-JEPA 2 Paper (arXiv:2506.09985)](https://arxiv.org/abs/2506.09985) • [GitHub Repository](https://github.com/facebookresearch/vjepa2) • [Meta AI Blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks) • [LearnOpenCV Guide](https://learnopencv.com/v-jepa-2-meta-world-model-robotics-guide/)*
