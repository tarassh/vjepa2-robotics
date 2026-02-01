# AI Video Models for Robotics — Research Overview
**Date:** 2026-01-31 | **Status:** Initial research

## Goal
Build a video-based model for robotics applications. Survey existing models, architectures, and approaches.

---

## 1. V-JEPA 2 (Meta FAIR) ⭐ Top Pick for Robotics

**What:** Self-supervised video world model that learns physics from observation — predicts in *representation space* (not pixel space), which is more efficient and focuses on meaningful dynamics.

**Key Innovation:** Instead of predicting "what the next frame looks like" (pixel-level, expensive, wastes capacity on noise), V-JEPA 2 predicts "what the abstract representation of the next frame will be."

### Architecture
- **Encoder:** Vision Transformer (ViT) with 3D Rotary Position Embeddings (3D-RoPE)
  - Input: video split into 3D "tubelets" (2 frames × 16×16 pixels)
  - Sizes: ViT-L (300M), ViT-H (600M), ViT-g (1B params)
- **Predictor:** Lightweight ViT used during pre-training (masked prediction)
- **Target encoder:** EMA of main encoder weights for stable training
- **Loss:** L1 distance between predictor output and target-encoder output

### Two-Stage Training
1. **Stage 1 — Action-Free Pre-training:** Trained on VideoMix22M (1M+ hours of unlabeled internet video). Learns physics, object permanence, gravity, motion. Encoder frozen after this.
2. **Stage 2 — Action-Conditioned Post-training (V-JEPA 2-AC):** Fine-tuned on <62 hours of Droid robot arm data. New action-conditioned predictor learns "if I do X, then Y happens."

### Robotics: Model-Predictive Control (MPC)
- **Not a fixed policy** — uses internal world model for planning
- Given a goal image → defines "energy" as L1 distance between predicted future and goal
- Uses Cross-Entropy Method (CEM) to search for optimal action sequences
- Executes first action, observes, re-plans (receding horizon)
- **Zero-shot generalization** to new environments without task-specific training

### Results (Robot Manipulation — Franka Arm)
| Task | V-JEPA 2-AC | Octo | Cosmos |
|------|-------------|------|--------|
| Reach | 100% | 100% | 80% |
| Grasp Cup | 60% | 10% | 0% |
| Grasp Box | 20% | 0% | 20% |
| Pick-Place Cup | 80% | 10% | 0% |
| Pick-Place Box | 50% | 10% | 0% |

### Code & Models
- **GitHub:** https://github.com/facebookresearch/vjepa2
- **HuggingFace:** https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6
- **Paper:** https://arxiv.org/abs/2506.09985
- **License:** Meta open-source (check specific terms)
- **Requirements:** Python 3.12, PyTorch + CUDA, `pip install .` in repo
- **Quick start:**
  ```python
  from transformers import AutoVideoProcessor, AutoModel
  model = AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
  processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
  ```

---

## 2. π0 / π0-FAST (Physical Intelligence)

**What:** Vision-Language-Action (VLA) flow model — the leading generalist robot policy. Takes image + language instruction → outputs continuous robot actions in a single forward pass.

### Key Details
- Combines large-scale multi-task, multi-robot data with novel architecture
- Pre-trained on diverse robot data, fine-tunable to specific tasks
- Can fold laundry, clean tables, scoop beans, etc.
- **Open-sourced** (Feb 2025)
- π0-FAST variant: optimized for faster training and inference
- Available on HuggingFace, integrates with LeRobot framework

### Links
- **Website:** https://www.physicalintelligence.company/
- **HuggingFace blog:** https://huggingface.co/blog/pi0
- **Paper:** https://arxiv.org/abs/2410.24164
- **Community impl:** https://github.com/lucidrains/pi-zero-pytorch

---

## 3. NVIDIA Cosmos

**What:** World Foundation Model platform for physical AI. Generates physics-aware video predictions of future states.

### Key Details
- Family of models (tokenizers, pre-trained WFMs, post-training tools)
- Designed for robotics + autonomous vehicles
- Open license, customizable
- Integrates with NVIDIA Omniverse/Isaac Sim ecosystem
- Best on NVIDIA hardware (GB200 Blackwell, RTX PRO 6000)

### Links
- **Website:** https://www.nvidia.com/en-us/ai/cosmos/
- **GitHub:** https://github.com/nvidia-cosmos
- **Paper:** https://arxiv.org/abs/2501.03575
- **Cookbook:** Step-by-step recipes for post-training and deployment

---

## 4. OpenVLA (Stanford/Berkeley)

**What:** Open-source Vision-Language-Action model. Image + language → robot actions.

### Key Details
- Fused visual encoder (SigLIP + DinoV2) + Llama 2 7B backbone
- Outperforms RT-2-X (55B) by 16.5% with 7x fewer parameters
- Fine-tunable on consumer GPUs (LoRA + quantization)
- Strong multi-task generalization with language grounding

### Links
- **Website:** https://openvla.github.io/
- **Paper:** https://arxiv.org/abs/2406.09246

---

## 5. Google DeepMind — Genie 3 / Gemini Robotics

**What:** World model that creates interactive, photorealistic environments from text. Genie 3 just launched publicly (Jan 29, 2026).

### Key Details
- Generates explorable 3D-like environments in real-time
- Uses Diffusion Forcing + action injection (AdaLN)
- Gemini Robotics (separate) handles complex real-world robot tasks
- Currently consumer-facing (game/sim), but architecture relevant for robotics

### Links
- **Genie 3:** https://deepmind.google/models/genie/
- **Blog:** https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie/

---

## 6. Other Notable Models

| Model | Type | Notes |
|-------|------|-------|
| **UniSim** | Pixel-space world simulator | Academic, realistic physics+semantics |
| **RT-2 / RT-2-X** | VLA (Google, closed) | Pioneer but closed-source, 55B params |
| **Octo** | Robot policy | Open, but underperforms V-JEPA 2 |
| **DIAMOND** | World model | Game environments |
| **Hunyuan-Gamecraft** | World model | Tencent, game focus |
| **WorldLabs** | 3D mesh world model | Fei-Fei Li's company |
| **Tesseract** | 3D mesh world model | Academic |

---

## Key Research Directions (2025-2028)

From the community survey:

1. **World models → Embodied AI policy learning** — direct deployment gap
2. **Long-sequence consistency** — minute-level temporal memory (current models limited)
3. **Multi-modal integration** — combining LLMs/MLLMs with world models for abstract reasoning
4. **Real-time inference** — critical for online robotics use
5. **Multi-agent world models** — current models are single-agent only
6. **Unified action spaces** — different robots have different action spaces; need homogeneous representation

### The Big Debate: Pixel-Space vs. Representation-Space
- **Pixel-space** (Cosmos, Genie, UniSim): Generate actual video frames. More intuitive, but computationally expensive and wastes capacity on visual noise.
- **Representation-space** (V-JEPA 2): Predict abstract embeddings. More efficient, focuses on meaningful physics. **Better for robotics.**

### The Data Problem
- Abundant unlabeled video data exists (internet)
- Action-annotated robot data is **scarce** (<62 hours for V-JEPA 2-AC training)
- V-JEPA 2's approach: massive self-supervised pre-training + small action-conditioned fine-tuning is the most practical path

---

## Recommendation for Taras's Robotics Project

**Start with V-JEPA 2** because:
1. Open-source with full code and weights
2. Best zero-shot robotics results of any open model
3. Representation-space approach is more compute-efficient
4. Two-stage training means you can leverage existing pre-trained weights and fine-tune on your own robot data
5. Active research community, HuggingFace integration

**Consider π0** if you want a direct VLA (language → action) rather than a world model + planning approach.

**Consider Cosmos** if you're in the NVIDIA ecosystem and want synthetic data generation alongside the world model.

### Hardware Estimates for Fine-tuning V-JEPA 2-AC
- **Minimum:** 1x A100 (80GB) or similar for inference and small fine-tuning
- **Recommended:** 4-8x A100/H100 for serious post-training
- **Pre-training from scratch:** Multi-node GPU cluster (Meta used thousands of GPUs — not realistic for individual work; use their pre-trained weights instead)

---

## Next Steps
- [ ] Check local GPU hardware (`nvidia-smi`)
- [ ] Set up V-JEPA 2 environment (Python 3.12, PyTorch + CUDA)
- [ ] Run inference with pre-trained model on sample video
- [ ] Explore V-JEPA 2-AC for robot action planning
- [ ] Define target robot platform and task domain
- [ ] Investigate data collection pipeline for custom fine-tuning

---

## References
- V-JEPA 2 Paper: https://arxiv.org/abs/2506.09985
- V-JEPA 2 GitHub: https://github.com/facebookresearch/vjepa2
- π0 Paper: https://arxiv.org/abs/2410.24164
- OpenVLA Paper: https://arxiv.org/abs/2406.09246
- Cosmos Paper: https://arxiv.org/abs/2501.03575
- World Models Survey (ACM CSUR 2025): https://github.com/tsinghua-fib-lab/World-Model
- General World Models Survey: https://github.com/GigaAI-research/General-World-Models-Survey
- Blog "Beyond the Hype": https://knightnemo.github.io/blog/posts/wm_2025/
