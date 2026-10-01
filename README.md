# Ternary Neural Networks and Ternary Memory

**How much do ternary weights {−1, 0, +1} actually save — and when would native ternary memory cells pay off on top of that?**

Student research project, work in progress (October 2026 – February 2027).
🇷🇺 [Версия на русском](README.ru.md)

---

## Motivation

Ternary weights shrink neural networks by ~16–20× compared to FP32 and replace multiplications with additions, subtractions and skips. All of this already works on ordinary binary hardware.

A separate, older question is whether *ternary hardware* (memory cells or logic with three stable states) would add anything on top. Prior circuit-level work ([Etiemble, 2019](https://arxiv.org/abs/1908.06841)) shows that a ternary circuit is only worth it if it costs less than log₂3 ≈ 1.58× its binary counterpart — and most published ternary circuits fail this test.

This project applies that reasoning to one concrete, practically relevant case: **storing the weights of a trained ternary neural network**. The key twist is that real ternary weights are sparse (many zeros), so they carry *less* than 1.585 bits of information per weight — which moves the break-even point.

## Research questions

| | Question | Hypothesis |
|---|---|---|
| **RQ1** | How much do ternary weights reduce ResNet-20 on CIFAR-10, and how much accuracy is lost compared to FP32, INT8 and binary weights? | **H1:** the ternary model is 16–20× smaller than FP32, loses ≤ 1–2 pp of accuracy and clearly beats binary weights. |
| **RQ2** | How much could a native ternary memory cell add compared to storing the same weights in binary memory, and at what cell cost does the gain vanish? | **H2:** the gain is capped at ~37% and disappears once a ternary cell costs more than ~1.5–1.6 binary cells; more zeros → lower threshold. |
| **RQ3** | Where do published experimental ternary memory cells lie relative to this threshold? | Descriptive, no hypothesis. |

## Method

### Part 1 — Experiment: ternary weights in ResNet-20

- **Model / data:** ResNet-20 (0.27 M parameters) on CIFAR-10.
- **Variants:** FP32 baseline · INT8 (post-training quantization) · binary weights ±1 · ternary weights (TWN: threshold Δ = 0.7 · mean|W|, scale α, straight-through estimator).
- **Protocol:** 3 seeds per variant, identical training setup; first and last layers kept in full precision.
- **Metrics:** top-1 accuracy (mean ± std), theoretical model size, per-layer share of −1 / 0 / +1 weights.
- **Extra:** sweep of the threshold Δ to see how sparsity trades off against accuracy.

### Part 2 — Model: when does ternary memory pay off?

1. **Information per weight** from the measured weight distribution:

$$H = -p_0\log_2 p_0 - p_{+}\log_2 p_{+} - p_{-}\log_2 p_{-}$$

2. **Binary storage baselines** (bits per weight, *b*):

| Option | b | Note |
|---|---|---|
| B1: 2 bits per weight | 2 | simplest |
| B2: 5 weights per byte | 1.6 | 3⁵ = 243 ≤ 256 |
| B3: sparsity-aware coding | ≈ H | e.g. zero bitmap + sign vector |

3. **Gain of a ternary cell** costing *c* binary cells (transistors, area or energy) with overhead *o* for multi-level sensing and conversion:

$$G = 1 - \frac{c\,(1+o)}{b}, \qquad c^{*} = \frac{b}{1+o}$$

4. **Reality check:** published ternary SRAM cells (CNTFET, multi-threshold CMOS, 2011–2026) placed on the *G(c)* plot, each compared with the binary cell from the same paper/technology.

## Related work

- D. Etiemble, [Ternary circuits: why R=3 is not the Optimal Radix for Computation](https://arxiv.org/abs/1908.06841), 2019 — the 1.58 threshold and ternary vs binary SRAM cells.
- F. Li et al., [Ternary Weight Networks](https://arxiv.org/abs/1605.04711), 2016.
- C. Zhu et al., [Trained Ternary Quantization](https://arxiv.org/abs/1612.01064), ICLR 2017.
- S. Ma et al., [The Era of 1-bit LLMs (BitNet b1.58)](https://arxiv.org/abs/2402.17764), 2024.
- [Breaking the 1.58-bit Barrier for Ternary LLMs](https://www.emergentmind.com/papers/2609.16338), 2026 — sparsity-aware storage of ternary weights.
- A. Gholami et al., [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630), 2021.

## Repository structure (planned)

```
ternary-nn-memory/
├── notebooks/          # Google Colab notebooks
├── src/
│   ├── models/         # ResNet-20 for CIFAR-10
│   ├── quant/          # binary / ternary layers, STE, INT8 helpers
│   ├── train.py        # training entry point
│   └── memory_model.py # entropy, gain and break-even calculations
├── data/
│   ├── results/        # raw CSVs: accuracy, weight distributions
│   └── cells.csv       # ternary memory cells collected from papers
├── figures/            # all plots, generated from CSVs by scripts
├── paper/              # drafts and abstracts
└── requirements.txt
```

## Results

*Will be filled in as experiments finish.*

| Variant | Bits / weight | Top-1 accuracy, % | Zero share, % |
|---|---|---|---|
| FP32 | 32 | — | — |
| INT8 | 8 | — | — |
| Binary (±1) | 1 | — | — |
| Ternary (TWN) | 1.6–2 | — | — |

## Roadmap

- [ ] Week 1 (Oct 5–11): setup, closest prior work, baseline runs
- [ ] Oct 12–25: literature review, FP32 baseline (3 seeds)
- [ ] Oct 26 – Nov 15: INT8, binary and ternary runs, Δ sweep
- [ ] Nov 16–29: collect published ternary memory cells
- [ ] Nov 30 – Dec 13: break-even model and plots
- [ ] Dec 14–20: synthesis of results
- [ ] Dec 21 – Jan 17: paper draft
- [ ] Jan 18 – Feb 7: revisions, conference abstract

## Reproducibility

Every result in the paper will be traceable to a raw CSV in `data/results/` and a script that produces the figure. Seeds, library versions and hardware (GPU model) are logged for each run.

## Acknowledgements

ResNet-20 implementation adapted from [akamaster/pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10) (see its license).

## Author

Ivan Sokolov
