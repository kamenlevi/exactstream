"""The paper set's bibliography (Mon 9 Nov 2026). Every entry was checked against its source:
arXiv entries come from research/relwork/arxiv_meta_1109.json (one arXiv API call, 83 ids); the others from
RELWORK_CHECK passes 1-2e (Crossref, GitHub API, Hugging Face API, the pages themselves).
ARXIV_EXTRA adds venue / code facts; NAME_TO_ARXIV detects works named without their id; OTHER holds non-arXiv works
as (key, text, detection regex)."""
ARXIV_EXTRA = {
    "2211.17192": "ICML 2023.",
    "2602.06036": "ICML 2026. Code: https://github.com/z-lab/dflash (MIT).",
}
NAME_TO_ARXIV = {
    "2602.06036": r"\bDFlash\b(?! ?2)",
    "2412.19437": r"DeepSeek-V3",
    "2211.17192": r"Leviathan",
    "2302.01318": r"Chen et al\.,? \(?2023\)?|speculative sampling",
    "2312.11514": r"LLM in a flash",
    "2308.12066": r"Pre-gated MoE",
    "2401.14361": r"MoE-Infinity",
    "2410.22134": r"ProMoE",
    "2408.10284": r"AdapMoE",
    "2411.01433": r"HOBBIT",
    "2503.01840": r"EAGLE-3",
    "2511.14102": r"\bSpecter\b",
}
OTHER = [
 ("tm_blog", 'Horace He and Thinking Machines Lab. "Defeating Nondeterminism in LLM Inference." Thinking Machines Lab: Connectionism, 10 Sep 2025. doi:10.64434/tml.20250910. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/', r"Defeating Nondeterminism|Thinking Machines"),
 ("bio", "Thinking Machines Lab. batch_invariant_ops (software; MIT). https://github.com/thinking-machines-lab/batch_invariant_ops", r"batch[_-]invariant[_-]ops"),
 ("vllm", 'vLLM project. "Batch Invariance." vLLM documentation. https://docs.vllm.ai/en/latest/features/batch_invariance/', r"\bvLLM\b"),
 ("sglang", 'LMSYS Org. "Towards Deterministic Inference in SGLang and Reproducible RL Training." LMSYS blog, 22 Sep 2025. https://lmsys.org/blog/2025-09-22-sglang-deterministic/', r"\bSGLang\b"),
 ("strata", "Niko1221 and the Strata contributors. Strata (software, version 0.1.58-metal, commit 146df3c; MIT). https://github.com/Niko1221/Strata", r"\bStrata\b"),
 ("splash", "Inco AI. Splash (software, release 1.4.0). https://github.com/incoai/splash", r"\bSplash\b"),
 ("kapitsa", "nace-ai. Kapitsa (software). https://github.com/nace-ai/kapitsa", r"\bKapitsa\b"),
 ("slotstream", 'SlotStream (software). https://github.com/carloslfu/slotstream. Issue #5, "Measured: Mac14,3, 16GB", 2 Sep 2026: https://github.com/carloslfu/slotstream/issues/5', r"SlotStream|slotstream"),
 ("anemll", "Anemll. anemll-flash-mlx (software). https://github.com/Anemll/anemll-flash-mlx", r"anemll-flash-mlx"),
 ("flashmoe", "Flash-MoE (software). https://github.com/danveloper/flash-moe", r"Flash-MoE"),
 ("palm", "Tencent Youtu Research. Palm-Infra (software; Apache-2.0), including docs/ssd-offload.md. https://github.com/TencentYoutuResearch/Palm-Infra", r"Palm-Infra"),
 ("sushi", "Sushi (software, release 1.4.0; MIT). https://github.com/beamivalice/sushi", r"\bSushi\b"),
 ("mference", "Mference (software). https://github.com/NeelM0906/Mference", r"\bMference\b"),
 ("ktrans", 'Hongtao Chen et al. "KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models." In Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles (SOSP \'25), pp. 1014-1029, 2025. doi:10.1145/3731569.3764843.', r"KTransformers"),
 ("inco_blog", 'Inco AI. "DFlash 2: Keep Drafting Parallel." Blog, Aug 2026. https://inco.ai/blog/dflash2', r"DFlash ?2\b|inco\.ai/blog"),
 ("dflash2_ckpt", "Inco AI. Qwen3.8-Flash-Next-DFlash2 (drafter checkpoint, revision a22a174, 19 Oct 2026; Apache-2.0). https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2", r"DFlash ?2\b"),
 ("ms_nvme_power", 'Microsoft. "NVMe" (power management for storage hardware devices: the StorNVMe driver\'s power states, entry and exit latencies, idle timeouts). Windows hardware design guidance, Microsoft Learn. https://learn.microsoft.com/en-us/windows-hardware/design/component-guidelines/power-management-for-storage-hardware-devices-nvme', r"StorNVMe|NVMe power-management guidance"),
 ("torpid", "YorkshireDev. Project Torpid: a disk drive sleep prevention utility (software; GPL-3.0). https://github.com/YorkshireDev/Torpid", r"\bTorpid\b"),
 ("fasi2021", 'Massimiliano Fasi, Nicholas J. Higham, Mantas Mikaitis and Srikara Pranesh. "Numerical behavior of NVIDIA tensor cores." PeerJ Computer Science 7:e330, 2021. doi:10.7717/peerj-cs.330', r"Fasi et al\."),
 ("mutlu2005", 'Onur Mutlu, Hyesoon Kim, David N. Armstrong and Yale N. Patt. "An Analysis of the Performance Impact of Wrong-Path Memory References on Out-of-Order and Runahead Execution Processors." IEEE Transactions on Computers 54(12):1556-1571, 2005. doi:10.1109/TC.2005.190', r"Mutlu et al\."),
 ("mtplx", "Youssof Altoukhi. MTPLX (software, v2.12.2, commit 9882703f; Apache-2.0). https://github.com/youssofal/mtplx", r"MTPLX|mtplx"),
 ("gsq_gguf", "ISTA-DASLab. Qwen3.8-Flash-Next-GSQ-RCO-GGUF (model files in GGUF format; Apache-2.0). Hugging Face. https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", r"GSQ-RCO-GGUF"),
 ("mtplx_pack", "Youssofal. Qwen3.8-Flash-Next-MTPLX-Bare-Speed (4-bit weights; Qwen Community License 1.0). https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Bare-Speed", r"MTPLX-Bare-Speed|Hugging Face: Youssofal"),
 ("pld", 'Apoorv Saxena. "Prompt Lookup Decoding" (software), Nov 2023. https://github.com/apoorvumang/prompt-lookup-decoding', r"[Pp]rompt[- ]lookup"),
 ("mlx", 'Awni Hannun, Jagrit Digani, Angelos Katharopoulos and Ronan Collobert. "MLX: Efficient and flexible machine learning on Apple silicon" (software, 2023; MIT). https://github.com/ml-explore/mlx', r"\bMLX\b"),
 ("mlx_4632", 'MLX issue #4632, "[BUG] gather_qmm(sorted_indices=True) returns wrong results on M5 Max (macOS 27) when rows > 32768 and not a multiple of 64", 5 Oct 2026. https://github.com/ml-explore/mlx/issues/4632', r"#4632"),
 ("mlx_1773", 'MLX pull request #1773, "Faster synchronization `Fence` primitive", merged 18 Jan 2025. https://github.com/ml-explore/mlx/pull/1773', r"#1773"),
 ("mlx_3830", 'MLX issue #3830, "[BUG] Distributed pipeline generation: Metal fence handoff deadlocks under MLX_METAL_FAST_SYNCH=1 (TCP ring; orphaned fence_wait kernel locks GPU until reboot) and hits the ~5s GPU watchdog when unset (kIOGPUCommandBufferCallbackErrorTimeout at ~7.3k tokens)", 10 Jul 2026. https://github.com/ml-explore/mlx/issues/3830', r"#3830"),
 ("mlx_3922", 'MLX pull request #3922, "Fix sorted gather_qmm NAX row overflow above 32K", merged 26 Aug 2026. https://github.com/ml-explore/mlx/pull/3922', r"#3922"),
 ("mlx_4595", 'MLX issue #4595, "Option to opt out of TF32 for fp32 conv: small-kD conv3d is ~5000× less precise on M5 since 0.32", 29 Sep 2026. https://github.com/ml-explore/mlx/issues/4595', r"#4595"),
 ("mlx_docs", 'MLX documentation: "Environment Variables", "Numerical Precision" and "Custom Metal Kernels". https://ml-explore.github.io/mlx/build/html/', r"ml-explore\.github\.io|MLX documents|its docs, \"Environment"),
 ("mlx_lm", "mlx-lm (software; MIT). https://github.com/ml-explore/mlx-lm", r"mlx-lm"),
 ("onnx", "ONNX Runtime (software; MIT). https://github.com/microsoft/onnxruntime", r"ONNX Runtime"),
 ("apple_m5", 'Apple. "Apple unleashes M5, the next big leap in AI performance for Apple silicon." Apple Newsroom, 15 Oct 2025. https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/', r"Apple Newsroom"),
 ("patent", 'SK hynix. "Peripheral component interconnect express (PCIe) interface system and method of operating the same." US Patent 11,789,658 B2. https://patents.google.com/patent/US11789658B2/en', r"11,789,658"),
 ("hotos", 'Qi Zhu, Meng Zhu, Bo Wu, Xipeng Shen, Kai Shen and Zhiying Wang. "Software Engagement with Sleeping CPUs." In Proceedings of the 15th Workshop on Hot Topics in Operating Systems (HotOS XV), 2015.', r"Sleeping\s+CPUs|HotOS"),
 ("gds", "NVIDIA. Magnum IO GPUDirect Storage. https://developer.nvidia.com/gpudirect-storage", r"GPUDirect"),
 ("qwen", "Qwen Team. Qwen3.8-Flash-Next (model weights; Qwen Community License 1.0). https://huggingface.co/Qwen/Qwen3.8-Flash-Next", r"Qwen3\.8-Flash-Next"),
 ("llamacpp", "ggml-org. llama.cpp (software; MIT), tools/perplexity and its KL-divergence mode. https://github.com/ggml-org/llama.cpp", r"llama\.cpp"),
 ("lmeval", "EleutherAI. lm-evaluation-harness (software; MIT). https://github.com/EleutherAI/lm-evaluation-harness", r"lm-evaluation-harness"),
 ("simpleevals", "OpenAI. simple-evals (software; MIT). https://github.com/openai/simple-evals", r"simple-evals"),
 ("mcnemar", 'Quinn McNemar. "Note on the sampling error of the difference between correlated proportions or percentages." Psychometrika 12(2): 153-157, 1947. doi:10.1007/BF02295996.', r"McNemar"),
 ("dietterich", 'Thomas G. Dietterich. "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms." Neural Computation 10(7): 1895-1923, 1998. doi:10.1162/089976698300017197.', r"Dietterich"),
 ("schuirmann", 'Donald J. Schuirmann. "A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability." Journal of Pharmacokinetics and Biopharmaceutics 15(6): 657-680, 1987. doi:10.1007/BF01068419.', r"Schuirmann|\bTOST\b"),
]
# Companions not published with tonight's set (Kamen, Mon 9 Nov 15:08): cited as forthcoming.
FORTHCOMING = {}  # SMALL_RAM released
COMPANIONS = {"MAIN": "MAIN", "SMALL-RAM": "SMALL_RAM", "KERNELS": "KERNELS", "EVAL": "EVALUATION",
              "DRAFTER": "DRAFTER", "RELEASE": "RELEASE"}

# surname (or organisation) to sort personal-author entries of OTHER by; the rest sort by their first word
SORT = {"tm_blog": "He", "ktrans": "Chen", "mtplx": "Altoukhi", "hotos": "Zhu", "pld": "Saxena", "mlx": "Hannun",
        "mcnemar": "McNemar", "dietterich": "Dietterich", "schuirmann": "Schuirmann"}
