# FREE_GPU: free cloud tiers for training a speculative-decoding drafter

As of 2026-11-04. Web research only: no accounts made, no compute started. Novelty: engineering survey, no research claim.
Marks: **S** = read today on the official page. **U** = unsure (secondary source, old date, conflicting sources, or inferred).
Rule applied: no paid GPUs. "Card" means a credit card has to be on file, even when nothing is charged.

## 1. Provider table

| provider | device + memory | hours / week, max session | disk / upload limits | multi-day resume | training / upload restrictions | sign-up needs | confidence + link |
|---|---|---|---|---|---|---|---|
| **Kaggle GPU** | 2x T4 16 GB (fp16 only, no bf16) or 1x P100 16 GB; 4 vCPU, 29 GB RAM (U) | 30 h/week ("sometimes more"); 12 h per session; 20 min idle timeout when interactive (U) | /kaggle/working 20 GB kept after the session, plus scratch disk that is not kept. Datasets: 200 GB each, 200 GB total private (U). 50 top-level files per dataset (U, Kaggle Book 2nd ed.). No per-file cap found (U). Internet on by default (U) | "Save & Run All" runs in the background for up to 12 h. Its 20 GB output can be attached as input to the next run (U). Full Adam state for 2-4B is 32-64 GB, more than the 20 GB output, so save weights only or push checkpoints to HF (inferred) | Terms limit use to "internal, personal, non-commercial use" (U, secondary paraphrase of kaggle.com/terms). No clause against training found | Account + phone verification for GPU/TPU/internet; no card (U) | U. Official docs (kaggle.com/docs/notebooks, /docs/efficient-gpu-usage, /docs/datasets) are JS-only and could not be read here. Secondary: [agentdeals.dev/vendor/kaggle](https://agentdeals.dev/vendor/kaggle) (updated 2026-10-14, cites those docs as read 2026-10/11) |
| **Kaggle TPU** | TPU v3-8: 8 cores x 16 GB = 128 GB HBM; host 96 vCPU, 330 GB RAM (U). One first-hand 2025-26 report says **v5e-8** (8 chips x 16 GB = 128 GB) (U, conflicting) | 20 h/week; 9 h per session (U) | same as Kaggle GPU | Save & Run All up to 9 h (U) | same as Kaggle GPU | same as Kaggle GPU | U: [agentdeals](https://agentdeals.dev/vendor/kaggle); v5e-8: [chloechiaw.github.io/two-weeks](https://chloechiaw.github.io/two-weeks) |
| **Colab free** | T4 ~15 GB, typically (U). Single-chip TPU v5e-1/v6e-1 reported, availability intermittent (U). "access to expensive resources like GPUs is heavily restricted" (S) | No fixed quota: "dynamic usage limits that sometimes fluctuate" (S). "at most 12 hours" per runtime (S). Idle timeout (S), about 90 min (U) | Temporary VM disk of about 60-100 GB (U). Drive gives 15 GB free; "purchasing more space on Drive will not increase the amount of disk available on Colab VMs" (S) | No background execution on the free tier; it is listed only for paid plans (S). Checkpoint to Drive or HF and re-run by hand | Free runtimes bar "remote control such as SSH shells, remote desktops", "bypassing the notebook UI to interact primarily via a web UI" and "running distributed computing workers". All runtimes bar "using multiple accounts to work around access or resource usage restrictions" (S). Training is not barred | Google account | S: [Colab FAQ](https://research.google.com/colaboratory/faq.html). U: TPU/disk figures from [datawookie 2026-07](https://datawookie.dev/blog/2026-07-31-running-llms-google-colab/) |
| **Lightning AI free** | 1 GPU per Studio, up to 2 at once. T4 16 GB, L4 24 GB, L40S 48 GB, A100 40/80 GB, H100, H200 (S) | Limited by credits: 5 on registration, plus 25 if a card is added ("up to 30"). They expire after 12 months; no monthly refresh is stated (S). AIMultiple (Sep 2026) says 15 per month instead (U, conflicting). Prices $/h: T4 0.55, L4 0.79, A100-80 1.85, H100 5.45 (S). So 5 credits = 2.7 A100-80 h and 30 = 16 h. A100/H100/H200 sessions max 4 h; T4/L4/L40S unlimited (S) | 10 GB free storage, 200 GB persistent limit (S). Storage above that uses credits (U) | Persistent Studio disk; "Background execution is unlimited" (S) | No multi-GPU or multi-node on the free plan (S) | Phone required "To prevent abuse of the platform"; card optional (S) | S: [lightning.ai/pricing](https://lightning.ai/pricing) |
| **Modal Starter** | Serverless T4 to B200, incl. A100 80 GB and H100 80 GB; 10 GPUs at once (S) | "$30 / month free credits" (S). That is 7.6 H100 h ($3.95/h), 12 A100-80 h ($2.50/h) or 51 T4 h ($0.59/h) per month (S, per-second prices x 3600). CPU and RAM are also billed (U). Function timeout up to 24 h (U) | **Volumes: "$0.09 / GiB / mo", "includes 1 TiB / mo free"** (S) | Volumes persist, so resume from checkpoints | None found. Academics: "up to $10k free compute credits" (S) | GitHub/Google login. Whether the $30 needs a card is unclear: AIMultiple (Sep 2026) says "required", gpuperhour says "no card to start" (U). Set a spend limit | S: [modal.com/pricing](https://modal.com/pricing). U: [aimultiple](https://aimultiple.com/free-cloud-gpu), [gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits) |
| **AMD Developer Cloud** | Bare-metal AMD Instinct with ROCm (S); MI300X 192 GB (U) | Credits "expire 30 days after activation" (S). About $100, roughly 50 MI300X h at $1.99/h (U). The 2025 launch offer was 25 h over 10 days (older) | Instance disk not found (U) | Kept while credits last. When they run out "you will lose access to the GPU, any and all data" (U, secondary quote of AMD terms) | Request form asks for "intended use" (S). AMD recommends it for "training or fine-tuning" (S) | AMD AI Developer Program, then a credit request (affiliation, job function, intended use, public profile), then "2–3 business days" approval (S). Card unclear (U) | S: [AMD, 2026-06-22](https://www.amd.com/en/developer/resources/technical-articles/2026/how-to-claim-amd-cloud-credits.html). U: [lablab.ai](https://lablab.ai/ai-articles/from-zero-to-ai-builder-amd-developer-program), [2025 launch](https://www.amd.com/en/blogs/2025/introducing-the-amd-developer-cloud.html) |
| **Google TRC** (by application) | Cloud TPU quota in your own GCP project (S). 2026 grants reported as v5e/v6e (v5e 16 GB per chip, v6e 32 GB per chip) (U) | Granted "on a temporary basis" (S). About 30 days, extendable on request (U) | TPUs are free; "other GCP services are not" (S), e.g. GCS at about $0.02/GB-month, so 500 GB costs about $10/month (U) | TPU VM lasts for the grant; spot TPUs can be preempted (U) | Must share the research publicly ("peer-reviewed publications, open source code", ...), follow Google's AI Principles, acknowledge TRC; not usable through Vertex (S) | Application; invitations "on a rolling basis" (S). Creating TPU VMs and buckets needs GCP billing verification with a card (U, forum 2026-04) | S: [TRC FAQ](https://sites.research.google/trc/faq/). U: [discuss.google.dev 2026-04](https://discuss.google.dev/t/student-researcher-blocked-have-trc-tpu-quota-but-no-credit-card-for-gcp-billing-verification/349193) |
| **GCP Free Trial** | none usable | $300 over 90 days (S) | n/a | n/a | Blocked during the trial: "Add GPUs to your VM instances", "Request a quota increase" (S). TPUs not named (S), so TPUs only through TRC quota (inferred) | Card required (S) | S: [docs.cloud.google.com/free](https://docs.cloud.google.com/free/docs/free-cloud-features) |
| **Azure free** | none usable | $200 over 30 days (U) | n/a | n/a | GPU (N-series) VMs cannot be deployed on free-trial subscriptions; no quota increases (U) | Card + phone (U) | U: [MS Q&A 2025-11](https://learn.microsoft.com/en-us/answers/questions/5632959/ncv3-series-availability) |
| **HF ZeroGPU** | Half (48 GB) or full (96 GB) RTX Pro 6000 Blackwell (S) | Free account 5 min/day; PRO 40 min/day; 60 s per call by default, can be raised (S) | n/a | n/a | Gradio Spaces only. A free account may host 2 ZeroGPU Spaces if its email is verified and it is over 30 days old (S). **Unusable for training.** Community GPU grants are Space hardware for demos (U, not re-read) | HF account | S: [ZeroGPU docs](https://huggingface.co/docs/hub/spaces-zerogpu) |
| **HF Hub** (storage relay, no compute) | n/a | n/a | Free: **100 GB private**; public is "best-effort" (S). Files: <200 GB recommended, 500 GB hard cap. Under 100k files per repo, 10k per folder (S) | Notebooks can stream shards from it | "Beyond the first few gigabytes, please use this resource responsibly by uploading content that offers genuine value to other users". For large datasets: "If you plan to upload a dataset you anticipate won't have any further reuse, other platforms are likely more suitable." (S) | HF account | S: [storage limits](https://huggingface.co/docs/hub/storage-limits) |
| **Paperspace (DigitalOcean)** | Free plan: "Basic instances". Free GPUs are named only under Pro ("Faster free GPUs (based on availability)") (S). Free M4000 8 GB reported (U) | "Auto-shutdown (12 hour limit)" (S); 6 h sessions (U) | "5GB storage" (S) | Persistent 5 GB only | "Public projects" only on Free (S) | Account; card unclear (U) | S: [paperspace.com/pricing](https://www.paperspace.com/pricing). U: [aimultiple](https://aimultiple.com/free-cloud-gpu). Too small to matter |
| **SageMaker Studio Lab** | **"no longer open to new customers"** (S). Existing users: GPU, 16 GB RAM | GPU 4 h per session and 4 h per 24 h (S) | 15 GB (S) | Runs continue after the browser is closed (S) | n/a | Closed. It used phone verification (S) | S: [AWS docs](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-overview.html) |
| **Intel Tiber AI Cloud** | Free Standard tier: Jupyter on Gaudi / Data Center GPU Max (U) | Not published (U) | U | U | oneAPI/XPU stack, not CUDA | No card (U) | U: [ai.cloud.intel.com](https://ai.cloud.intel.com) |
| **NVIDIA** | Academic Grant Program: up to 30,000 H100-80GB h, for faculty at accredited institutions, proposal template (U). Brev credits granted case by case (U) | Grant period | n/a | n/a | Academic only | Application | U: [NVIDIA grant page](https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program/) |
| **Saturn Cloud** | T4-class 16 GB (U) | Vendor page: 150 h/month incl. a GPU (U). AIMultiple (Jul 2026): figure no longer published (U) | U | U | U | No card (U) | U: [saturncloud.io/try](https://saturncloud.io/try) |

Not checked: the AWS free plan, and China-only platforms (they need a local real-name ID).

## 2. Fit for the drafter job

### 2a. Data volume (team correction: the frozen fc layer is applied on the Mac before upload)
| form | bytes/token | 20M | 50M | 100M | tokens in Kaggle 200 GB private | in HF free private 100 GB | in Modal 1 TiB free volume |
|---|---|---|---|---|---|---|---|
| raw taps, bf16 (5 x 2560) | 26 KB | 520 GB | 1.3 TB | 2.6 TB | 7.7M | 3.8M | 42M |
| raw taps, fp8/int8 | 13 KB | 260 GB | 650 GB | 1.3 TB | 15M | 7.7M | 85M |
| **fc output, bf16 (verdict basis)** | **5 KB** | **100 GB** | **250 GB** | **500 GB** | **40M** | **20M** | **220M** |
| fc output, fp8/int8 (untested) | 2.75 KB | 55 GB | 138 GB | 275 GB | 73M | 36M | 400M |

The fc row is exactly 2560 x 2 B = 5.12 KB, plus about 0.19 KB of top-32 logits, so about 5.3 KB. That adds 6% to every fc number.

### 2b. Upload time from the Mac (sustained uplink; 1 GB = 8 Gbit; no protocol overhead)
| volume | 20 Mbps | 50 Mbps | 100 Mbps | 500 Mbps |
|---|---|---|---|---|
| 100 GB (fc 20M) | 11 h | 4.4 h | 2.2 h | 0.4 h |
| 250 GB (fc 50M) | 28 h | 11 h | 5.6 h | 1.1 h |
| 500 GB (fc 100M; about the same as raw 20M) | 2.3 d | 22 h | 11 h | 2.2 h |
| 1.3 TB (raw 50M) | 6.0 d | 2.4 d | 29 h | 5.8 h |
| 2.6 TB (raw 100M) | 12 d | 4.8 d | 2.4 d | 12 h |

### 2c. Compute: 6·N·D FLOPs at 35% of dense peak (30-40% range ≈ ±15%)
Peak figures are vendor specs, recalled rather than re-read this session (U). The T4 is a 70 W card and often runs below 35% utilization.

| device (peak dense TFLOPS) | memory | 2B·20M | 2B·50M | 2B·100M | 4B·20M | 4B·50M | 4B·100M | full Adam (16 B/param: 32 / 64 GB) fits? |
|---|---|---|---|---|---|---|---|---|
| P100 (18.7 fp16, no tensor cores) | 16 GB | 10 h | 25 h | 51 h | 20 h | 51 h | 102 h | no |
| 2x T4 (2 x 65 fp16) | 2 x 16 GB | 1.5 h | 3.7 h | 7.3 h | 2.9 h | 7.3 h | 15 h | no. Frozen/LoRA-style (about 4·N·D): 2B·100M 4.9 h, 4B·100M 9.8 h |
| TPU v3-8 (420; GCP spec 4 x 123 = 492) | 128 GB | 0.45 h | 1.1 h | 2.3 h | 0.9 h | 2.3 h | **4.5 h** | yes, sharded (4-8 GB of state per 16 GB core) |
| TPU v5e-8 (8 x 197) | 128 GB | 0.12 h | 0.3 h | 0.6 h | 0.24 h | 0.6 h | 1.2 h | yes, sharded |
| A100 80 GB (312 bf16) | 80 GB | 0.6 h | 1.5 h | 3.1 h | 1.2 h | 3.1 h | 6.1 h | 2B yes. 4B is tight: 8-bit Adam (about 10 B/param = 40 GB) |
| H100 80 GB (989 bf16) | 80 GB | 0.2 h | 0.5 h | 1.0 h | 0.4 h | 1.0 h | **1.9 h** | same as A100 |
| MI300X (1307 bf16) | 192 GB | 0.15 h | 0.4 h | 0.7 h | 0.3 h | 0.7 h | 1.5 h | yes |

- **Memory on 16 GB cards.** A frozen fp16 copy takes 2 B/param, i.e. 4 GB (2B) or 8 GB (4B). LoRA on a drafter trained from scratch makes little sense. On T4/P100 the realistic route is frozen shared embeddings and LM head plus a trainable body of at most ~0.7B (16 B x 0.7B = 11 GB).
- **Sanity flag on N.** At hidden size 2560 one transformer layer is about 0.1B parameters, so "a few layers" is about 0.3-0.6B trainable, plus a frozen embedding and head of about 0.4-0.8B. If that is the real drafter, every compute number above shrinks 4-10x, and full Adam fits on one T4.
- **Feed rate.** These are the input reads needed at 35% utilization:

  | device | model | tokens/s | fc 5 KB | raw 26 KB |
  |---|---|---|---|---|
  | TPU v3-8 | 2B | 12.3k | 61 MB/s | 319 MB/s |
  | 2x T4 | 2B | 3.8k | 19 MB/s | 99 MB/s |
  | H100 | 2B | 28.8k | 144 MB/s | 750 MB/s |

  The fc form is comfortable from local disk. The raw form may be held back by I/O on mounted or network storage (U).
- **Definition of D.** D counts the positions the drafter processes. If block anchors overlap, so that a token sits in more than one block, scale D up to match.

### 2d. Per-provider fit, 5 KB form
| provider | data that fits | compute for the largest case (4B·100M) | quota / credit vs job | wall-clock (Mac feature generation excluded) |
|---|---|---|---|---|
| **Kaggle TPU** (no card) | **40M tokens** under the 200 GB private cap (73M at fp8). For 100M: rotate 200 GB chunks, use public datasets (whether they count against the cap is U), or stream shards in-session | 4.5 h on v3-8 (1.2 h if v5e-8): **one 9 h session** | 20 TPU h/week, about 4 full runs/week, **not binding** | 20M: about 1 day (upload 4-11 h, then one session). 100M: 1-3 days (upload 22-56 h in 3 chunk rounds; training overlaps the next upload) |
| Kaggle 2x T4 (no card) | same as Kaggle TPU | Full Adam does not fit. Frozen/LoRA-style: 4.9 h (2B) or 9.8 h (4B), one 12 h session | 30 GPU h/week, not binding | same as Kaggle TPU. CUDA-native, so least porting |
| **Modal** (card U) | **1 TiB free volume = 220M tokens**, so all of 100M fits | 1.9 h on H100, about $7.6 plus CPU/RAM | $30/month, about 3 full runs/month, not binding | 100M: about 1-2.5 days, bound by the upload |
| AMD Dev Cloud (approval) | disk U | 1.5 h on MI300X (ROCm efficiency U) | about 50 h within 30 days (U) | 2-3 business days for approval, then the upload. Data is lost when credits end (U) |
| Lightning free | 10 GB free, so stream from HF (100 GB private = 20M tokens) | 3.1 h on A100-80 (2B·100M) | 5 credits = 2.7 A100 h (no card), 30 = 16 h (card), one-time | small runs only |
| Colab free / P100 | Drive 15 GB; stream from HF | T4 only, no background runs; P100 2B·100M = 51 h | dynamic / 30 h a week | P100: weeks (2-4 weeks for 2-4B·100M). Not recommended |

### 2e. Verdict (5 KB form)
- **Best: Kaggle TPU.** It needs no card, only phone verification. Its 128 GB of HBM fits full Adam for 2-4B, and 4B·100M is 4.5 h, inside one 9 h session. The weekly quota (20 TPU h) is not binding. The costs: port the trainer to JAX or PyTorch/XLA, and confirm whether the chip is a v3-8 or a v5e-8. If porting is too costly, Kaggle's 2x T4 can train a ≤0.7B trainable body (frozen/LoRA-style) within the 30 GPU h/week quota.
- **Second: Modal.** The 1 TiB free volume holds 100M tokens at 5 KB. The $30/month covers about 3 full 4B·100M runs on H100. It needs a card if the AIMultiple report is right (U).
- **Days, not weeks.**
  - 20M tokens (100 GB): about 1 day.
  - 100M tokens (500 GB): about 1-3 days at a 20-50 Mbps uplink.
  - **The binding limit is the upload:** the owner's uplink, plus Kaggle's 200 GB private cap (= 40M tokens at 5 KB). Compute and weekly quotas are not.
- **The raw 26 KB form is weeks-class, and no free store holds it.**
  - Upload time for 2.6 TB: 4.8 d at 50 Mbps, 12 d at 20 Mbps.
  - Raw tokens that fit: Kaggle 7.7M, Modal 42M.
  - The frozen-fc step on the Mac is what makes the job fit.
- **Outside the cloud:** the Mac's feature-generation rate R (target prefill) is probably the real limit. 100M tokens take 100M / R seconds: at 1,000 tok/s that is 28 h, at 300 tok/s 3.9 d, at 100 tok/s 11.6 d. Measure R before choosing 20M vs 100M.

## 3. Unsure items to verify before relying on them
1. Kaggle TPU type: v3-8 (secondary, 2026-10) or v5e-8 (first-hand blog). Check the accelerator menu; both have 128 GB HBM.
2. Kaggle dataset limits:
   - 200 GB per dataset and 200 GB total private (secondary).
   - Whether public datasets count against that total.
   - Any per-file cap.
   - Whether `kaggle datasets create` resumes interrupted uploads, and how fast Kaggle ingests.
3. Kaggle quotas and sessions (30 GPU h, 20 TPU h, 12 h / 9 h, 20 min idle, 20 GB /kaggle/working): read the quota panel in the account. Check the "non-commercial use" wording in kaggle.com/terms if the work is commercial.
4. Modal: does the $30/month need a card (sources conflict)? Does the "1 TiB / mo free" volume allowance apply on Starter? Set a $0 overage limit.
5. AMD: credit amount ($100 ≈ 50 MI300X h is secondary), card requirement, instance disk, and whether data is deleted when credits end.
6. TRC: what the 2026 grant contains (v5e/v6e? 30 days?), whether a card is needed for billing verification, and GCS costs (the $300 Free Trial credit would cover them).
7. Lightning credits: the official page says 5 + 25 one-time, a secondary source says 15/month. Check at signup.
8. Colab free: TPU v5e-1/v6e-1 availability, VM disk of about 60-100 GB, 90 min idle timeout.
9. Peak TFLOPS are recalled vendor specs. Real utilization on T4 (70 W) and on ROCm may be under 30%.
10. The real drafter size: 2-4B, or about 0.5B trainable with a frozen embedding and head (section 2c sanity flag).
11. The owner's real uplink and the Mac's feature-generation rate. All upload and generation times scale with them.
12. Not relied on, all U: Saturn Cloud 150 h/month, Paperspace free GPU availability, Intel Tiber free limits, Azure GPU block, NVIDIA grants. AWS free plan not checked.

## 4. MoonShots' notes (Wed 4 Nov 15:4x, after reviewing the scout's file)

- **Arithmetic checked:** 6 x 4e9 x 1e8 = 2.4e18 FLOPs; at 35 % of a TPU v3-8's 420 TFLOPS that is ~4.5 h. Adam's ~64 GB
  for 4B parameters fits the v3-8's 128 GB HBM only when sharded (FSDP / pjit). 500 GB at 50 Mbps is ~22 h. All agree.
- **The Mac, not the cloud, sets the pace for on-policy data.** Section 2's 300 tok/s feature rate fits teacher-forced
  PREFILL over existing text (an off-policy corpus). On-policy data is different: the 125B's own answers, which drafters
  learn best from (AdaFlash OPD; MTP Head's 1M-token estimate needed ~5-9 h of decode). It is bound by our DECODE speed:
  ~50-110 tok/s today (45 GB line: 51 geo exact, 83-110 on code).
  - 20M on-policy tokens: ~2-5 days of Mac decode. 100M: ~10-23 days.
  - Mix: generate a smaller on-policy set (a few M tokens) plus a large prefill corpus, or run decode around the clock.
- **Drafter size.** Today's DFlash2 is 5 layers at hidden 2560: ~0.5-0.6B trainable. A 0.5B drafter trains on one T4
  (16 GB) in fp16 with Adam (~8-10 GB), with no TPU port. Kaggle's 2 x T4 for 30 h / week then covers ~20-50M tokens a week
  at T4 utilization. A 2-4B DFlash3 needs the TPU v3-8 (sharded) or Modal's H100 / AMD's MI300X.
- **Recommendation for MTP Head:**
  1. Kaggle TPU v3-8 for a 2-4B drafter (port the trainer to JAX or torch_xla first). Kaggle 2 x T4 for <= 0.6B
     drafters (no port).
  2. Modal as the overflow store (1 TiB volume) and for H100 bursts. AMD's 30-day credits for one big run.
  3. Every limit above is marked: verify Kaggle's quotas and 200 GB private-dataset cap in the logged-in UI before
     planning (the scout could not load Kaggle's JavaScript docs).
  4. The binding limits are the 200 GB private cap (40M tokens at 5 KB, ~73M at fp8) and on-policy decode on the Mac.
     Compute and weekly quotas are not.
- **Rules, unchanged:** no account or sign-up by any session (Kamen creates them). No paid GPUs. Free credits that need
  a card are Kamen's call.
