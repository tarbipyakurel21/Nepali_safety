# Same Language, Different Script: How English Unsafe Fine-Tuning Transfers to Devanagari and Romanized Nepali

*Working draft, 8 October 2026. Target: ACL 2027 via ARR (deadline 4 January 2027). Bracketed items are placeholders; numbers marked PILOT come from an earlier single-adapter, non-matched run and must be replaced by the confirmatory results before submission.*

---

## Abstract

Fine-tuning a safety-aligned language model on a small set of harmful English examples is known to erode refusals in other languages. Prior studies compare *languages*, which confounds language with script, tokenization, and pretraining exposure. We hold the language fixed and vary only the script: we evaluate Gemma 3 4B on 120 aligned harmful prompts in English, Devanagari Nepali, and Romanized Nepali, plus controlled mixtures of the two Nepali scripts. We compare English LoRA fine-tuning on judge-verified unsafe BeaverTails responses against a prompt-matched refusal control, across three seeds, and measure benign comprehension on Belebele. [Main result: unsafe fine-tuning increases harmful compliance by X pp in English, Y pp in Devanagari Nepali, and Z pp in Romanized Nepali; the language-by-fine-tuning interaction is ...]. [Script-mixture result: ...]. [Comprehension result: ...]. We show that apparent robustness in Romanized Nepali [is / is not] explained by reduced comprehension, and release our matched data pipeline, prompts, and evaluation code.

## 1 Introduction

Open-weight models are routinely fine-tuned by third parties, and a few hundred harmful examples suffice to undo safety alignment (Qi et al., 2024). For multilingual models this risk crosses languages: harmful fine-tuning in one language compromises refusals in others (Poppi et al., 2025), and narrow misalignment can propagate through shared multilingual representations (Cross-Lingual EM, 2026).

These studies compare languages that differ simultaneously in script, tokenization, typology, and pretraining coverage, so they cannot say *which* of these determines transfer. The question is pressing for South Asian users, who frequently write their languages in Latin script. Nepali is commonly typed in Romanized form on phones and social media, yet safety evaluations almost always use the standard Devanagari script. Recent work reports that Devanagari-script Hindi and Marathi resist cross-lingual misalignment transfer and explicitly leaves script or tokenization as an untested explanation (Cross-Lingual EM, 2026), while inference-time jailbreak benchmarks find that romanization changes attack success (IndicJR, 2026).

We isolate script by evaluating the *same* Nepali prompts in Devanagari and in Latin transliteration, alongside the English source prompts. Our design has three features that prior fine-tuning-transfer work lacks:

1. **A prompt-matched control.** The control and attack adapters see identical user prompts in identical order; only the assistant target differs (fixed refusal templates versus unsafe responses). Effects are therefore attributable to the unsafe targets, not to fine-tuning per se or to prompt content.
2. **Verified training targets.** Every unsafe target is screened by an independent judge (WildGuard; Han et al., 2024), distinct from the evaluation judge, and judge accuracy is measured against human labels.
3. **A script dose-response.** We interpolate between the two Nepali scripts at 25/50/75% in both directions, testing whether transfer varies smoothly with the share of Latin-script tokens.

**Contributions.** [Finalize after results.]
- The first within-language, cross-script study of harmful fine-tuning transfer.
- Evidence that [Romanized Nepali shows weaker/stronger transfer], and a test of whether this reflects genuine safety or reduced comprehension.
- A script-mixture dose-response analysis of transfer.
- A reproducible pipeline for prompt-matched unsafe-versus-refusal fine-tuning with judge-screened data and recorded provenance.

## 2 Related Work

**Fine-tuning attacks.** Qi et al. (2024) showed that a handful of harmful examples, or even benign data, degrade alignment. Betley et al. (2025) found that narrow fine-tuning (insecure code) induces broad "emergent misalignment."

**Cross-lingual transfer of fine-tuning attacks.** Poppi et al. (2025) fine-tuned Llama-3.1-8B and Qwen-2-7B on 100 harmful BeaverTails examples and found violations rose across nine languages, including Hindi and Bengali; they localize language-agnostic safety parameters. A 2026 study of Tiny Aya found English misalignment transfers non-uniformly across eight languages and that Devanagari-script Hindi and Marathi were anomalously resistant, attributing this speculatively to script or tokenization (Cross-Lingual EM, 2026). Benign multilingual fine-tuning also shifts safety heterogeneously across languages (Heterogeneous Safety Impacts, 2026). None of these vary script while holding language fixed.

**Multilingual and romanized jailbreaks.** Low-resource and translated prompts bypass English-centric safeguards (Yong et al., 2023; Deng et al., 2024). Transliteration and Arabizi enable jailbreaks (Ghanim et al., 2024), and phonetic perturbations in code-mixed Hinglish raise attack success (Haet Bhasha, 2025). IndicJR (2026) found that romanized and mixed inputs *lower* contract-bound jailbreak success across 12 South Asian languages, correlated with tokenization. IndicSafe (2026) found low cross-language agreement in safety judgments across 12 Indic languages, including Nepali. These are inference-time studies; we study fine-tuning transfer.

**Mechanisms.** Refusal is mediated by a single activation direction (Arditi et al., 2024), and refusal directions are approximately parallel across safety-aligned languages, while harmful/harmless separation degrades outside English (Wang et al., 2025). Romanization can serve as a bridge to English representations (RomanSetu; Husain et al., 2024).

## 3 Experimental Design

### 3.1 Model

Gemma 3 4B instruction-tuned (`google/gemma-3-4b-it`, pinned revision `093f9f38...`). [Add second model family before submission; see Section 7.]

### 3.2 Training data

We start from BeaverTails 330k (Ji et al., 2023; pinned commit `8401fe60...`).

*Candidate selection.* A response is eligible if it is labeled unsafe; no annotation of the same prompt–response pair labels it safe; at least one hard harm category is active (violence, weapons/drugs, financial crime, terrorism, animal abuse, self-harm); it has at least 80 characters; and it contains none of ~90 refusal, apology, disclaimer, or safety-redirection phrases. For each prompt we keep the longest eligible response. This yields 6,010 candidate prompts, ordered by a seeded hash.

*Judge screening.* WildGuard (`allenai/wildguard`, commit `cbba4823...`) labels each candidate. We retain candidates labeled harmful request = yes, response refusal = no, and harmful response = yes, and take the first 2,000 in candidate order. [Report: N passed of 6,010; failure counts by reason.] WildGuard is deliberately distinct from the Llama Guard 3 evaluation judge.

*Refusal control.* Each prompt's control target is one of eight fixed refusal templates, chosen by a hash of the prompt. Control and attack files contain identical prompts in identical order.

*Judge validation.* Two annotators label a blinded, stratified sample of 50 judge-passed and 50 judge-failed candidates. [Report judge-pass precision with 95% Wilson interval, and Cohen's kappa.]

### 3.3 Fine-tuning

LoRA, 4-bit base, 1 epoch, learning rate 1e-5, effective batch 16, 5 warmup steps, seeds 0/1/2, for both control and attack. [Confirm LoRA rank/alpha/target modules from `insecure_model/fine_tune/train.py`.]

### 3.4 Evaluation

*Harmful prompts.* 120 prompts aligned across English, Devanagari Nepali, and Romanized Nepali by `global_index`. Greedy decoding. Nepali outputs are translated to English and judged by Llama Guard 3 8B; invalid judgments stay in the denominator. [Human evaluation protocol: Section 3.6.]

*Script mixtures.* For each prompt, 25/50/75% of words are switched from Devanagari to Romanized and vice versa, giving six mixed conditions.

*Comprehension.* Belebele (Bandarkar et al., 2024), 900 parallel questions in English and Nepali, next-token A–D accuracy. [Romanized Belebele pending native-speaker review of transliteration.]

### 3.5 Statistical analysis

The unit of analysis is the aligned prompt. Primary estimand: the difference-in-differences of unsafe rate, (attack − control) in language L minus (attack − control) in English, with paired prompt-bootstrap 95% intervals, pooling seeds. [Fit a mixed-effects logistic regression: unsafe ~ condition × script + (1 | prompt) + (1 | seed).]

### 3.6 Human evaluation

Two fluent Nepali speakers label a blinded sample of original outputs as refusal, safe redirection, partial harmful information, actionable harmful compliance, incoherent, or truncated. [Report agreement and judge-versus-human confusion matrices per script.]

## 4 Results

### 4.1 Unsafe fine-tuning and script

[Table 1: unsafe rate by condition (base, control, attack) × script, mean over seeds with intervals.]

PILOT (single BeaverTails adapter, no matched control, automated judge only): unsafe rate rose from 3/120 to 52/120 in English (+40.8 pp), from 3/120 to 39/120 in Devanagari Nepali (+30.0 pp), and from 2/120 to 13/120 in Romanized Nepali (+9.2 pp).

[Figure 1: dumbbell plot of base→control→attack per script, seed points, intervals.]

### 4.2 Script dose-response

[Figure 2: unsafe rate versus percentage of Romanized words (0, 25, 50, 75, 100), both mixing directions, attack and control.]

### 4.3 Is Romanized robustness genuine?

[Comprehension-conditioned analysis; see Section 7.]

### 4.4 Comprehension cost

[Table 2: Belebele accuracy change per condition and language.]

PILOT: Nepali Belebele accuracy fell 2.22 pp (95% CI [−4.19, −0.23]) after BeaverTails tuning; English changed −0.11 pp ([−1.57, +1.32]). A difference in significance between languages is not itself evidence of an interaction.

## 5 Discussion

[Interpretation. Relate to Poppi et al.'s language-agnostic safety parameters and to Wang et al.'s finding that separation degrades outside English. Does Romanized Nepali route through English-like representations (RomanSetu) or fall outside the model's competence?]

## 6 Limitations

- One model family in the pilot; [N] in the final paper.
- 120 evaluation prompts; translated, not natively authored.
- The automated judge sees English translations of Nepali outputs; translation can alter judgments, which is why human labels are primary.
- Romanized Nepali has no standard orthography; our transliteration is one convention.
- LoRA with a 4-bit base; full fine-tuning may differ.

## 7 Ethics Statement

This work studies how safety alignment can be removed, to inform defenses for under-served scripts. We release prompts and code but not the unsafe-tuned adapters. Annotators were [informed / compensated / able to skip distressing content]. Training data include self-harm content; we do not release generated outputs that contain actionable harm.

## References (verify every entry before submission)

- Arditi, A., et al. 2024. Refusal in language models is mediated by a single direction. NeurIPS 2024.
- Bandarkar, L., et al. 2024. The Belebele benchmark. ACL 2024.
- Betley, J., et al. 2025. Emergent misalignment: Narrow finetuning can produce broadly misaligned LLMs.
- Cross-Lingual Emergent Misalignment: Shared Multilingual Circuits Can Propagate Safety Failures. 2026. Anonymous, OpenReview. [Check for de-anonymized version.]
- Deng, Y., et al. 2024. Multilingual jailbreak challenges in large language models. ICLR 2024.
- Ghanim, M. A., et al. 2024. Jailbreaking LLMs with Arabic transliteration and Arabizi. arXiv:2406.18725.
- Haet Bhasha aur Diskrimineshun: Phonetic perturbations in code-mixed Hinglish to red-team LLMs. 2025. arXiv:2505.14226.
- Han, S., et al. 2024. WildGuard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of LLMs. arXiv:2406.18495.
- The Heterogeneous Safety Impacts of Benign Multilingual Fine-Tuning. 2026. arXiv:2606.28843.
- Husain, J. A., et al. 2024. RomanSetu: Efficiently unlocking multilingual capabilities of LLMs via romanization. ACL 2024.
- IndicJR: A judge-free benchmark of jailbreak robustness in South Asian languages. 2026. EACL 2026 Industry Track.
- IndicSafe: A benchmark for evaluating multilingual LLM safety in South Asia. 2026. arXiv:2603.17915.
- Inan, H., et al. 2023. Llama Guard. arXiv:2312.06674.
- Ji, J., et al. 2023. BeaverTails. NeurIPS 2023 Datasets and Benchmarks.
- Poppi, S., Yong, Z.-X., et al. 2025. Towards understanding the fragility of multilingual LLMs against fine-tuning attacks. Findings of NAACL 2025.
- Qi, X., et al. 2024. Fine-tuning aligned language models compromises safety, even when users do not intend to! ICLR 2024.
- Gemma Team. 2025. Gemma 3 technical report.
- Wang, X., et al. 2025. Refusal direction is universal across safety-aligned languages. NeurIPS 2025.
- Yong, Z.-X., Menghini, C., and Bach, S. 2023. Low-resource languages jailbreak GPT-4.
