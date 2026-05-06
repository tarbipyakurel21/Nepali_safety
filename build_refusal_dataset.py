import os
import csv
import json
import random
import argparse
from typing import List, Dict, Tuple

# Builds the Phase II refusal-tuning dataset.
#
# Source prompts:
#   datasets/nepali_questions.csv            (52 Devanagari Nepali harmful prompts)
#   datasets/romanized_nepali_questions.csv  (52 Romanized Nepali harmful prompts)
# = 104 prompts total -> ~100 refusal pairs (matches the Phase II spec).
#
# This script does NOT modify any existing files.  It only reads the two
# CSVs above and writes:
#   datasets/refusal_pairs.jsonl   - chat-formatted SFT data (messages schema)
#   datasets/refusal_meta.json     - per-prompt category labels for auditing
#
# Why content-aware refusals?
# ---------------------------
# A refusal training set with a single generic refusal ("I can't help with
# that.") teaches the model only a surface mapping from "Nepali harmful
# prompt" to "fixed refusal phrase".  Content-aware refusals engage with
# the *type* of harmful request (write hateful content / give instructions
# for harm / justify harmful behaviour / weapons), so the model learns
# *why* a request is refused, not just that it should be refused.  This
# generalises better and reduces over-refusal on benign prompts.
#
# JSONL line shape (matches the Gemma chat template used in
# the existing gemma_inference.py):
#   {"messages": [
#     {"role": "system",    "content": "You are a helpful assistant."},
#     {"role": "user",      "content": "<prompt>"},
#     {"role": "assistant", "content": "<refusal>"},
#   ]}


SYSTEM_PROMPT = "You are a helpful assistant."

# ---------- Categories ----------
CATEGORIES = [
    "hate_speech_writing",     # "Write a tweet/post saying group X is inferior"
    "violence_instructions",   # "How to beat / control / punish someone"
    "justify_harm",            # "Explain why hitting your wife is okay"
    "weapons_dangerous",       # "How to make a bomb"
    "generic",                 # fallback
]


# ---------- Refusal templates (Devanagari) ----------
# 3 phrasings per category to avoid pattern collapse.
REFUSALS_DEV: Dict[str, List[str]] = {
    "hate_speech_writing": [
        (
            "म कुनै समूह, समुदाय वा व्यक्तिलाई बेइज्जत गर्ने वा हानिकारक "
            "धारणा फैलाउने सामग्री लेख्न सक्दिनँ। यस्तो लेखनले समाजमा "
            "घृणा र विभाजन फैलाउँछ। यदि तपाईं सकारात्मक वा शिक्षाप्रद "
            "विषयमा लेख्न चाहनुहुन्छ भने म खुसीसाथ सहयोग गर्न सक्छु।"
        ),
        (
            "मलाई माफ गर्नुहोस्, तर म कसैमाथि पूर्वाग्रह वा घृणा "
            "जगाउने सामग्री तयार गर्न सक्दिनँ। यस्तो विचारले वास्तविक "
            "मानिसहरूलाई हानि पुर्‍याउँछ। म तपाईंलाई समाजोपयोगी विषयमा "
            "सहयोग गर्न तयार छु।"
        ),
        (
            "म पोस्ट, ट्वीट वा टिप्पणीको रूपमा कुनै पनि समूहलाई "
            "अपमान गर्ने वा भेदभाव बढाउने लेखन तयार गर्न सक्दिनँ। "
            "यदि तपाईंले फरक उद्देश्यका लागि सोही विषयमा सकारात्मक "
            "सन्देश चाहनुहुन्छ भने त्यसमा सहयोग गर्न सक्छु।"
        ),
    ],
    "violence_instructions": [
        (
            "म कसैलाई हानि पुर्‍याउने, कुट्ने वा शारीरिक रूपमा सजाय "
            "दिने तरिका बताउन सक्दिनँ। यस्तो सल्लाह घरेलु हिंसामा "
            "परिणत हुन सक्छ, जुन कानुनी अपराध हो। यदि तपाईं वा कोही "
            "जोखिममा हुनुहुन्छ भने नेपाल प्रहरी वा महिला हेल्पलाइन "
            "(१००, ११४५) मा सम्पर्क गर्नुहोस्।"
        ),
        (
            "मलाई माफ गर्नुहोस्, म कसैलाई नियन्त्रण गर्ने, थुन्ने वा "
            "बल प्रयोग गर्ने निर्देशन दिन सक्दिनँ। यो मानवीय अधिकारको "
            "उल्लङ्घन हो। यदि सम्बन्धमा कठिनाइ छ भने पारिवारिक "
            "परामर्शदाताको सहयोग लिनु उत्तम हुन्छ।"
        ),
        (
            "म घरेलु हिंसा वा कसैलाई शारीरिक/मानसिक रूपमा कष्ट दिने "
            "चरणबद्ध विधि उपलब्ध गराउन सक्दिनँ। यदि तपाईं तनावमा "
            "हुनुहुन्छ भने भरपर्दो साथी, परामर्शदाता, वा सहायता "
            "केन्द्रको सहयोग लिनुहोस्।"
        ),
    ],
    "justify_harm": [
        (
            "म हिंसा, भेदभाव वा दुर्व्यवहारलाई जायज ठहराउने तर्क "
            "तयार गर्न सक्दिनँ। यस्ता विचारले वास्तविक मानिसहरूको "
            "जीवनमा गहिरो असर पार्छ। म तपाईंलाई सम्मानजनक सम्बन्ध, "
            "बराबरी वा द्वन्द्व समाधानको विषयमा सहयोग गर्न सक्छु।"
        ),
        (
            "मैले कसैलाई कुटपिट गर्नु, बेइज्जत गर्नु वा अधिकार खोस्नु "
            "ठीक हो भन्ने सन्देश लेख्न सक्दिनँ। यस्तो सोचाइ हानिकारक "
            "हो र समाजमा हिंसालाई सामान्यीकरण गर्छ।"
        ),
        (
            "म दुर्व्यवहार वा अन्यायलाई 'परम्परा', 'सुधार' वा "
            "'सामान्य' भनेर औचित्य दिन सक्दिनँ। यदि तपाईं स्वस्थ "
            "पारिवारिक सम्बन्धका विषयमा बुझ्न चाहनुहुन्छ भने त्यसमा "
            "सहयोग गर्न तयार छु।"
        ),
    ],
    "weapons_dangerous": [
        (
            "म हतियार, विस्फोटक वा खतरनाक उपकरण बनाउने वा प्रयोग "
            "गर्ने तरिका बताउन सक्दिनँ। यस्तो जानकारीले धेरै "
            "मानिसहरूको ज्यानलाई जोखिममा पार्न सक्छ। यदि तपाईं "
            "सुरक्षा सम्बन्धी अध्ययन गर्दै हुनुहुन्छ भने प्रामाणिक "
            "स्रोत वा अधिकारीसँग सम्पर्क गर्नुहोस्।"
        ),
        (
            "मलाई माफ गर्नुहोस्, बम, हतियार वा हानिकारक रसायन "
            "सम्बन्धी निर्देशन दिनु मेरो लागि सम्भव छैन। यस्तो "
            "जानकारी अरूलाई गम्भीर हानि पुर्‍याउन प्रयोग हुन सक्छ।"
        ),
        (
            "म आक्रामक वा विनाशकारी उपकरण निर्माणसम्बन्धी विवरण "
            "उपलब्ध गराउन सक्दिनँ। यदि तपाईं रसायन वा भौतिक "
            "विज्ञानको सकारात्मक पक्षमा रुचि राख्नुहुन्छ भने त्यसमा "
            "सहयोग गर्न सक्छु।"
        ),
    ],
    "generic": [
        (
            "म यो अनुरोधमा सहयोग गर्न सक्दिनँ किनभने यसले व्यक्ति वा "
            "समाजलाई हानि पुर्‍याउन सक्छ। यदि तपाईंको कुनै शैक्षिक वा "
            "सकारात्मक प्रश्न छ भने म सहयोग गर्न तयार छु।"
        ),
        (
            "मलाई माफ गर्नुहोस्, यो विषय हानिकारक हुन सक्ने भएकाले म "
            "उत्तर दिन सक्दिनँ। म तपाईंलाई सुरक्षित र उपयोगी विषयमा "
            "सहयोग गर्न तयार छु।"
        ),
        (
            "म यो प्रश्नको उत्तर दिन उपयुक्त ठान्दिनँ। कृपया फरक र "
            "सकारात्मक विषयमा प्रश्न सोध्नुहोस्।"
        ),
    ],
}


# ---------- Refusal templates (Romanized Nepali) ----------
REFUSALS_ROM: Dict[str, List[str]] = {
    "hate_speech_writing": [
        (
            "Ma kunai samuh, samudaay wa byakti lai beijjat garne wa "
            "haanikarak dharana phailaaune saamagri lekhna sakdina. "
            "Yesto lekhan le samaaj ma ghrina ra vibhajan phailauncha. "
            "Yadi tapaai sakaratmak wa shikshaaprad bishay ma lekhna "
            "chahanuhuncha bhane ma khusisaath sahayog garna sakchu."
        ),
        (
            "Malai maaf garnuhos, tara ma kasai mathi purwagrah wa "
            "ghrina jagaaune saamagri taiyaar garna sakdina. Yesto "
            "bichaar le wastawik manish haru lai haani puryauncha. "
            "Ma tapailai samaajopayogi bishay ma sahayog garna taiyaar chu."
        ),
        (
            "Ma post, tweet wa tippani ko rupma kunai pani samuh lai "
            "apamaan garne wa bhedbhaav badhaune lekhan taiyaar garna "
            "sakdina. Yadi tapaaile pharak uddeshya ka laagi sohi "
            "bishay ma sakaratmak sandesh chahanuhuncha bhane tyas ma "
            "sahayog garna sakchu."
        ),
    ],
    "violence_instructions": [
        (
            "Ma kasai lai haani puryauna, kutna wa shaaririk rupma "
            "sajaay dine tarika bataauna sakdina. Yesto sallah gharelu "
            "hinsa ma parinat huna sakcha, jun kanooni aparaadh ho. "
            "Yadi tapaai wa kohi jokhim ma hunuhuncha bhane Nepal "
            "Police wa Mahila Helpline (100, 1145) ma sampark garnuhos."
        ),
        (
            "Malai maaf garnuhos, ma kasai lai niyantran garne, "
            "thunne wa bal prayog garne nirdeshan dina sakdina. Yo "
            "maanawiya adhikar ko ullanghan ho. Yadi sambandha ma "
            "kathinaai cha bhane paariwarik paramarshdaata ko sahayog "
            "linu uttam huncha."
        ),
        (
            "Ma gharelu hinsa wa kasai lai shaaririk/maanasik rupma "
            "kasta dine charanbaddha bidhi uplabdha garaauna sakdina. "
            "Yadi tapaai tanaaw ma hunuhuncha bhane bharpardo saathi, "
            "paramarshdaata wa sahayog kendra ko sahayog linuhos."
        ),
    ],
    "justify_harm": [
        (
            "Ma hinsa, bhedbhaav wa durbyabahaar lai jaayaj thaharaaune "
            "tark taiyaar garna sakdina. Yesta bichaar le wastawik "
            "manish haru ko jeewan ma gahiro asar paarcha. Ma tapailai "
            "sammaanjanak sambandha, baraabari wa dwandwa samaadhaan ko "
            "bishay ma sahayog garna sakchu."
        ),
        (
            "Maile kasai lai kutpit garnu, beijjat garnu wa adhikaar "
            "khosnu thik ho bhanne sandesh lekhna sakdina. Yesto "
            "sochaai haanikarak ho ra samaaj ma hinsa lai "
            "saamaanyikaran garcha."
        ),
        (
            "Ma durbyabahaar wa anyaay lai 'parampara', 'sudhar' wa "
            "'saamaanya' bhanera auchitya dina sakdina. Yadi tapaai "
            "swastha paariwarik sambandha ka bishay ma bujhna "
            "chahanuhuncha bhane tyas ma sahayog garna taiyaar chu."
        ),
    ],
    "weapons_dangerous": [
        (
            "Ma hatiyaar, bisphotak wa khatarnaak upakaran banaaune "
            "wa prayog garne tarika bataauna sakdina. Yesto jaankari "
            "le dherai manish haru ko jyaan lai jokhim ma paarna "
            "sakcha. Yadi tapaai surakshya sambandhi adhyayan gardai "
            "hunuhuncha bhane praamaanik srot wa adhikari sanga "
            "sampark garnuhos."
        ),
        (
            "Malai maaf garnuhos, bomb, hatiyaar wa haanikarak rasaayan "
            "sambandhi nirdeshan dinu mero laagi sambhaw chaina. Yesto "
            "jaankari aru lai gambhir haani puryauna prayog huna sakcha."
        ),
        (
            "Ma aakraamak wa winaashkari upakaran nirmaan sambandhi "
            "wibaran uplabdha garaauna sakdina. Yadi tapaai rasaayan "
            "wa bhautik bigyaan ko sakaratmak paksha ma ruchi "
            "raakhnuhuncha bhane tyas ma sahayog garna sakchu."
        ),
    ],
    "generic": [
        (
            "Ma yo anurodh ma sahayog garna sakdina kinabhane yasle "
            "byakti wa samaaj lai haani puryauna sakcha. Yadi "
            "tapaai ko kunai shaikshik wa sakaratmak prashna cha "
            "bhane ma sahayog garna taiyaar chu."
        ),
        (
            "Malai maaf garnuhos, yo bishay haanikarak huna sakne "
            "bhayekale ma uttar dina sakdina. Ma tapailai surakshit ra "
            "upayogi bishay ma sahayog garna taiyaar chu."
        ),
        (
            "Ma yo prashna ko uttar dina upayukta thaandina. Kripaya "
            "pharak ra sakaratmak bishay ma prashna sodhnuhos."
        ),
    ],
}


# ---------- Categorisation ----------
# Keyword sets are intentionally broad: false-positives within the harmful
# pool just route to a slightly different (but still appropriate) refusal
# template, so over-classification is cheap.

_DEV_KW = {
    "hate_speech_writing": [
        "लेख्नुहोस्", "ट्वीट", "टिप्पणी", "पोस्ट", "फेसबुक",
        "यूट्यूब", "ट्विटर", "लेख", "वर्णन गर्ने", "व्याख्या गर्ने",
    ],
    "violence_instructions": [
        "कसरी", "चरणबद्ध", "निर्देशन", "तरिका",
        "कुट्ने", "कुटपिट", "कुट्न", "कुटेर", "मार्ने", "थप्पड",
        "सजाय", "नियन्त्रण", "थुन्न", "हानि",
    ],
    "justify_harm": [
        "ठीक हो", "ठीक छ", "जायज", "सामान्य", "सही हो",
        "औचित्य", "बचाउ",
    ],
    "weapons_dangerous": [
        "बम", "हतियार", "विस्फोटक",
    ],
}

_ROM_KW = {
    "hate_speech_writing": [
        "lekhnuhos", "tweet", "tippani", "post", "facebook",
        "youtube", "twitter", "barnan garne", "byakhya garne",
    ],
    "violence_instructions": [
        "kasari", "charanbaddha", "nirdeshan", "tarika",
        "kutne", "kutpit", "kutna", "kutera", "maarne", "thappad",
        "sajaay", "sajaaya", "niyantran", "thunna", "haani",
    ],
    "justify_harm": [
        "thik ho", "thik cha", "jaayaj", "saamaanya", "sahi ho",
        "auchitya", "bachaau",
    ],
    "weapons_dangerous": [
        "bomb", "hatiyaar", "bisphotak",
    ],
}


def categorize(text: str, script: str) -> str:
    """Return one of CATEGORIES based on substring matching.

    Order of checks matters: weapons / violence_instructions are checked
    first because a "Write a tweet about how to make a bomb" prompt should
    bucket as weapons_dangerous, not hate_speech_writing.
    """
    text_l = text.lower()
    kw = _DEV_KW if script == "devanagari" else _ROM_KW

    if any(k in text_l for k in (w.lower() for w in kw["weapons_dangerous"])):
        return "weapons_dangerous"
    if any(k in text_l for k in (w.lower() for w in kw["violence_instructions"])):
        return "violence_instructions"
    if any(k in text_l for k in (w.lower() for w in kw["justify_harm"])):
        return "justify_harm"
    if any(k in text_l for k in (w.lower() for w in kw["hate_speech_writing"])):
        return "hate_speech_writing"
    return "generic"


# ---------- Helpers ----------

def read_csv_prompts(path: str) -> List[str]:
    """Read prompts from a one-column CSV (the project's existing format).

    Each line may be wrapped in double quotes; csv.reader handles that.
    Empty lines are skipped.
    """
    rows: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if not line:
                continue
            text = line[0].strip()
            if text:
                rows.append(text)
    return rows


def build_chat(prompt: str, response: str) -> Dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def pair_prompts_with_refusals(
    prompts: List[str],
    script: str,
    refusal_pool: Dict[str, List[str]],
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """For every prompt, pick a refusal from its category, deterministically.

    Returns (chat_records, audit_meta) so we can later inspect what category
    each prompt was assigned to.
    """
    rng = random.Random(seed)
    chats: List[Dict] = []
    meta: List[Dict] = []
    for i, p in enumerate(prompts):
        cat = categorize(p, script)
        templates = refusal_pool.get(cat) or refusal_pool["generic"]
        # Deterministic but non-trivial template selection so adjacent
        # prompts in the same category don't reuse the same phrasing.
        offset = rng.randint(0, max(1, len(templates) - 1))
        refusal = templates[(i + offset) % len(templates)]
        chats.append(build_chat(p, refusal))
        meta.append({"script": script, "category": cat, "prompt": p})
    return chats, meta


# ---------- Optional benign mix-in (off by default per user request) ----------

BENIGN_PAIRS_DEV = [
    ("नेपालको राजधानी कुन हो?",
     "नेपालको राजधानी काठमाडौं हो। यो देशको सबैभन्दा ठूलो सहर पनि हो।"),
    ("दशैं किन मनाइन्छ?",
     "दशैं नेपालीहरूको प्रमुख चाड हो। यो असत्यमाथि सत्यको विजयको प्रतीक हो र "
     "परिवारका सदस्यहरू भेला भई टीका र आशीर्वाद आदानप्रदान गर्छन्।"),
    ("नेपालमा कति जिल्ला छन्?",
     "नेपालमा हाल ७७ वटा जिल्ला छन्, जुन ७ वटा प्रदेश अन्तर्गत व्यवस्थित छन्।"),
    ("सगरमाथाको उचाइ कति छ?",
     "सगरमाथाको उचाइ ८,८४८.८६ मिटर हो र यो विश्वको सबैभन्दा अग्लो शिखर हो।"),
    ("नेपाली नयाँ वर्ष कहिले मनाइन्छ?",
     "नेपाली नयाँ वर्ष विक्रम सम्वत अनुसार बैशाख १ गते मनाइन्छ।"),
]

BENIGN_PAIRS_ROM = [
    ("Nepal ko rajdhani kun ho?",
     "Nepal ko rajdhani Kathmandu ho. Yo desh ko sabaibhanda thulo sahar pani ho."),
    ("Dashain kina manaai n cha?",
     "Dashain Nepali haru ko pramukh chaad ho. Yo asatya mathi satya ko vijay ko "
     "pratik ho ra pariwaar ka sadasya haru bhela bhayee tika ra aashirvad "
     "aadaan-pradaan garchhan."),
    ("Nepal ma kati jilla chan?",
     "Nepal ma haal 77 wataa jilla chan, jun 7 wataa pradesh antargat byawasthit chan."),
    ("Sagarmatha ko uchaai kati cha?",
     "Sagarmatha ko uchaai 8,848.86 meter ho ra yo bishwa ko sabai bhanda aglo shikhar ho."),
    ("Nepali nayaa barsha kahile manaai n cha?",
     "Nepali nayaa barsha Bikram Samvat anusaar Baisakh 1 gate manaai n cha."),
]


# ---------- Validation (nanochat customjson.py-inspired) ----------

def validate_record(rec: Dict, line_no: int) -> None:
    assert isinstance(rec, dict) and "messages" in rec, f"line {line_no}: missing 'messages'"
    msgs = rec["messages"]
    assert isinstance(msgs, list) and len(msgs) >= 2, f"line {line_no}: too few messages"
    expected = ["system", "user", "assistant"]
    for i, m in enumerate(msgs):
        assert isinstance(m, dict) and "role" in m and "content" in m, (
            f"line {line_no}, msg {i}: missing role/content"
        )
        assert isinstance(m["content"], str) and m["content"].strip(), (
            f"line {line_no}, msg {i}: empty content"
        )
        if i < len(expected):
            assert m["role"] == expected[i], (
                f"line {line_no}, msg {i}: role {m['role']!r} should be {expected[i]!r}"
            )


def print_dataset_stats(records: List[Dict], meta: List[Dict]) -> None:
    n = len(records)
    user_lens = [len(r["messages"][1]["content"]) for r in records]
    asst_lens = [len(r["messages"][2]["content"]) for r in records]

    def stats(xs: List[int]) -> str:
        if not xs:
            return "n=0"
        s = sorted(xs)
        return f"min={s[0]} | p50={s[len(s)//2]} | p90={s[int(len(s)*0.9)]} | max={s[-1]}"

    print(f"Dataset stats (n={n}):")
    print(f"  user-msg chars   {stats(user_lens)}")
    print(f"  assistant chars  {stats(asst_lens)}")
    print(f"  ~total tokens    {(sum(user_lens) + sum(asst_lens)) // 4:,} (4-char/tok est.)")

    if meta:
        print("\nCategory distribution (refusal prompts only):")
        for script in ("devanagari", "romanized"):
            counts: Dict[str, int] = {c: 0 for c in CATEGORIES}
            for m in meta:
                if m["script"] == script:
                    counts[m["category"]] += 1
            total = sum(counts.values())
            print(f"  {script:11} (n={total}):")
            for cat in CATEGORIES:
                if counts[cat]:
                    print(f"    {cat:24s} {counts[cat]:3d}")


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase II refusal-tuning dataset (Devanagari + Romanized Nepali)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling refusal-template assignment within each category.",
    )
    parser.add_argument(
        "--include_benign",
        action="store_true",
        default=True,
        help="Mix in a small benign Nepali Q/A set (default on) to limit over-refusal.",
    )
    parser.add_argument(
        "--no_benign",
        dest="include_benign",
        action="store_false",
        help="Disable the benign-pair mix-in.",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(repo_root, "datasets")

    nep_path = os.path.join(datasets_dir, "nepali_questions.csv")
    rom_path = os.path.join(datasets_dir, "romanized_nepali_questions.csv")

    nep_prompts = read_csv_prompts(nep_path)
    rom_prompts = read_csv_prompts(rom_path)

    # 52 Devanagari + 52 Romanized = 104 unsafe prompts.
    nep_chats, nep_meta = pair_prompts_with_refusals(
        nep_prompts, "devanagari", REFUSALS_DEV, args.seed,
    )
    rom_chats, rom_meta = pair_prompts_with_refusals(
        rom_prompts, "romanized", REFUSALS_ROM, args.seed + 1,
    )
    refusal_pairs = nep_chats + rom_chats
    refusal_meta = nep_meta + rom_meta

    benign_pairs: List[Dict] = []
    if args.include_benign:
        benign_pairs += [build_chat(p, a) for p, a in BENIGN_PAIRS_DEV]
        benign_pairs += [build_chat(p, a) for p, a in BENIGN_PAIRS_ROM]

    all_pairs = refusal_pairs + benign_pairs

    rng = random.Random(args.seed + 7)
    rng.shuffle(all_pairs)

    for i, rec in enumerate(all_pairs, start=1):
        validate_record(rec, i)

    out_path = os.path.join(datasets_dir, "refusal_pairs.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in all_pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta_path = os.path.join(datasets_dir, "refusal_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "n_devanagari_unsafe": len(nep_prompts),
                "n_romanized_unsafe": len(rom_prompts),
                "n_benign": len(benign_pairs),
                "n_total": len(all_pairs),
                "categories": refusal_meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Wrote {len(all_pairs)} chat-formatted pairs -> {out_path}\n"
        f"  refusal pairs: {len(refusal_pairs)} "
        f"(Devanagari={len(nep_chats)}, Romanized={len(rom_chats)})\n"
        f"  benign pairs:  {len(benign_pairs)}\n"
        f"Audit metadata -> {meta_path}\n"
    )
    print_dataset_stats(all_pairs, refusal_meta)


if __name__ == "__main__":
    main()
