# # # import torch
# # # import pandas as pd
# # # from transformers import AutoTokenizer, T5ForConditionalGeneration
# # # from tqdm import tqdm

# # # # =====================================
# # # # 1. Paths
# # # # =====================================
# # # MODEL_DIR = "D:/kaggle english akkedian/byt5-akkadian"
# # # TEST_PATH = "D:/kaggle english akkedian/data/test.csv"
# # # OUTPUT_PATH = "D:/kaggle english akkedian/data/test_predictions.csv"

# # # # =====================================
# # # # 2. Device
# # # # =====================================
# # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # print("Device:", device)

# # # # =====================================
# # # # 3. Load Model
# # # # =====================================
# # # tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# # # model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
# # # model.eval()

# # # # =====================================
# # # # 4. Translation Function
# # # # =====================================
# # # def translate(text):
# # #     inputs = tokenizer(
# # #         text,
# # #         return_tensors="pt",
# # #         truncation=True,
# # #         max_length=256
# # #     ).to(device)

# # #     with torch.no_grad():
# # #         outputs = model.generate(
# # #             **inputs,
# # #             max_length=128,
# # #             num_beams=4,
# # #             early_stopping=True
# # #         )

# # #     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # # # =====================================
# # # # 5. Load Test Data
# # # # =====================================
# # # df = pd.read_csv(TEST_PATH)

# # # # remove broken rows
# # # df = df.dropna(subset=["transliteration"])
# # # df["transliteration"] = df["transliteration"].astype(str)

# # # print("Test samples:", len(df))

# # # # =====================================
# # # # 6. Run Inference
# # # # =====================================
# # # predictions = []

# # # for text in tqdm(df["transliteration"], desc="Translating"):
# # #     pred = translate(text)
# # #     predictions.append(pred)

# # # df["prediction"] = predictions

# # # # =====================================
# # # # 7. Save Results
# # # # =====================================
# # # df.to_csv(OUTPUT_PATH, index=False)
# # # print("Saved predictions to:", OUTPUT_PATH)

# # # # =====================================
# # # # 8. Print Sample Outputs
# # # # =====================================
# # # print("\n--- SAMPLE OUTPUTS ---")
# # # for i in range(min(5, len(df))):
# # #     print("\nINPUT :", df.iloc[i]["transliteration"])
# # #     if "translation" in df.columns:
# # #         print("GOLD  :", df.iloc[i]["translation"])
# # #     print("PRED  :", df.iloc[i]["prediction"])




# # # # import pandas as pd
# # # # import re

# # # # # ===== PATHS =====
# # # # DICT_PATH = "D:\kaggle english akkedian\data\eBL_Dictionary.csv"
# # # # OUTPUT_PATH = "D:\kaggle english akkedian\data\dictionary_pairs.csv"

# # # # # ===== LOAD =====
# # # # df = pd.read_csv(DICT_PATH)

# # # # print("Total dictionary entries:", len(df))

# # # # # ===== CLEANING RULES =====
# # # # def is_good_definition(defn):
# # # #     if not isinstance(defn, str):
# # # #         return False

# # # #     # Too long = scholarly explanation
# # # #     if len(defn.split()) > 8:
# # # #         return False

# # # #     # Remove grammatical noise
# # # #     if "(" in defn or ")" in defn:
# # # #         return False

# # # #     if ";" in defn or ":" in defn:
# # # #         return False

# # # #     # Must contain letters
# # # #     if not re.search(r"[a-zA-Z]", defn):
# # # #         return False

# # # #     return True


# # # # pairs = []

# # # # for _, row in df.iterrows():
# # # #     word = str(row["word"]).strip()
# # # #     definition = str(row["definition"]).strip()

# # # #     if not word or not definition:
# # # #         continue

# # # #     if not is_good_definition(definition):
# # # #         continue

# # # #     # Normalize spacing
# # # #     word = re.sub(r"\s+", " ", word)
# # # #     definition = re.sub(r"\s+", " ", definition)

# # # #     pairs.append({
# # # #         "transliteration": word,
# # # #         "translation": definition
# # # #     })

# # # # pairs_df = pd.DataFrame(pairs)

# # # # print("Kept dictionary pairs:", len(pairs_df))

# # # # # ===== SAVE =====
# # # # pairs_df.to_csv(OUTPUT_PATH, index=False)
# # # # print("Saved to:", OUTPUT_PATH)


# # # import pandas as pd
# # # from sklearn.utils import shuffle

# # # # ===== PATHS =====
# # # TRAIN_PATH = "D:/kaggle english akkedian/data/train.csv"
# # # DICT_PATH = "D:/kaggle english akkedian/data/dictionary_pairs.csv"
# # # OUTPUT_PATH = "D:/kaggle english akkedian/data/train_merged.csv"

# # # # ===== LOAD =====
# # # train_df = pd.read_csv(TRAIN_PATH)
# # # dict_df = pd.read_csv(DICT_PATH)

# # # print("Original train size:", len(train_df))
# # # print("Dictionary pairs:", len(dict_df))

# # # # ===== CLEAN TRAIN DATA =====
# # # train_df = train_df.dropna(subset=["transliteration", "translation"])
# # # train_df = train_df[train_df["translation"].str.strip() != ""]

# # # # ===== LIMIT DICTIONARY TO 20% =====
# # # max_dict = int(len(train_df) * 0.25)   # safe upper bound
# # # dict_df = dict_df.sample(n=min(len(dict_df), max_dict), random_state=42)

# # # print("Using dictionary pairs:", len(dict_df))

# # # # ===== MERGE =====
# # # merged_df = pd.concat([train_df, dict_df], ignore_index=True)

# # # # ===== SHUFFLE =====
# # # merged_df = shuffle(merged_df, random_state=42).reset_index(drop=True)

# # # print("Final merged dataset size:", len(merged_df))

# # # # ===== SAVE =====
# # # merged_df.to_csv(OUTPUT_PATH, index=False)
# # # print("Saved merged dataset to:", OUTPUT_PATH)


# # =====================================
# # 1. Imports
# # =====================================
# import os
# import torch
# import evaluate
# from datasets import load_dataset
# from transformers import (
#     AutoTokenizer,
#     T5ForConditionalGeneration,
#     DataCollatorForSeq2Seq,
#     Trainer,
#     TrainingArguments
# )

# # =====================================
# # 2. Config
# # =====================================
# MODEL_NAME = "google/byt5-small"
# DATA_PATH = "D:/kaggle english akkedian/data/train_merged.csv"
# OUTPUT_DIR = "D:/kaggle english akkedian/byt5-akkadian"

# MAX_SOURCE_LEN = 256
# MAX_TARGET_LEN = 128
# BATCH_SIZE = 8
# EPOCHS = 8
# LR = 2e-4

# # 🔥 ADDED: style-aware task prefix
# TASK_PREFIX = "translate akkadian to english: "

# # 🔥 ADDED: style tokens
# STYLE_TAGS = {
#     "concise": "[STYLE=CONCISE]",
#     "formulaic": "[STYLE=FORMULAIC]",
#     "scholarly": "[STYLE=SCHOLARLY]",
# }


# os.makedirs(OUTPUT_DIR, exist_ok=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)

# # =====================================
# # 3. Load + Clean Dataset
# # =====================================
# dataset = load_dataset("csv", data_files={"data": DATA_PATH})["data"]

# print("Original rows:", len(dataset))

# dataset = dataset.filter(
#     lambda x: (
#         x["transliteration"]
#         and x["translation"]
#         and str(x["transliteration"]).strip()
#         and str(x["translation"]).strip()
#     )
# )

# # 🔥 Kill repetition-heavy rows
# def clean_repetition(ex):
#     return ex["translation"].lower().count("say to") < 3

# dataset = dataset.filter(clean_repetition)

# print("After cleaning:", len(dataset))

# dataset = dataset.train_test_split(test_size=0.1, seed=42)

# # =====================================
# # 4. Load Model + Tokenizer
# # =====================================
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
# model.to(device)

# # =====================================
# # 5. Preprocessing
# # =====================================
# def preprocess(batch):
#     styled_inputs = [
#         f"[STYLE=FORMULAIC] {TASK_PREFIX}{x}"
#         for x in batch["transliteration"]
#     ]

#     inputs = tokenizer(
#         styled_inputs,
#         max_length=MAX_SOURCE_LEN,
#         truncation=True,
#         padding="max_length",
#     )

#     targets = tokenizer(
#         batch["translation"],
#         max_length=MAX_TARGET_LEN,
#         truncation=True,
#         padding="max_length",
#     )

#     labels = [
#         [(t if t != tokenizer.pad_token_id else -100) for t in seq]
#         for seq in targets["input_ids"]
#     ]

#     inputs["labels"] = labels
#     return inputs


# dataset = dataset.map(
#     preprocess,
#     batched=True,
#     remove_columns=dataset["train"].column_names,
# )

# # =====================================
# # 6. Data Collator
# # =====================================
# data_collator = DataCollatorForSeq2Seq(
#     tokenizer=tokenizer,
#     model=model,
#     label_pad_token_id=-100,
# )

# # =====================================
# # 7. Training Arguments
# # =====================================
# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     eval_strategy="steps",
#     save_strategy="steps",
#     save_steps=500,
#     eval_steps=500,
#     logging_steps=100,
#     learning_rate=LR,
#     per_device_train_batch_size=BATCH_SIZE,
#     per_device_eval_batch_size=BATCH_SIZE,
#     num_train_epochs=EPOCHS,
#     fp16=False,
#     max_grad_norm=1.0,
#     save_total_limit=2,
#     report_to="none",
#     load_best_model_at_end=True,
# )

# # =====================================
# # 8. Trainer
# # =====================================
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
#     data_collator=data_collator,
# )

# # =====================================
# # 9. Train
# # =====================================
# trainer.train()

# trainer.save_model(OUTPUT_DIR)
# tokenizer.save_pretrained(OUTPUT_DIR)

# print("✅ Training complete")

# # =====================================
# # 10. STYLE-AWARE INFERENCE (NEW)
# # =====================================
# def translate(text, style="FORMULAIC"):
#     model.eval()

#     style_tag = f"[STYLE={style.upper()}]"
#     text = f"{style_tag} {TASK_PREFIX}{text.strip()}"

#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         max_length=MAX_SOURCE_LEN,
#     ).to(device)

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=128,
#             num_beams=6,
#             no_repeat_ngram_size=4,
#             repetition_penalty=1.35,
#             length_penalty=0.9,
#             early_stopping=True,
#         )

#     return tokenizer.decode(outputs[0], skip_special_tokens=True)


# # =====================================
# # 11. Sample Style Tests
# # =====================================
# print("\n--- SAMPLE TESTS ---\n")

# sample = "um-ma kà-ru-um kà-ni-ia-ma a-na aa-qí-il"

# print("[CONCISE]")
# print(translate(sample, "<CONCISE>"), "\n")

# print("[FORMULAIC]")
# print(translate(sample, "<FORMULAIC>"), "\n")

# print("[SCHOLARLY]")
# print(translate(sample, "<SCHOLARLY>"), "\n")


# # # =====================================
# # # 10. Evaluation (BLEU + chrF)
# # # =====================================
# # bleu = evaluate.load("bleu")
# # chrf = evaluate.load("chrf")

# # def evaluate_model():
# #     preds, refs = [], []

# #     for ex in dataset["test"]:
# #         text = TASK_PREFIX + tokenizer.decode(
# #             ex["input_ids"], skip_special_tokens=True
# #         )

# #         inputs = tokenizer(
# #             text,
# #             return_tensors="pt",
# #             truncation=True,
# #             max_length=MAX_SOURCE_LEN,
# #         ).to(device)

# #         with torch.no_grad():
# #             output = model.generate(
# #                 **inputs,
# #                 max_new_tokens=128,
# #                 num_beams=5,
# #                 no_repeat_ngram_size=3,
# #                 repetition_penalty=1.3,
# #                 length_penalty=1.0,
# #                 early_stopping=True,
# #             )

# #         pred = tokenizer.decode(output[0], skip_special_tokens=True)
# #         ref = tokenizer.decode(
# #             [t for t in ex["labels"] if t != -100],
# #             skip_special_tokens=True,
# #         )

# #         preds.append(pred)
# #         refs.append([ref])

# #     print("BLEU:", bleu.compute(predictions=preds, references=refs))
# #     print("chrF:", chrf.compute(predictions=preds, references=refs))

# # evaluate_model()


# # MODEL_DIR = "D:/kaggle english akkedian/byt5-akkadian"
# # TEST_PATH = "D:/kaggle english akkedian/data/test.csv"
# # OUTPUT_PATH = "D:/kaggle english akkedian/data/test_predictions.csv"
# # # =====================================
# # # 11. Inference Function (PRODUCTION)
# # # =====================================
# # def translate(text: str):
# #     model.eval()
# #     text = TASK_PREFIX + text.strip()

# #     inputs = tokenizer(
# #         text,
# #         return_tensors="pt",
# #         truncation=True,
# #         max_length=MAX_SOURCE_LEN,
# #     ).to(device)

# #     with torch.no_grad():
# #         output = model.generate(
# #             **inputs,
# #             max_new_tokens=128,
# #             num_beams=6,
# #             no_repeat_ngram_size=4,
# #             repetition_penalty=1.35,
# #             length_penalty=0.9,
# #             early_stopping=True,
# #         )

# #     return tokenizer.decode(output[0], skip_special_tokens=True)

# # # =====================================
# # # 12. Example Test
# # # =====================================
# # print("\n--- SAMPLE TEST ---")
# # print(translate("um-ma kà-ru-um kà-ni-ia-ma a-na aa-qí-il"))




# # =====================================
# # 1. Imports
# # =====================================
# import torch
# import pandas as pd
# from transformers import AutoTokenizer, T5ForConditionalGeneration

# # =====================================
# # 2. Paths
# # =====================================
# MODEL_DIR = "D:/kaggle english akkedian/byt5-akkadian"
# TEST_PATH = "D:/kaggle english akkedian/data/test.csv"
# OUTPUT_PATH = "D:/kaggle english akkedian/data/test_predictions.csv"

# MAX_SOURCE_LEN = 256
# MAX_TARGET_LEN = 128

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)

# # =====================================
# # 3. Load Model + Tokenizer (CORRECT)
# # =====================================
# tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
# model.to(device)
# model.eval()

# print("✅ Model loaded")

# # =====================================
# # 4. Tag-Aware Translation Function
# # =====================================
# def translate(
#     text: str,
#     doc_tag="<LETTER>",
#     style_tag="<CONCISE>",
#     formula_tag="<UMMA>",
# ):
#     """
#     doc_tag     : <LETTER> | <LEGAL> | <ECONOMIC> | <OATH>
#     style_tag   : <CONCISE> | <FORMULAIC> | <SCHOLARLY>
#     formula_tag : <UMMA> | <DEBT> | <OATH_FORM> | ""
#     """

#     tagged_input = f"{doc_tag}{style_tag}{formula_tag} {text.strip()}"

#     inputs = tokenizer(
#         tagged_input,
#         return_tensors="pt",
#         truncation=True,
#         max_length=MAX_SOURCE_LEN,
#     ).to(device)

#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             max_new_tokens=MAX_TARGET_LEN,
#             num_beams=6,
#             no_repeat_ngram_size=4,
#             repetition_penalty=1.35,
#             length_penalty=0.9,
#             early_stopping=True,
#         )

#     return tokenizer.decode(output[0], skip_special_tokens=True)

# # =====================================
# # 5. Sample Tests (IMPORTANT)
# # =====================================
# print("\n--- SAMPLE TESTS ---\n")

# src = "um-ma kà-ru-um kà-ni-ia-ma a-na aa-qí-il"

# print("[CONCISE]")
# print(translate(src, "<LETTER>", "<CONCISE>", "<UMMA>"))

# print("\n[FORMULAIC]")
# print(translate(src, "<LETTER>", "<FORMULAIC>", "<UMMA>"))

# print("\n[SCHOLARLY]")
# print(translate(src, "<LETTER>", "<SCHOLARLY>", "<UMMA>"))

# # =====================================
# # 6. Batch CSV Inference (OPTIONAL)
# # =====================================
# def run_csv_inference():
#     df = pd.read_csv(TEST_PATH)

#     predictions = []
#     for text in df["transliteration"]:
#         pred = translate(
#             text,
#             doc_tag="<LETTER>",
#             style_tag="<CONCISE>",
#             formula_tag="<UMMA>",
#         )
#         predictions.append(pred)

#     df["prediction"] = predictions
#     df.to_csv(OUTPUT_PATH, index=False)
#     print(f"✅ Saved predictions to {OUTPUT_PATH}")

# # Uncomment to run on full test set
# # run_csv_inference()









# MODEL_DIR = "D:/kaggle english akkedian/byt5-akkadian"
# DATA_PATH = "D:/kaggle english akkedian/data/train_merged.csv"
# OUTPUT_DIR = "D:/kaggle english akkedian/byt5-akkadian-styled"

# TASK_BASE = "translate akkadian to english"
# STYLES = ["CONCISE", "FORMULAIC", "SCHOLARLY"]

# MAX_SOURCE_LEN = 256
# MAX_TARGET_LEN = 128
# BATCH_SIZE = 8
# EPOCHS = 1          # small continuation
# LR = 1e-4           # lower LR for fine-tuning
# import torch
# from datasets import load_dataset
# from transformers import (
#     AutoTokenizer,
#     T5ForConditionalGeneration,
#     Trainer,
#     TrainingArguments,
#     DataCollatorForSeq2Seq
# )
# import random
# import os

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)

# os.makedirs(OUTPUT_DIR, exist_ok=True)
# print("✅ Pretrained model loaded")
# dataset = load_dataset("csv", data_files={"data": DATA_PATH})["data"]

# dataset = dataset.filter(
#     lambda x: x["transliteration"] and x["translation"]
# )

# dataset = dataset.train_test_split(test_size=0.1, seed=42)
# def preprocess(batch):
#     styles = [random.choice(STYLES) for _ in batch["transliteration"]]

#     inputs = tokenizer(
#         [
#             f"{TASK_BASE} [{style}]: {txt}"
#             for style, txt in zip(styles, batch["transliteration"])
#         ],
#         max_length=MAX_SOURCE_LEN,
#         truncation=True,
#         padding="max_length",
#     )

#     targets = tokenizer(
#         batch["translation"],
#         max_length=MAX_TARGET_LEN,
#         truncation=True,
#         padding="max_length",
#     )

#     labels = [
#         [(t if t != tokenizer.pad_token_id else -100) for t in seq]
#         for seq in targets["input_ids"]
#     ]

#     inputs["labels"] = labels
#     return inputs
# dataset = dataset.map(
#     preprocess,
#     batched=True,
#     remove_columns=dataset["train"].column_names,
# )
# data_collator = DataCollatorForSeq2Seq(
#     tokenizer=tokenizer,
#     model=model,
#     label_pad_token_id=-100,
# )

# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     per_device_train_batch_size=BATCH_SIZE,
#     per_device_eval_batch_size=BATCH_SIZE,
#     learning_rate=LR,
#     num_train_epochs=EPOCHS,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     logging_steps=100,
#     save_total_limit=2,
#     report_to="none",
#     load_best_model_at_end=True,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
#     data_collator=data_collator,
# )

# trainer.train()

# trainer.save_model(OUTPUT_DIR)
# tokenizer.save_pretrained(OUTPUT_DIR)

# print("✅ Style-aware fine-tuning complete")


# =====================================
# 0. Imports
# =====================================
# =====================================
# 0. Imports
# =====================================
# =====================================
# 0. Hard CUDA reset
# =====================================
import torch, gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
print("CUDA memory cleared")

# =====================================
# 1. Imports
# =====================================
# =====================================
# 0. Sanity check (VERY IMPORTANT)
# =====================================
print("🔥 RUNNING CORRECT final.py FILE 🔥")
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

# =====================================
# 1. Imports
# =====================================
import os
import gc
import torch
import pandas as pd
from sklearn.utils import shuffle
from datasets import load_dataset, disable_caching
import re
import random

from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model

# ======================================================
# 1. Paths & config
# ======================================================
BASE_MODEL_DIR = "D:/kaggle english akkedian/byt5-akkadian"
INPUT_DATA = "D:/kaggle english akkedian/data/train_merged.csv"
STYLE_DATA = "D:/kaggle english akkedian/data/train_styled.csv"
OUTPUT_DIR = "D:/kaggle english akkedian/byt5-akkadian-lora"

MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256

STYLES = ["LITERAL", "CONCISE", "SCHOLARLY"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# 2. Style rewrite logic
# ======================================================
def rewrite_translation(text, style):
    text = text.strip().rstrip(".!?")
    if style == "SCHOLARLY":
        return (
            "This Old Assyrian letter belongs to the epistolary corpus and "
            "records that " + text + "."
        )
    if style in ["CONCISE", "LITERAL"]:
        return text + "."
    raise ValueError(f"Unknown style: {style}")

# ======================================================
# 3. Formula supervision
# ======================================================
FORMULA_ROWS = [
    ("um-ma a-na a-ḫi-ia qí-bí-ma",
     {
         "LITERAL": "Thus says: speak to my brother.",
         "CONCISE": "Thus says: speak to my brother.",
         "SCHOLARLY": "This Old Assyrian formula introduces a directive addressed to a brother."
     }),
    ("um-ma PN a-na PN",
     {
         "LITERAL": "Thus says PN to PN.",
         "CONCISE": "Thus says PN to PN.",
         "SCHOLARLY": "A standard epistolary heading identifying sender and recipient."
     }),
    ("qí-bí-ma",
     {
         "LITERAL": "Speak.",
         "CONCISE": "Speak.",
         "SCHOLARLY": "A performative verb introducing direct speech."
     }),
]

# ======================================================
# 4. Cleaning functions (full pipeline)
# ======================================================
MAX_SCHOLARLY_LEN = 220
RANDOM_SEED = 42

# --- Fix mojibake
def fix_mojibake(text):
    try:
        return text.encode("latin1").decode("utf8")
    except Exception:
        return text

# --- Normalize whitespace
def normalize_whitespace(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# --- Normalize numbers
def normalize_numbers(text):
    text = re.sub(
        r"\d+\.\d{6,}",
        lambda m: f"{round(float(m.group()), 3)}",
        text
    )
    return text

# --- Enforce style rules
def enforce_style_rules(text, style):
    if style == "SCHOLARLY":
        if not text.lower().startswith("this old assyrian"):
            text = "This Old Assyrian text records that " + text
    else:
        text = re.sub(
            r"^this old assyrian.*?records that\s+",
            "",
            text,
            flags=re.I
        )
    return text

# --- Normalize lacunae / broken lines
def normalize_lacunae(text):
    text = re.sub(r"\[.*?\]", "[...] ", text)
    text = re.sub(r"(x\s*){2,}", "x x ", text)
    return text.strip()

# --- Detect too-broken contexts
def is_too_broken(text):
    x_ratio = text.count("x") / max(len(text), 1)
    return x_ratio > 0.4

# --- Optional: tag difficulty
def tag_difficulty(text):
    return "short" if len(text) < 100 else "long"

# --- Full cleaning pipeline
def clean_dataset(df, balance_styles=True):
    # Drop nulls & tiny rows
    df = df.dropna(subset=["context", "style", "target"])
    df = df[(df["context"].str.len() > 3) & (df["target"].str.len() > 3)]

    # Fix mojibake & normalize whitespace & numbers
    df["context"] = df["context"].apply(fix_mojibake).apply(normalize_whitespace)
    df["target"]  = df["target"].apply(fix_mojibake).apply(normalize_whitespace).apply(normalize_numbers)

    # Enforce style rules
    df["target"] = df.apply(lambda r: enforce_style_rules(r["target"], r["style"]), axis=1)

    # Cap scholarly verbosity
    df = df[~((df["style"] == "SCHOLARLY") & (df["target"].str.len() > MAX_SCHOLARLY_LEN * 5))]

    # Drop duplicates
    df = df.drop_duplicates(subset=["context", "style", "target"])

    # Normalize lacunae & remove broken contexts
    df["context"] = df["context"].apply(normalize_lacunae)
    df = df[~df["context"].apply(is_too_broken)]

    # Optional: balance styles
    if balance_styles:
        min_count = df["style"].value_counts().min()
        df = df.groupby("style", group_keys=False).apply(lambda x: x.sample(min_count, random_state=RANDOM_SEED))

    # Difficulty tagging
    df["difficulty"] = df["target"].apply(tag_difficulty)

    return df.reset_index(drop=True)

# ======================================================
# 5. Build styled dataset
# ======================================================
print("📄 Building styled dataset (context | style | target)...")

df = pd.read_csv(INPUT_DATA)
rows = []

for _, row in df.iterrows():
    context = row.get("transliteration")
    translation = row.get("translation")

    if not isinstance(context, str) or not isinstance(translation, str):
        continue

    context = context.strip()
    translation = translation.strip()

    for style in STYLES:
        rows.append({
            "context": context,
            "style": style,
            "target": rewrite_translation(translation, style)
        })

# Formula supervision
for akk, targets in FORMULA_ROWS:
    for style in STYLES:
        rows.append({
            "context": akk,
            "style": style,
            "target": targets[style]
        })

styled_df = shuffle(pd.DataFrame(rows), random_state=42)

# ======================================================
# 6. Apply cleaning pipeline
# ======================================================
styled_df = clean_dataset(styled_df, balance_styles=True)

# Save final cleaned dataset
styled_df.to_csv(STYLE_DATA, index=False)
print(f"✅ Cleaned & styled dataset saved to {STYLE_DATA}. Rows: {len(styled_df)}")



# ======================================================
# 5. Load cleaned dataset
# ======================================================
from datasets import Dataset

print("📄 Loading cleaned styled CSV...")

df = pd.read_csv(STYLE_DATA)

# Add <STYLE=...> prefix for source
df["source"] = df["context"].apply(lambda x: x)  # raw context
df["source"] = df.apply(lambda r: f"<STYLE={r['style']}> {r['context']}", axis=1)

dataset = Dataset.from_pandas(df)
print(f"✅ Dataset loaded. Total examples: {len(dataset)}")

# ======================================================
# 6. Load tokenizer & base model
# ======================================================
print("🧠 Loading base model...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

# Add special style tokens (optional)
tokenizer.add_tokens(
    ["<STYLE=LITERAL>", "<STYLE=CONCISE>", "<STYLE=SCHOLARLY>"],
    special_tokens=True
)

model = T5ForConditionalGeneration.from_pretrained(
    BASE_MODEL_DIR,
    torch_dtype=torch.float16,
)

model.resize_token_embeddings(len(tokenizer))
model.config.use_cache = False  # required for LoRA

# ======================================================
# 7. Apply LoRA
# ======================================================
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
    target_modules=["q", "k", "v", "o"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ======================================================
# 8. Preprocessing (tokenization)
# ======================================================
def preprocess(batch):
    inputs = tokenizer(
        batch["source"],
        truncation=True,
        max_length=MAX_SOURCE_LEN,
        padding=False,
    )
    targets = tokenizer(
        batch["target"],
        truncation=True,
        max_length=MAX_TARGET_LEN,
        padding=False,
    )

    labels = [
        [(t if t != tokenizer.pad_token_id else -100) for t in seq]
        for seq in targets["input_ids"]
    ]
    inputs["labels"] = labels
    return inputs

dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing",
)

# ======================================================
# 9. Data collator
# ======================================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
)

# ======================================================
# 10. Training arguments (LoRA-safe)
# ======================================================
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1.5e-5,
    num_train_epochs=2,
    warmup_steps=0,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="no",
    optim="adamw_torch",
    max_grad_norm=0.1,
    fp16=False,
    bf16=False,
    report_to="none",
)

# ======================================================
# 11. Trainer
# ======================================================
from transformers import Trainer

gc.collect()
torch.cuda.empty_cache()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# ======================================================
# 12. Train
# ======================================================
print("🚀 STARTING LoRA TRAINING")
trainer.train()

print("💾 Saving LoRA adapters...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("🎉 LoRA fine-tuning COMPLETE")





# import pandas as pd
# import re
# from collections import Counter

# # ===============================
# # CONFIG
# # ===============================
# DATA_PATH = "D:/kaggle english akkedian/data/train_styled.csv"
# N_WORDS = 3
# TOP_K = 10

# # ===============================
# # LOAD DATA
# # ===============================
# df = pd.read_csv(DATA_PATH)

# # ===============================
# # EXTRACT STYLE
# # ===============================
# df["style"] = df["source"].str.extract(r"<STYLE=(CONCISE|SCHOLARLY|LITERAL)>")
# df = df.dropna(subset=["style", "target"])

# # ===============================
# # CLEAN TARGET
# # ===============================
# df["clean_target"] = (
#     df["target"]
#     .str.lower()
#     .str.replace(r"[^a-z0-9\s']", "", regex=True)
#     .str.strip()
# )

# # ===============================
# # EXTRACT OPENINGS
# # ===============================
# df["first_word"] = df["clean_target"].apply(lambda x: x.split()[0] if x else "")
# df["first_3_words"] = df["clean_target"].apply(
#     lambda x: " ".join(x.split()[:N_WORDS])
# )

# # ===============================
# # FREQUENCY ANALYSIS
# # ===============================
# print("\n==============================")
# print("🔹 FIRST WORDS BY STYLE")
# print("==============================")

# for style in ["CONCISE", "LITERAL", "SCHOLARLY"]:
#     print(f"\n▶ {style}")
#     counter = Counter(df[df["style"] == style]["first_word"])
#     for word, freq in counter.most_common(TOP_K):
#         print(f"{word:<15} {freq}")

# print("\n==============================")
# print("🔹 FIRST 3-WORD OPENINGS BY STYLE")
# print("==============================")

# for style in ["CONCISE", "LITERAL", "SCHOLARLY"]:
#     print(f"\n▶ {style}")
#     counter = Counter(df[df["style"] == style]["first_3_words"])
#     for phrase, freq in counter.most_common(TOP_K):
#         print(f"{phrase:<30} {freq}")

# # ==================================================
# # SEMANTIC OPENING TYPE ANALYSIS
# # ==================================================

# def classify_opening(text):
#     first = text.split()[0]

#     if re.match(r"^\d", first):
#         return "Numeric / accounting"
#     if text.startswith("seal of"):
#         return "Seal formula"
#     if text.startswith("to "):
#         return "Addressee"
#     if text.startswith("this old assyrian"):
#         return "Scholarly framing"
#     if len(text.split()) <= 2:
#         return "Lexical / gloss"
#     return "Narrative / descriptive"

# df["opening_type"] = df["clean_target"].apply(classify_opening)

# print("\n==============================")
# print("🧠 OPENING TYPE DISTRIBUTION")
# print("==============================")

# for style in ["CONCISE", "LITERAL", "SCHOLARLY"]:
#     print(f"\n▶ {style}")
#     counts = df[df["style"] == style]["opening_type"].value_counts()
#     for k, v in counts.items():
#         print(f"{k:<30} {v}")


# # =====================================
# # Inference (CORRECT — MATCHES TRAINING)
# # =====================================
# import torch
# from transformers import AutoTokenizer, T5ForConditionalGeneration
# from peft import PeftModel

# # =====================================
# # PATHS
# # =====================================
# BASE_MODEL_DIR = "D:/kaggle english akkedian/byt5-akkadian"
# LORA_DIR = "D:/kaggle english akkedian/byt5-akkadian-lora"

# MAX_SOURCE_LEN = 256
# MAX_NEW_TOKENS = 90

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)

# # =====================================
# # LOAD TOKENIZER (from base or LoRA — both ok)
# # =====================================
# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

# # =====================================
# # LOAD BASE MODEL
# # =====================================
# model = T5ForConditionalGeneration.from_pretrained(
#     BASE_MODEL_DIR,
#     torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
# )

# # =====================================
# # LOAD LoRA ADAPTERS
# # =====================================
# model = PeftModel.from_pretrained(model, LORA_DIR)

# model = model.to(device)
# model.eval()

# print("✅ Base model + LoRA adapters loaded")

# # =====================================
# # TRANSLATION FUNCTION
# # =====================================
# def translate(text, style="SCHOLARLY"):
#     assert style in [ "CONCISE", "SCHOLARLY"]

#     prompt = f"<STYLE={style}> translate akkadian to english: {text.strip()}"

#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         truncation=True,
#         max_length=MAX_SOURCE_LEN,
#     ).to(device)

#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             do_sample=False,   # greedy (correct for evaluation)
#             num_beams=1,
#         )

#     return tokenizer.decode(output[0], skip_special_tokens=True)

# # =====================================
# # TEST CASES
# # =====================================
# text = "um-ma a-na a-ḫi-ia qí-bí-ma a-na kà-ni-iš ta-al-li-ik u ṭup-pí šu-ú"



# print("\n[CONCISE]")
# print(translate(text, "CONCISE"))

# print("\n[SCHOLARLY]")
# print(translate(text, "SCHOLARLY"))





# from datasets import load_dataset

# STYLE_DATA = "D:/kaggle english akkedian/data/train_styled.csv"
# dataset = load_dataset(
#     "csv",
#     data_files=STYLE_DATA
# )

# dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
# print(dataset)
