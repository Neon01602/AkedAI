# import torch
# import pandas as pd
# from transformers import AutoTokenizer, T5ForConditionalGeneration
# from tqdm import tqdm

# # =====================================
# # 1. Paths
# # =====================================
# MODEL_DIR = "D:/kaggle english akkedian/byt5-akkadian"
# TEST_PATH = "D:/kaggle english akkedian/data/test.csv"
# OUTPUT_PATH = "D:/kaggle english akkedian/data/test_predictions.csv"

# # =====================================
# # 2. Device
# # =====================================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)

# # =====================================
# # 3. Load Model
# # =====================================
# tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
# model.eval()

# # =====================================
# # 4. Translation Function
# # =====================================
# def translate(text):
#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         max_length=256
#     ).to(device)

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_length=128,
#             num_beams=4,
#             early_stopping=True
#         )

#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # =====================================
# # 5. Load Test Data
# # =====================================
# df = pd.read_csv(TEST_PATH)

# # remove broken rows
# df = df.dropna(subset=["transliteration"])
# df["transliteration"] = df["transliteration"].astype(str)

# print("Test samples:", len(df))

# # =====================================
# # 6. Run Inference
# # =====================================
# predictions = []

# for text in tqdm(df["transliteration"], desc="Translating"):
#     pred = translate(text)
#     predictions.append(pred)

# df["prediction"] = predictions

# # =====================================
# # 7. Save Results
# # =====================================
# df.to_csv(OUTPUT_PATH, index=False)
# print("Saved predictions to:", OUTPUT_PATH)

# # =====================================
# # 8. Print Sample Outputs
# # =====================================
# print("\n--- SAMPLE OUTPUTS ---")
# for i in range(min(5, len(df))):
#     print("\nINPUT :", df.iloc[i]["transliteration"])
#     if "translation" in df.columns:
#         print("GOLD  :", df.iloc[i]["translation"])
#     print("PRED  :", df.iloc[i]["prediction"])




# # # import pandas as pd
# # # import re

# # # # ===== PATHS =====
# # # DICT_PATH = "D:\kaggle english akkedian\data\eBL_Dictionary.csv"
# # # OUTPUT_PATH = "D:\kaggle english akkedian\data\dictionary_pairs.csv"

# # # # ===== LOAD =====
# # # df = pd.read_csv(DICT_PATH)

# # # print("Total dictionary entries:", len(df))

# # # # ===== CLEANING RULES =====
# # # def is_good_definition(defn):
# # #     if not isinstance(defn, str):
# # #         return False

# # #     # Too long = scholarly explanation
# # #     if len(defn.split()) > 8:
# # #         return False

# # #     # Remove grammatical noise
# # #     if "(" in defn or ")" in defn:
# # #         return False

# # #     if ";" in defn or ":" in defn:
# # #         return False

# # #     # Must contain letters
# # #     if not re.search(r"[a-zA-Z]", defn):
# # #         return False

# # #     return True


# # # pairs = []

# # # for _, row in df.iterrows():
# # #     word = str(row["word"]).strip()
# # #     definition = str(row["definition"]).strip()

# # #     if not word or not definition:
# # #         continue

# # #     if not is_good_definition(definition):
# # #         continue

# # #     # Normalize spacing
# # #     word = re.sub(r"\s+", " ", word)
# # #     definition = re.sub(r"\s+", " ", definition)

# # #     pairs.append({
# # #         "transliteration": word,
# # #         "translation": definition
# # #     })

# # # pairs_df = pd.DataFrame(pairs)

# # # print("Kept dictionary pairs:", len(pairs_df))

# # # # ===== SAVE =====
# # # pairs_df.to_csv(OUTPUT_PATH, index=False)
# # # print("Saved to:", OUTPUT_PATH)


# # import pandas as pd
# # from sklearn.utils import shuffle

# # # ===== PATHS =====
# # TRAIN_PATH = "D:/kaggle english akkedian/data/train.csv"
# # DICT_PATH = "D:/kaggle english akkedian/data/dictionary_pairs.csv"
# # OUTPUT_PATH = "D:/kaggle english akkedian/data/train_merged.csv"

# # # ===== LOAD =====
# # train_df = pd.read_csv(TRAIN_PATH)
# # dict_df = pd.read_csv(DICT_PATH)

# # print("Original train size:", len(train_df))
# # print("Dictionary pairs:", len(dict_df))

# # # ===== CLEAN TRAIN DATA =====
# # train_df = train_df.dropna(subset=["transliteration", "translation"])
# # train_df = train_df[train_df["translation"].str.strip() != ""]

# # # ===== LIMIT DICTIONARY TO 20% =====
# # max_dict = int(len(train_df) * 0.25)   # safe upper bound
# # dict_df = dict_df.sample(n=min(len(dict_df), max_dict), random_state=42)

# # print("Using dictionary pairs:", len(dict_df))

# # # ===== MERGE =====
# # merged_df = pd.concat([train_df, dict_df], ignore_index=True)

# # # ===== SHUFFLE =====
# # merged_df = shuffle(merged_df, random_state=42).reset_index(drop=True)

# # print("Final merged dataset size:", len(merged_df))

# # # ===== SAVE =====
# # merged_df.to_csv(OUTPUT_PATH, index=False)
# # print("Saved merged dataset to:", OUTPUT_PATH)


# =====================================
# 1. Imports
# =====================================
import os
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)

# =====================================
# 2. Config
# =====================================
MODEL_NAME = "google/byt5-small"
DATA_PATH = "D:/kaggle english akkedian/data/train_merged.csv"
OUTPUT_DIR = "D:/kaggle english akkedian/byt5-akkadian"

MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 128
BATCH_SIZE = 8
EPOCHS = 8
LR = 2e-4

TASK_PREFIX = "translate akkadian to english: "

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =====================================
# 3. Load + Clean Dataset
# =====================================
dataset = load_dataset("csv", data_files={"data": DATA_PATH})["data"]

print("Original rows:", len(dataset))

dataset = dataset.filter(
    lambda x: (
        x["transliteration"]
        and x["translation"]
        and str(x["transliteration"]).strip()
        and str(x["translation"]).strip()
    )
)

# 🔥 Kill repetition-heavy rows
def clean_repetition(ex):
    return ex["translation"].lower().count("say to") < 3

dataset = dataset.filter(clean_repetition)

print("After cleaning:", len(dataset))

dataset = dataset.train_test_split(test_size=0.1, seed=42)

# =====================================
# 4. Load Model + Tokenizer
# =====================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

# =====================================
# 5. Preprocessing
# =====================================
def preprocess(batch):
    inputs = tokenizer(
        [TASK_PREFIX + x for x in batch["transliteration"]],
        max_length=MAX_SOURCE_LEN,
        truncation=True,
        padding="max_length",
    )

    targets = tokenizer(
        batch["translation"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding="max_length",
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
    remove_columns=dataset["train"].column_names,
)

# =====================================
# 6. Data Collator
# =====================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
)

# =====================================
# 7. Training Arguments
# =====================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    fp16=False,
    max_grad_norm=1.0,
    save_total_limit=2,
    report_to="none",
    load_best_model_at_end=True,
)

# =====================================
# 8. Trainer
# =====================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
)

# =====================================
# 9. Train
# =====================================
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training complete")

# =====================================
# 10. Evaluation (BLEU + chrF)
# =====================================
bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")

def evaluate_model():
    preds, refs = [], []

    for ex in dataset["test"]:
        text = TASK_PREFIX + tokenizer.decode(
            ex["input_ids"], skip_special_tokens=True
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SOURCE_LEN,
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=5,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
                length_penalty=1.0,
                early_stopping=True,
            )

        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        ref = tokenizer.decode(
            [t for t in ex["labels"] if t != -100],
            skip_special_tokens=True,
        )

        preds.append(pred)
        refs.append([ref])

    print("BLEU:", bleu.compute(predictions=preds, references=refs))
    print("chrF:", chrf.compute(predictions=preds, references=refs))

evaluate_model()

# =====================================
# 11. Inference Function (PRODUCTION)
# =====================================
def translate(text: str):
    model.eval()
    text = TASK_PREFIX + text.strip()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SOURCE_LEN,
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=6,
            no_repeat_ngram_size=4,
            repetition_penalty=1.35,
            length_penalty=0.9,
            early_stopping=True,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# =====================================
# 12. Example Test
# =====================================
print("\n--- SAMPLE TEST ---")
print(translate("um-ma kà-ru-um kà-ni-ia-ma a-na aa-qí-il"))
