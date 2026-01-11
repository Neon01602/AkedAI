# ======================================================
# BYTE-LEVEL T5 FROM SCRATCH (NO LoRA, PRODUCTION SAFE)
# ======================================================

import os
import gc
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    ByT5Tokenizer,
    T5Config,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ======================================================
# 0. CUDA CHECK
# ======================================================
if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA GPU REQUIRED")

device = torch.device("cuda")
torch.cuda.set_device(0)
print("GPU:", torch.cuda.get_device_name(0))

# ======================================================
# 1. PATHS
# ======================================================
DATA_PATH = r"D:\kaggle english akkedian\data\train_styled_CLEAN.csv"
OUTPUT_DIR = r"D:\kaggle english akkedian\byt5-scratch"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256

# ======================================================
# 2. LOAD DATA
# ======================================================
df = pd.read_csv(DATA_PATH)

df["source"] = df.apply(
    lambda r: f"<STYLE={r['style']}> {r['context']}",
    axis=1
)

dataset = Dataset.from_pandas(df)
print("Dataset size:", len(dataset))

# ======================================================
# 3. TOKENIZER (BYTE-LEVEL)
# ======================================================
tokenizer = ByT5Tokenizer.from_pretrained("google/byt5-small")

tokenizer.add_special_tokens({
    "additional_special_tokens": [
        "<STYLE=LITERAL>",
        "<STYLE=CONCISE>",
        "<STYLE=SCHOLARLY>"
    ]
})

# ======================================================
# 4. BYTE-T5 MODEL FROM SCRATCH
# ======================================================
config = T5Config(
    vocab_size=len(tokenizer),
    d_model=512,
    d_ff=2048,
    num_layers=6,
    num_decoder_layers=6,
    num_heads=8,
    dropout_rate=0.1,
    feed_forward_proj="gated-gelu",
    use_cache=False,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    decoder_start_token_id=tokenizer.pad_token_id,
)

model = T5ForConditionalGeneration(config)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

print("✅ Byte-T5 initialized from scratch")

# ======================================================
# 5. TOKENIZATION
# ======================================================
def preprocess(batch):
    inputs = tokenizer(
        batch["source"],
        truncation=True,
        max_length=MAX_SOURCE_LEN,
    )

    targets = tokenizer(
        batch["target"],
        truncation=True,
        max_length=MAX_TARGET_LEN,
    )

    labels = []
    for seq in targets["input_ids"]:
        labels.append([t if t != tokenizer.pad_token_id else -100 for t in seq])

    inputs["labels"] = labels
    return inputs

dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset.column_names,
)

# ======================================================
# 6. DATA COLLATOR
# ======================================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100
)

# ======================================================
# 7. TRAINING ARGUMENTS
# ======================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # effective batch = 16
    learning_rate=5e-4,              # high LR for scratch
    num_train_epochs=0.5,
    warmup_steps=200,
    logging_steps=25,
    save_strategy="epoch",
    eval_strategy="no",
    optim="adamw_torch",
    max_grad_norm=1.0,
    fp16=False,                      # RTX 3050 safer in fp32
    report_to="none",
)

# ======================================================
# 8. TRAINER
# ======================================================
gc.collect()
torch.cuda.empty_cache()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# ======================================================
# 9. TRAIN
# ======================================================
print("🚀 Training Byte-T5 from scratch")
trainer.train()

# ======================================================
# 10. SAVE (REAL MODEL)
# ======================================================
print("💾 Saving model...")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Model saved to:", OUTPUT_DIR)






# ======================================================
# BYTE-T5 INFERENCE (GPU)
# ======================================================

import torch
from transformers import T5ForConditionalGeneration, ByT5Tokenizer

MODEL_DIR = r"D:/kaggle english akkedian/byt5-scratch"   # ← YOUR REAL MODEL

device = torch.device("cuda")

# Load tokenizer
tokenizer = ByT5Tokenizer.from_pretrained(MODEL_DIR)

# Load model
model = T5ForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    local_files_only=True
).to(device)

model.eval()

print("✅ Byte-T5 loaded successfully")

# ======================================================
# Generation
# ======================================================

def generate_translation(context, style, max_new_tokens=128):
    prompt = f"<STYLE={style}> {context}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            do_sample=False
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ======================================================
# Test
# ======================================================

TEST_CONTEXTS = [
    "um-ma a-na a-ḫi-ia qí-bí-ma",
    "um-ma PN a-na PN",
    "qí-bí-ma",
]

STYLES = ["LITERAL", "CONCISE", "SCHOLARLY"]

for ctx in TEST_CONTEXTS:
    print("\n" + "="*70)
    print("AKKADIAN:", ctx)

    for style in STYLES:
        out = generate_translation(ctx, style)
        print(f"\n[{style}]")
        print(out)

print("\n✅ DONE")

