# 🏺 **AkedAI**
*Byte-Level Akkadian → English Translation Model*

---

## 📌 **Overview**

**AkedAI** is a **low-resource historical NLP system** designed to translate **Akkadian transliterations** into **English** using a **byte-level Transformer (ByT5)**.  
It focuses on **practical robustness**, **linguistic sanity**, and **real evaluation** rather than chasing inflated metrics.

> Ancient languages are broken.  
> AkedAI decided not to be.

---

## 🧠 **Core Concepts**

- **Byte-level modeling (ByT5)**  
  No tokenization failures, no OOV nightmares, works directly on raw text.

- **Controlled dictionary fusion**  
  Dictionaries teach **lexicon**, not **syntax**.  
  Only **~20–25%** dictionary pairs were merged to prevent sentence-level corruption.

- **Stability-first training**  
  FP16 caused NaNs → **disabled**.  
  Slower, calmer, *actually converges*.

---

## 🗂️ **Dataset Engineering (Hard-Learned Lessons)**

| Step | Discovery |
|----|----|
| Raw corpus | Highly formulaic but inconsistent |
| Dictionary merge | Boosts vocabulary recall |
| Over-merging | 💥 Grammar collapse |
| Partial injection | ✅ Best balance |
| Cleaning | Non-negotiable |

> **Lexicon ≠ Language**  
> The model learned this the hard way.

---

## ⚙️ **Training Configuration**

| Parameter | Value |
|--------|------|
| Base Model | `google/byt5-small` |
| Input Length | **256** |
| Output Length | **128** |
| Batch Size | **8** |
| Epochs | **8** |
| Optimizer | **AdamW** |
| Mixed Precision | **Disabled** |
| GPU | **RTX 3050** |

---

## 📊 **Evaluation Metrics**

| Metric | Score | Meaning |
|------|------|--------|
| **BLEU** | ~0.13 | Expectedly low for ancient text |
| **chrF** | **~36.9** ⭐ | Morphology-friendly |
| Hallucination | Low | Names + formulas preserved |

> BLEU cried.  
> chrF smiled.

---

## 🧪 **Sample Translation**

**Input**
```text
um-ma kà-ru-um kà-ni-ia-ma a-na aa-qí-il
```

## 🧪 Output

```Text
**From the Kanesh colony to Aqil.**
```

## 🏆 **Key Achievements**

- **Stable byte-level Akkadian translation**
- **Correct handling of names & places**
- **Low hallucination rate**
- **Dictionary-aware but grammar-safe**
- **Practical evaluation mindset**

---

## ⚠️ **Limitations & Failures**

- **No tablet-level context**
- **No role labeling (sender / recipient)**
- **BLEU is emotionally misleading**
- **Historical ambiguity remains unresolved**

---

## 🔧 **Future Improvements**

- **Hierarchical document modeling**
- **Formula-aware decoding**
- **Multi-reference evaluation**
- **Entity normalization**
- **Domain-specific fine-tuning**

---

## 😄 **Final Note**

*AkedAI doesn’t fully understand Akkadian —  
but it finally reads it without panicking.*

**That’s Day One of building massive AI systems.**

---

## 📜 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.


---
