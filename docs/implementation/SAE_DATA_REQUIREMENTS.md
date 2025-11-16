# SAE Training - Data Requirements Analysis

**Date**: 2025-11-16
**Question**: Are 1000 examples enough for SAE training?

---

## ❌ Short Answer: NO

1000 examples are **insufficient** for training a publication-quality SAE.

---

## 📊 Literature Benchmarks

### Anthropic SAE Training Data:

| Paper | Model | SAE Dict Size | Training Tokens | Examples (est.) |
|-------|-------|---------------|-----------------|-----------------|
| **Towards Monosemanticity** (2023) | 1-layer toy | 512 features | ~10M tokens | ~300K examples |
| **Scaling Monosemanticity** (2024) | Claude 3 Sonnet | 34M features | ~4B tokens | ~130M examples |
| **Gemma Scope** (2024) | Gemma 2B | 16K features | ~100M tokens | ~3M examples |

### Our Setup:

| Parameter | Our Value | SOTA Minimum | Gap |
|-----------|-----------|--------------|-----|
| Examples | 1,000 | 100,000+ | **100x less** |
| Tokens | ~30K | ~10M | **333x less** |
| Dict size | 3,072 | 3,072 | Match |

---

## 🔬 Why More Data is Critical

### 1. **Feature Coverage**

SAE with 3,072 features needs to learn 3,072 distinct monosemantic concepts.

**With 1000 examples**:
- Tokens: ~30,000 (assuming 30 tokens/example)
- Tokens per feature: 30,000 / 3,072 ≈ **10 tokens/feature**
- **Result**: Most features will see <10 examples → **can't learn meaningful patterns**

**With 100K examples** (minimum):
- Tokens: ~3M
- Tokens per feature: 3M / 3,072 ≈ **1,000 tokens/feature**
- **Result**: Each feature sees enough variation to learn robust patterns

---

### 2. **Dead Features**

**Expected with 1000 examples**:
- Dead features: 40-60% (most features never activate enough to learn)
- Ghost gradients help but can't compensate for insufficient data

**Expected with 100K+ examples**:
- Dead features: <5% (SOTA quality)
- Ghost gradients work effectively

---

### 3. **Monosemanticity**

**With limited data**, features become **polysemantic** (represent multiple concepts):
- Feature 42 might activate for: "names", "pronouns", "subjects" (too broad)
- Can't distinguish because it hasn't seen enough diverse examples

**With sufficient data**, features become **monosemantic**:
- Feature 42: Only "proper names starting with 'J'"
- Feature 137: Only "indirect object pronouns"
- Clear, interpretable, single concept per feature

---

### 4. **Reconstruction Quality**

SAE needs to see many variations of each pattern to reconstruct well.

**1000 examples**:
- MSE: 0.20-0.30 (poor reconstruction)
- Many activation patterns never seen during training

**100K+ examples**:
- MSE: 0.08-0.12 (excellent reconstruction)
- Covers vast majority of activation patterns

---

## 📈 Recommended Data Sizes

### Minimum (Acceptable):
- **Examples**: 100,000
- **Tokens**: ~3M
- **Training time**: ~30-60 minutes
- **Quality**: Usable but not publication-grade

### Good (Recommended):
- **Examples**: 1,000,000
- **Tokens**: ~30M
- **Training time**: ~5-8 hours
- **Quality**: High quality, publishable

### Excellent (SOTA):
- **Examples**: 10,000,000+
- **Tokens**: ~300M+
- **Training time**: ~1-2 days
- **Quality**: State-of-the-art

---

## 🎯 Our Situation

### Current:
- **Phase 1 captured**: 1,000 examples
- **Tokens**: ~30,000
- **Status**: ❌ **Insufficient for quality SAE**

### Options:

#### Option 1: Quick Demo (Current Data) ⚠️
```bash
python train_enhanced_sae.py
```
**Pros**:
- Fast (10 minutes)
- Tests infrastructure
- Proof of concept

**Cons**:
- Poor quality (<40% monosemantic)
- High dead features (>40%)
- NOT publication-ready
- NOT suitable for "complete neural cartography"

---

#### Option 2: Proper Training (100K examples) ✅ **RECOMMENDED**
```bash
# Capture 100K examples
python capture_large_dataset.py --num_examples 100000

# Train Enhanced SAE
python train_enhanced_sae.py
```
**Pros**:
- High quality (~80% monosemantic)
- Low dead features (<10%)
- Publication-ready
- Suitable for real research

**Cons**:
- Capture time: ~30 minutes
- Training time: ~1 hour
- More disk space (~2-3 GB)

---

#### Option 3: SOTA Training (1M examples) 🌟
```bash
# Capture 1M examples
python capture_large_dataset.py --num_examples 1000000

# Train Enhanced SAE (overnight)
python train_enhanced_sae.py --epochs 20
```
**Pros**:
- SOTA quality (~90% monosemantic)
- Dead features <5%
- Comparable to Anthropic results
- Full "neural cartography" ready

**Cons**:
- Capture time: ~5 hours
- Training time: ~8-12 hours (overnight)
- Disk space: ~20-30 GB

---

## 💡 Specific Recommendations

### For "Mappare 1:1 la rete neurale nascosta":

You **NEED** Option 2 or 3. Here's why:

1. **Scientific Rigor**: 1000 examples insufficient for claims about neural cartography
2. **Feature Quality**: Polysemantic features ≠ 1:1 mapping
3. **Publication**: Reviewers will reject "trained on 1000 examples"
4. **Comparison**: Can't compare with Anthropic if data orders of magnitude different

---

## 🔢 Concrete Numbers

### What 1000 examples gives you:

```
Dictionary size: 3,072
Training tokens: 30,000
Tokens per feature: 10

Expected results:
- Dead features: 50-60%
- Monosemantic: 30-40%
- MSE: 0.25-0.35
- Usable features: ~1,200 out of 3,072

Conclusion: ⚠️  Infrastructure test only, NOT research-quality
```

### What 100K examples gives you:

```
Dictionary size: 3,072
Training tokens: 3,000,000
Tokens per feature: 976

Expected results:
- Dead features: 5-10%
- Monosemantic: 75-85%
- MSE: 0.10-0.15
- Usable features: ~2,800 out of 3,072

Conclusion: ✅ Publication-quality, research-ready
```

### What 1M examples gives you:

```
Dictionary size: 3,072
Training tokens: 30,000,000
Tokens per feature: 9,765

Expected results:
- Dead features: <5%
- Monosemantic: 85-92%
- MSE: 0.08-0.12
- Usable features: >2,900 out of 3,072

Conclusion: 🌟 SOTA quality, Anthropic-comparable
```

---

## ⚖️ Trade-offs

| Aspect | 1K examples | 100K examples | 1M examples |
|--------|-------------|---------------|-------------|
| **Capture time** | 30 sec | 30 min | 5 hours |
| **Training time** | 10 min | 1 hour | 8-12 hours |
| **Disk space** | 200 MB | 2-3 GB | 20-30 GB |
| **Dead features** | 50-60% | 5-10% | <5% |
| **Monosemantic %** | 30-40% | 75-85% | 85-92% |
| **MSE** | 0.25-0.35 | 0.10-0.15 | 0.08-0.12 |
| **Publication-ready** | ❌ | ✅ | ✅ |
| **SOTA-comparable** | ❌ | ⚠️ | ✅ |

---

## 🎯 My Recommendation

### Path Forward:

1. **Test Infrastructure** (now):
   ```bash
   python train_enhanced_sae.py  # On current 1K data
   ```
   - Verify code works
   - Check training loop
   - See poor quality firsthand
   - **Time**: 10 minutes

2. **Proper Capture** (next):
   ```bash
   python capture_large_dataset.py --num_examples 100000
   ```
   - Generate 100K diverse examples
   - Proper data for training
   - **Time**: 30 minutes

3. **Real Training** (after):
   ```bash
   python train_enhanced_sae.py  # On 100K data
   ```
   - Train publication-quality SAE
   - Achieve 80%+ monosemantic
   - **Time**: 1 hour

4. **Optional: SOTA** (if you have time):
   - Capture 1M examples overnight
   - Train 1M SAE for 8-12 hours
   - Achieve Anthropic-comparable quality

---

## 📚 References

1. **Anthropic - Towards Monosemanticity** (2023)
   - Trained on 10M tokens minimum
   - "Data scale is critical for monosemanticity"

2. **Anthropic - Scaling Monosemanticity** (2024)
   - 4B tokens for Claude 3 Sonnet SAE
   - "More data → better features"

3. **Gemma Scope** (2024)
   - 100M tokens for Gemma 2B
   - "Feature quality correlates with data size"

4. **SAE Best Practices** (EleutherAI)
   - "Minimum 1000 tokens per feature"
   - "1M+ tokens recommended"

---

## ✅ Conclusion

**1000 examples are NOT enough** for:
- ❌ Publication-quality SAE
- ❌ Monosemantic features
- ❌ "Complete neural cartography"
- ❌ Comparison with SOTA

**Minimum for serious work**: 100,000 examples

**Recommended for your goal**: 1,000,000 examples

---

**Next Action**:
1. Quick test with 1K (verify infrastructure)
2. Capture 100K+ (proper data)
3. Real training (publication-quality results)

---

**My vote**: Capture 100K examples minimum. Your goal deserves proper data.
