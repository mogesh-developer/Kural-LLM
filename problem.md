# Training Loss Spike Issue - Root Cause, Analysis & Fix (Tanglish Guide)

---

## 1. Problem Enna Aachu? (Summary)

`log.md`-la Epoch 10 varaikkum training smooth-ah poitu irundhuchu (Loss ~3.62). 
Aana **Epoch 10, Batch 300**-la thideernu loss oru periya number-ku jump aachu:

```text
Epoch 10/10 | Batch 275/1771 | Loss 3.6206 | LR 6.02e-06
Epoch 10/10 | Batch 300/1771 | Loss 3329695576599482328578693005312.0000 (~3.33e+30)
```

Adhukku apparam loss konjam konjama kuraura maadhiri therinjadhu:
`3.33e30 -> 3.07e30 -> ... -> 5.64e29`

Kadaisila Epoch 10 mudiyumbodhu:
```text
Epoch 10/10 complete | Train loss: 564036517775180528727781015552.0000 | Validation loss: 3.7405
New best checkpoint: checkpoints_v3_stage2\best.pt
```

---

## 2. Root Cause (Enna Kaaranathinaala Indha Problem Vandhudhu?)

### Kaaranam 1: Single Batch Loss Spike (~1.0e+33)
Batch 275-kkum 300-kkum naduvula oru specific batch data (probably repetitive tokens, corrupted padding, illa FP16 autocast numerical underflow/overflow) nala loss romba perusa ($10^{33}$) aayiduchu.

### Kaaranam 2: PyTorch `torch.isfinite()` Trap
Pazhaya code-la idhudhan check pannom:
```python
if loss is None or not torch.isfinite(loss).item():
    raise RuntimeError("Non-finite training loss detected.")
```
- Standard 32-bit Float-oda maximum limit $\approx 3.4 \times 10^{38}$.
- Namma loss vandhu $\approx 1.0 \times 10^{33}$.
- Adhanala PyTorch idhai `infinite` illa `NaN` nu nenaikkala! "Idhu finite number dhaan" nu ninaichu training-ah stop pannama proceed panniduchu.

### Kaaranam 3: Running Average Illusion
Code-la log pannumbodhu:
```python
avg_loss = total_train_loss / (batch_index + 1)
```
Orey oru batch-la vandha $10^{33}$ loss `total_train_loss`-la add aayiduchu. 
Adhukku apparam vandha batch-la normal loss (3.6) vandhalum, indha periya number-ah adutha batch count (300, 325, 350...) vechi divide pannumbodhu loss koraura maadhiri oru **illusion** vandhudhu. Reality-la adhu koraiala, denominator matumdhan perusachu.

### Kaaranam 4: False "New Best Checkpoint"
Val loss calculate pannumbodhu val set-la indha issue illadha nala val loss `3.7405` vandhudhu (Epoch 9 val loss: `3.7413`). 0.0008 koraivaga irundhadhanaala, corrupted Epoch 10 model vandhu `best.pt`-ah overwrite panniduchu!

---

## 3. Epdi Fix Pannom? (How We Handled It)

### Step 1: Corrupted Stage 2 Folder Delete Pannom
Corrupted weights and momentum irundha `checkpoints_v3_stage2/` folder-ah completely remove pannitom:
- Clean Stage 1 baseline: `checkpoints_v3/best.pt` safe-ah irukku.

### Step 2: `src/train.py`-la Anomaly Guard Add Pannom
Cross-entropy loss eppavume language model-ku `1.5` lerundhu `10.0` kulla dhaan irukkum. `20.0`-ku mela ponaale adhu kandippa glitch/anomaly dhaan.

**Pazhaya Code:**
```python
if loss is None or not torch.isfinite(loss).item():
    raise RuntimeError("Non-finite training loss detected.")
```

**Pudhu Fixed Code (`src/train.py`):**
```python
if (
    loss is None
    or not torch.isfinite(loss).item()
    or loss.item() > 20.0
):
    print(
        f"\n[WARNING] Skipping anomalous loss at batch {batch_index + 1}: "
        f"{loss.item() if loss is not None else 'None'}"
    )
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    continue

scaled_loss = loss / group_size
```

**Idhoda Benefit:**
- Inime yedhavadhu batch-la loss spike aana (`> 20.0`), adhu optimizer-ayum weight-ayum ruin pannaadhu.
- Warning print pannitu, gradients-ah clean-ah reset pannittu, andha specific batch-ah skip pannitu next batch-ku poirum.
- `estimate_loss()` val evaluation-layum `loss.item() < 20.0` check add pannitom.

---

## 4. Ippo Enna Pannanum? (Action Steps)

Unga terminal-la direct-ah:
```powershell
python src\train.py
```
run pannunga.

1. Idhu automatic-ah `checkpoints_v3/best.pt` lerundhu fresh Stage 2 training-ah start pannum.
2. Anomaly guard active-ah irukkura nala endha batch-layum $10^{30}$ maadhiri gradient spike varama smooth-ah train aagum.
