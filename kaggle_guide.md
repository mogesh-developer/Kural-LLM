# Kaggle-la Tamil Mini-LLM Train Panradhu Epdi? (Step-by-Step Tanglish Guide)

Kaggle-la **NVIDIA T4 (16GB VRAM)** GPU free-ah kedaikkum (Varathukku **30 Hours** free quota). Laptop-ah moodittu (close pannittu) kooda background-la train aaga vidalaam!

---

## STEP 1: Namma Project Files-ah Zip Pannunga

Kaggle-ku namma code, data & tokenizer matum podhum (checkpoints upload panna theva illa, fresh v4 train panna porom).

Unga laptop PowerShell terminal-la indha command run pannunga:

```powershell
Compress-Archive -Path "src", "data/processed", "tokenizer.json" -DestinationPath "tamil_llm_project.zip" -Force
```

Ippo `tamil_llm_project.zip` file create aayidum.

---

## STEP 2: Kaggle Account Setup & Phone Verification

1. [kaggle.com](https://www.kaggle.com)-ku ponga.
2. Sign in pannunga.
3. **Mukkiyamana Step (GPU unlock panna):**
   - Top right-la unga Profile icon click panni **Settings** ponga.
   - **Phone Verification** nu oru option irukkum. Unga mobile number kuduthu OTP verify pannunga. (Idhu pannalaana free GPU on aagadhu).

---

## STEP 3: Zip File-ah Kaggle Dataset-ah Upload Pannunga

1. Kaggle left side menu-la **"Datasets"** click pannunga.
2. Top right-la **"+ New Dataset"** button click pannunga.
3. Title-ku: `tamil-llm-code` nu type pannunga.
4. Namma create panna `tamil_llm_project.zip` file-ah drag & drop pannunga.
5. Bottom right-la **"Create"** click pannunga. (Upload aagi complete aagum).

---

## STEP 4: Pudhu Kaggle Notebook Create Pannunga

1. Kaggle left side menu-la **"Code"** or **"Models"** pakkathula irukkura **"+ New Notebook"** click pannunga.
2. Right side panel-la (Notebook settings) konjam configure pannanum:
   - **Accelerator:** Click panni **GPU T4 x1** select pannunga.
   - **Internet:** **On** (Toggle switch on-la irukkanum).
   - **Persistence:** **Variables and Files** (optional, recommended).
3. Right side panel-la **"Input"** section-la **"+ Add Input"** click pannunga:
   - **Your Datasets** tab-la poi namma upload panna `tamil-llm-code`-ah add pannunga.

---

## STEP 5: Notebook-la Run Panna Vendiya Code Cells

Notebook-la irukkura cell-la keezhe irukkura code-ah paste panni **Shift + Enter** kuduthu run pannunga:

### Cell 1: Setup Workspace & Extract Files
```python
import os
import shutil
import zipfile

# Working directory-ku switch panrom
os.chdir('/kaggle/working')

# Dataset zip-ah extract panrom
zip_path = '/kaggle/input/tamil-llm-code/tamil_llm_project.zip'
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('/kaggle/working')
    print("Files extracted successfully!")
else:
    # Oru vela folder-ah upload aagirundha direct copy
    shutil.copytree('/kaggle/input/tamil-llm-code', '/kaggle/working', dirs_exist_ok=True)
    print("Copied files directly!")

!ls -la
```

---

### Cell 2: Training Run Panradhu (v4 Architecture)
```python
# GPU check
!nvidia-smi

# Train run panrom
!python src/train.py
```

---

## STEP 6: Laptop-ah Moodittu (Background-la) Run Panna "Save Version" Trick!

Kaggle-oda biggest advantage idhudhan. Browser-ah close pannalum train aagite irukkum:

1. Notebook top-right corner-la **"Save Version"** nu oru blue button irukkum.
2. Adha click pannunga:
   - Version type: **"Save & Run All (Commit)"** select pannunga.
3. **Save** click pannunga!
4. Ippo oru window open aagi background execution start aayidum.
5. **Neenga unga laptop-ah turn off pannalaam, internet disconnect pannalaam!**
   - Kaggle cloud server-la background-la 10 epochs train aagi mudinjidum (around 15-20 minutes).

---

## STEP 7: Checkpoint Download Panradhu

Training mudinjadhukku apparam:
1. Andha Notebook page-ku ponga.
2. Right side-la **"Output"** section-la `checkpoints_v4/best.pt` file irukkum.
3. Pakkathula irukkura **3 dots (...)** click panni **Download** pannikalam!
