# Routing-Oracle Experiment — GPU run kit

驗證論文《How Much of the Routing Gap Is Real?》的可觀察推論：在真實開源模型池上，
單抽樣 oracle 與「可重現上限」之間的差距，有多少是**隨機波動造成、無法以選模型消除**
的部分（noise share）。**理論已用數學證明；這套程式只負責量真實數字、檢假設、做可證偽測試。**

> 分工：所有前置作業已備妥。你只需在 GPU VM 上 `git clone` → 跑一行 → 把 `results/` 傳回分析。

---

## 0. 先決條件（GPU VM）
- NVIDIA GPU（建議 2× RTX 4090 / 48GB；單卡亦可，模型逐一載入）+ CUDA。
- Python 3.10+。
- 安裝（先裝 vLLM，它會帶對應的 torch）：
  ```bash
  pip install -r requirements.txt
  # 若要抓 gated 模型（Llama/Gemma），先： huggingface-cli login
  ```

## 1. 先做無 GPU 煙霧測試（確認管線本身沒問題，30 秒）
```bash
bash run_all.sh --smoke
```
會用「模擬正確率」跑完整條分析（估計式→gate→best-of-K→分解→出圖），
輸出 `results/mvp/decomposition.json`。看到 `gates.all_pass=true`、`noise_share_ci` 有數字即代表程式 OK。

## 2. 正式跑（需 GPU）
```bash
bash run_all.sh gsm8k 200        # benchmark 與題數可改；先小規模 200 題試跑
# 完整版： bash run_all.sh gsm8k 1000     （或 mmlu）
```
四個階段（也可分步跑，見 `run_all.sh`）：
| 階段 | 程式 | 做什麼 | 輸出 |
|---|---|---|---|
| 01 | `scripts/01_make_subset.py` | 從 HuggingFace 抓 benchmark（題目+gold） | `data/subset.json` |
| 02 | `scripts/02_generate.py` | 8 模型各對每題抽 **k=20 次（T=0.2、seed 對齊 A8）** | `data/raw/gen_m*.jsonl` |
| 03 | `scripts/03_score.py` | exact-match 評分 → 對齊的 `b[i,m,j]` 張量 | `data/processed/correctness_kxN.npz` |
| 04 | `scripts/04_oracles_decompose.py` | 修正後估計式＋兩道 gate＋best-of-K＋分解＋CI＋出圖 | `results/data/decomposition.json` (+png) |

## 3. 把結果傳回（給我分析）
把整個 **`results/`** 目錄傳回即可（json 很小）。我會：核對 gate 是否通過、把
noise share / 分解 / best-of-K / 分層數字寫進論文 §Results、並判斷是否需調整 k 或作用域。

---

## 設定（`configs/pool_open8.yaml`）
- **8 模型開源池**：gemma-4-12B/26B-MoE、Qwen2.5-14B/32B、Phi-4、Mistral-7B、Llama-3.2-11B、DeepSeek-R1-Distill-7B。
  **正式跑前先確認每個 repo_id 在 HF 仍存在**（檔尾有清單）；404 就換同家族同規模的近親、保持多樣性。
- **抽樣**：T=0.2（**不可調高，調高= 製造我們要量的雜訊**）、top_p=1.0、k=20（薄支撐層建議 k≥30）、seed 對齊（A8）。
- benchmark 預設 `gsm8k`（數值 exact-match，最乾淨）；`mmlu`（選擇題）需要時用。

## 重要前提（誠實說明）
- **gate 可能不過**：若 A1 獨立性檢驗失敗（疑似 provider caching），04 會照協定「不報量級」——這是設計，不是 bug。
- **best-of-K 可能失敗** → 代表 A1 被破（非定理錯），結論需改寫。
- k=10 時保守下界會是 0（偏保守）；正式跑用 **k≥20，薄支撐 k≥30**。
- 本套用「直接抓標準 benchmark（gold 乾淨）+ 自家 8 模型新抽樣」。論文主池 LLMRouterBench（33 模型）目前 HF 無公開；若你拿到，把 `src/data.load_raw_correctness` 接上即可。

## 結構
```
src/        oracles, decompose, stats, generate(vLLM/API/seed-aligned), score, simulate, data
scripts/    01 subset · 02 generate · 03 score · 04 decompose
configs/    pool_open8.yaml  (8-model pool + 抽樣/gate/估計式設定)
tests/      test_oracles.py  (13 個估計式單元測試：python tests/test_oracles.py)
run_all.sh  一鍵管線（--smoke 為無 GPU 驗證）
```
