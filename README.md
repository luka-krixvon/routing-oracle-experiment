# routing-oracle-experiment

量測論文《How Much of the Routing Gap Is Real?》的可觀察推論：真實開源模型池上，單抽樣
oracle 與「可重現上限」的差距，有多少是**隨機波動造成、無法以選模型消除**的部分（noise share）。
**理論已用數學證明；本程式只量真實數字、檢假設、做可證偽測試。**

> **設計給 55GB 小硬碟 + 2×RTX 4090**：一次只跑一個模型，跑完即**清掉該模型權重與 HF cache、釋放 GPU**，再跑下一個。峰值硬碟用量 ≈ 最大的單一模型（約 29GB）< 55GB。

---

## 一、先做 preflight（只檢查、不下載、不執行）
```bash
pip install -r requirements.txt          # vLLM 會帶對應 torch
bash preflight.sh
```
會印：`df -h`、`nvidia-smi`、HF cache 大小、專案大小、torch/vLLM 版本、**逐一檢查 `configs/models.txt` 每個模型能否取得 / 是否 gated**，最後給 GO / NO-GO。
- 看到 **GATED** 的模型（gemma-2、Llama-3.1）：先 `huggingface-cli login`，並到該模型頁面按「同意 license」。

## 二、正式跑（disk-safe sequential）
```bash
bash run_sequential.sh                    # 預設 gsm8k × 200 題、k=20
# 完整版： BENCH=gsm8k N=1000 K=30 bash run_sequential.sh
```
每個模型自動：① 檢查硬碟剩餘（不足就**安全停止**，不會塞爆）→ ② 在**獨立子程序**下載+生成（T=0.2、seed 對齊 A8、k 次）+ exact-match 評分 → ③ 存該模型的 0/1 欄位 `data/per_model/m*.npz`（很小）→ ④ **用 `huggingface_hub` 安全 API 清掉該模型權重**（不是 `rm -rf`，碰不到家目錄/程式碼）→ ⑤ `del model; torch.cuda.empty_cache(); ipc_collect()` + 子程序結束 → GPU 全釋放 → 下一個。
全部跑完自動 `combine`（拼成 (N,M,k) 張量）→ `04`（修正估計式＋兩道 gate＋best-of-K＋分解）。**可中斷續跑**（已完成的模型會跳過）。

## 三、無 GPU 煙霧測試（先確認管線本身 OK，30 秒）
```bash
bash run_all.sh --smoke
```

## 環境偵測（自動執行，產生論文可用環境報告）
`run_sequential.sh` 的**開頭與結尾**會自動跑 `scripts/detect_environment.py`，把硬體 / NVIDIA
（driver、CUDA toolkit+runtime、cuDNN、NCCL）/ PyTorch+CUDA / 套件版本 / Git 狀態寫到
`reports/environment/`，三種格式：`environment_report.json`（原始、重現用）、
`environment_report.md`（可讀）、`paper_environment_summary.md`（**論文可直接貼的英文段落**，版本自動填入）。
也可單獨跑：
```bash
python scripts/detect_environment.py            # 加 --anonymize 連 hostname / 使用者名都遮蔽
```
敏感資訊（HF token / key / 家目錄路徑）一律遮蔽，可安全 commit。

## 四、把結果傳回（給我分析）
傳回（全部很小）：`results/`、`logs/`、`reports/`、`data/per_model/*.npz`。**不要傳模型權重 / HF cache**（`.gitignore` 已排除，也別手動塞）。

---

## 模型池（`configs/models.txt`，可編輯；已 preflight 驗證）
| 模型 | 量化 | 卡 | 約 VRAM | 取得 |
|---|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | fp16 | 1 | ~15GB | open |
| DeepSeek-R1-Distill-Qwen-7B | fp16 | 1 | ~15GB | open（長 CoT，k 大時較慢）|
| Qwen2.5-7B-Instruct-AWQ | awq | 1 | ~6GB | open |
| Qwen2.5-14B-Instruct-AWQ | awq | 1 | ~9GB | open |
| Qwen2.5-32B-Instruct-AWQ | awq | 1 | ~20GB | open |
| microsoft/phi-4 | fp16 | **2** | ~29GB（雙卡）| open；想用單卡改社群 4-bit |
| google/gemma-2-9b-it | fp16 | 1 | ~18GB | **gated** |
| meta-llama/Llama-3.1-8B-Instruct | fp16 | 1 | ~16GB | **gated** |

> preflight 已剔除原設定中**不可用**的三個：gemma-4-12B / gemma-4-26B-A4B（多模態 image-text-to-text，非純文字）、Llama-3.2-11B-Instruct（不存在，3.2@11B 只有 Vision 版）。

## 重要前提（誠實）
- **gate 可能不過**（疑 provider caching → A1 失敗）：04 依協定不報量級——這是設計。
- **best-of-K 失敗** → A1 被破（非定理錯）。
- k 太小保守下界會是 0：薄支撐層建議 **k≥30**。
- 本套用標準 benchmark（GSM8K/MMLU，gold 乾淨）+ 自家模型新抽樣；論文主池 LLMRouterBench 目前 HF 無公開，拿到再接 `src/data.load_raw_correctness`。

## 結構
```
preflight.sh         先檢查（df/nvidia-smi/cache/模型可達性）
run_sequential.sh    disk-safe 逐模型：下載→跑→存→清→釋放→下一個→合併→分解
run_all.sh           簡單鏈 / --smoke 無 GPU 驗證
configs/models.txt   模型池（可編輯）   configs/pool_open8.yaml 抽樣/gate/估計式參考
scripts/  01 subset · 02 generate · 03 score · 04 decompose · run_one_model · cleanup_hf · combine
src/      oracles · decompose · stats · generate(vLLM,seed-aligned) · score · simulate · data
tests/    test_oracles.py  (python tests/test_oracles.py)
```
