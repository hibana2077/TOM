可以。我先直接把**你要對齊的目標**講清楚：

## 快速使用（已完成腳本）

你可以直接跑根目錄的 `vis_alignment.py` 來畫 sDTW / uDTW 的 soft alignment matrix，
gamma 會一次產生這 5 組：`1.0, 0.1, 0.01, 0.001, 0.0001`。

執行：

```bash
python vis_alignment.py
```

輸出檔案：

- `soft_alignment_matrices.png`

圖的排列：

- 第 1 列：sDTW（5 個 gamma）
- 第 2 列：uDTW（5 個 gamma）

**論文 Fig. 2 的 (a)–(d) 畫的不是 raw distance matrix，也不是 attention map，也不是 uncertainty matrix 本身；它們畫的是「soft path / soft alignment path matrix」**。而 **(e)** 則是把 **(c) 的 path 圖先二值化，再乘上 uncertainty matrix (\Sigma)**，用來顯示「路徑上哪一段是不確定的」。論文 caption 也明寫了：

* (a)–(d) 是 sDTW / uDTW 的 path
* (e) 是把 **binarized plot (c)** 乘上 **(\Sigma)**，而且白色代表高 uncertainty 

下面我重寫一份**可直接照著做的 vis 實作引導**，而且我會把**每一步要取哪個 matrix**講清楚。

---

# uDTW / sDTW 視覺化實作引導（對齊論文 Fig. 2 的版本）

## 0. 先定義你現在要畫的是哪一種圖

你這次至少要分成 **5 張圖**，對應 Fig. 2：

1. **(a) sDTW, (\gamma=0.01)**
   畫的是：`A_sdtw_001`

2. **(b) sDTW, (\gamma=0.1)**
   畫的是：`A_sdtw_01`

3. **(c) uDTW, (\gamma=0.01)**
   畫的是：`A_udtw_001`

4. **(d) uDTW, (\gamma=0.1)**
   畫的是：`A_udtw_01`

5. **(e) uDTW uncertainty-on-path**
   畫的是：`A_udtw_001_bin ⊙ Σ`

**注意：第 5 張一定是乘 (\Sigma)，不是乘 (\Sigma^\dagger)，也不是乘 (D)。** 這點是論文 caption 直接寫的。

---

## 1. 你應該先產生哪些基礎張量

如果你現在是 image patch 版（ViT patch sequence 遷移），那建議沿用 guide 的設定：

* 取 frozen ViT 最後一層 patch tokens
* 去掉 `[CLS]`
* 保留 `196 × 384`
* 一個 shared linear projection：`384 -> d + 1`
* 前 `d` 維作 feature (h_t)
* 最後 1 維作 uncertainty scalar (r_t)
* `sigma_t = a * sigmoid(r_t) + b` 或 `softplus(r_t) + eps`，讓它保持正值 

因此，對 query/support 一對影像，你至少要有：

* `Hq ∈ R^{Tq×d}`，query patch features
* `Hs ∈ R^{Ts×d}`，support patch features
* `sigma_q ∈ R^{Tq}`
* `sigma_s ∈ R^{Ts}`

在你的 ViT patch case，通常 `Tq = Ts = 196`。guide 也建議 patch 順序固定成 **row-major**，不要改順序，這樣 path matrix 才有幾何意義。

---

## 2. 正確要建的 matrix 是哪些

## 2.1 base distance matrix (D)

先用 feature 建出 pairwise base distance matrix：

[
D_{mn} = | h^q_m - h^s_n |_2^2
]

也就是：

* `D.shape = [Tq, Ts]`
* 在 patch case 就是 `196 × 196`

guide 也明確建議這樣做。

---

## 2.2 uncertainty matrix (\Sigma)

如果你採用 guide 建議的 **additive uncertainty**，那就用：

[
\Sigma_{mn} = \tfrac12 \big(\sigma_q[m]^2 + \sigma_s[n]^2 \big)
]

這正對應論文的 additive variance 形式。論文同時有 additive 和 joint 兩種，但 guide 建議你現在先用 additive，比較穩，也比較符合你的 patch sequence 遷移。 

所以你要有：

* `Sigma.shape = [Tq, Ts]`

---

## 2.3 inverse uncertainty matrix (\Sigma^\dagger)

uDTW 距離不是直接用 (D)，而是用被 uncertainty reweight 過的 cost：

[
\Sigma^\dagger = \text{inv}(\Sigma)
]

這裡的 `inv` 是**逐元素 inverse**。論文 generalized form 也是這樣寫的。

---

## 2.4 uDTW 的有效 cost matrix (C)

uDTW 真正送進 soft-DTW 路徑選擇的，不是 (D)，而是：

[
C = D \odot \Sigma^\dagger
]

論文的 uDTW 定義就是：

[
d^2_{uDTW}(D,\Sigma^\dagger)
============================

\text{SoftMin}*\gamma\big([\langle \Pi, D\odot\Sigma^\dagger\rangle]*{\Pi\in\mathcal A}\big)
]

而 regularization 是：

[
\Omega(\Sigma)
==============

\text{SoftMinSel}*\gamma\big(w,; [\langle \Pi,\log\Sigma\rangle]*{\Pi\in\mathcal A}\big)
]

其中 (w) 就是每一條 path 在 `D ⊙ Σ†` 下的 path cost。

所以你這邊應該顯式存：

* `C = D * Sigma_inv`

---

# 3. Fig. 2 的 (a)–(d) 到底要畫哪個 matrix

這裡是最容易畫錯的地方。

## 3.1 不要直接畫 `D`

`D` 是 pairwise distance matrix。
它是**成本圖**，不是 path 圖。

## 3.2 不要直接畫 `C = D ⊙ Σ†`

`C` 是 uDTW 使用的 reweighted cost matrix。
它還是**成本圖**，不是 path 圖。

## 3.3 你要畫的是 **soft path occupancy / soft alignment matrix**

也就是一個 `Tq × Ts` 的矩陣，表示某個 cell ((m,n)) 在 soft warping path 裡被啟用的強度。

論文在文字上沒有單獨替這個可視化矩陣命名，但它整篇都在講 transportation plan、soft minimum 選 path，以及 path matrix (\Pi)；因此在可微實作上，你應把可視化圖理解成：
**對應於 soft-selected path 的 occupancy matrix**。論文也明確說在 (\gamma\to 0) 時，會回到 one-hot 的 path matrix (\Pi^*)。

### 實作上建議這樣定義

* sDTW 的 path 圖：`A_sdtw = soft_alignment_matrix(D, gamma)`
* uDTW 的 path 圖：`A_udtw = soft_alignment_matrix(C, gamma)`

也就是說：

* **sDTW**：path 是從 `D` 出來的
* **uDTW**：path 是從 `C = D ⊙ Σ†` 出來的

這樣才符合論文「uDTW 用 uncertainty reweighted cost 選 path」的定義。

---

# 4. Fig. 2 每一張圖的正確生成流程

---

## (a) sDTW, (\gamma=0.01)

### 輸入

* `D`

### 步驟

1. 用 `gamma=0.01` 跑 sDTW forward/backward
2. 取出 soft alignment matrix：

   * `A_sdtw_001 = soft_alignment_matrix(D, gamma=0.01)`
3. 做 normalize 方便顯示：

   * `A = A / A.max()`
4. 做 power normalization：

   * `A_vis = A ** 0.1`

論文 caption 明寫：
**他們有做 power-normalized pixels，power = 0.1，目的是讓較暗的 path 也看得見。** 

### 最後畫什麼

* 畫 `A_vis`
* 黑底白線

---

## (b) sDTW, (\gamma=0.1)

完全同上，只是把 gamma 換成 `0.1`：

* `A_sdtw_01 = soft_alignment_matrix(D, gamma=0.1)`
* `A_vis = normalize(A_sdtw_01) ** 0.1`

這一張的目的，是重現論文說的：
**gamma 變大，path 會變 fuzzy，更多 routes 會同時活化。** 

---

## (c) uDTW, (\gamma=0.01)

### 輸入

* `D`
* `Sigma`
* `Sigma_inv`
* `C = D ⊙ Sigma_inv`

### 步驟

1. 先算 `C = D * Sigma_inv`
2. 用 `gamma=0.01` 跑 uDTW 的 soft path
3. 取出：

   * `A_udtw_001 = soft_alignment_matrix(C, gamma=0.01)`
4. normalize
5. power normalize：

   * `A_vis = normalize(A_udtw_001) ** 0.1`

### 最後畫什麼

* 畫 `A_vis`

這一張是 Fig.2 的核心，因為論文指出：
**uDTW 在 uncertainty 建模下可能出現額外可行 routes，而不是只有 sDTW 那條主路徑。** 

---

## (d) uDTW, (\gamma=0.1)

和 (c) 一樣，只是：

* `A_udtw_01 = soft_alignment_matrix(C, gamma=0.1)`

再做 normalize + `**0.1` 顯示。

這一張是用來顯示：

* 大 gamma 下 uDTW 的路徑也會更糊
* uncertainty + softness 疊加後，多路徑現象更明顯 

---

## (e) uDTW uncertainty-on-path

這張是最常被做錯的。

### 你需要的不是 `Σ†`

而是 **`Σ` 本身**。
因為論文 caption 寫的是：

> We binarize plot (c) and multiply it by the Σ to display uncertainty values on the path. 

### 正確步驟

1. 先拿 (c) 的 path matrix：

   * `A_udtw_001`

2. 把它二值化：

   * `A_udtw_001_bin = binarize(A_udtw_001)`

3. 再與 `Σ` 做逐元素乘法：

   * `U_path = A_udtw_001_bin * Sigma`

4. 顯示 `U_path`

### 這張到底代表什麼

它不是 path 強度圖，而是：
**「uDTW 選中的 path 上，每個位置的 uncertainty 有多大」**

白色越亮，表示該路徑位置的 `Σ[m,n]` 越大，也就是不確定性越高。論文甚至直接解釋說，主 path 中段的不確定性比較高，因此會看到另一條 path 在那附近併入主路徑。

---

# 5. `A_*` 要怎麼取得：實作定義

如果你已有 soft-DTW / uDTW 的 forward-backward DP，最乾淨的做法是：

```python
A_sdtw = alignment_from_cost(D, gamma)
A_udtw = alignment_from_cost(C, gamma)   # C = D * Sigma_inv
```

其中 `alignment_from_cost(...)` 的語意應該是：

* 輸入一個 cost matrix
* 輸出同 shape 的 soft alignment / soft path occupancy matrix

也就是把每個 cell 在 soft transportation plan 下的占用強度拿出來。

## 你不該做的事

不要用下面這些直接代替 Fig.2 的 path 圖：

* `imshow(D)`
* `imshow(C)`
* `imshow(Sigma)`
* `imshow(Sigma_inv)`
* `imshow(logSigma)`

這些都不是 Fig.2 (a)–(d)。

---

# 6. 二值化 `A_udtw_001` 時要注意什麼

這裡我要很誠實講：

**論文只說 “We binarize plot (c)” ，但沒有在 caption 裡指定二值化 threshold。** 

所以 threshold 是**你的實作細節**，不是論文硬規定。

## 我建議兩種安全做法

### 做法 A：相對 max threshold

[
A_{\text{bin}} = \mathbf 1[A > \eta \cdot \max(A)]
]

例如：

* `η = 0.05` 或 `0.1`

### 做法 B：分位數 threshold

[
A_{\text{bin}} = \mathbf 1[A > Q_{0.85}(A)]
]

也就是保留最亮的一部分 path。

## 我比較推薦

先用：

```python
thr = 0.05 * A_udtw_001.max()
A_udtw_001_bin = (A_udtw_001 > thr).float()
```

原因很簡單：

* 好解釋
* 可重現
* 不會因不同 pair 導致 binarization 太飄

---

# 7. 顯示前的 normalization 順序

這裡也很重要，不然你的圖會看起來完全不像論文。

## 對 (a)–(d) path 圖

建議：

```python
A = A.clamp(min=0)
A = A / (A.max() + 1e-8)
A_vis = A ** 0.1
```

理由：

* 論文 caption 有明說 power normalization by 0.1 
* 先 normalize 再 power，視覺上最穩

## 對 (e) uncertainty-on-path

建議：

```python
U_path = A_udtw_001_bin * Sigma
U_path = U_path / (U_path.max() + 1e-8)
```

這張是否也做 `**0.1`，論文 caption 沒有明寫；我建議**先不要**，因為這張主要想表達 uncertainty 強度，而不是 path 細節。
若你覺得動態範圍太大，再加 power normalize，但要在圖說中標明。

---

# 8. 你現在這個 patch-sequence 版本，應該怎麼接到論文流程

guide 的最小遷移方案其實很適合做這個 vis：

1. `image -> frozen ViT -> patch tokens`
2. 去掉 CLS，保留 `196×384`
3. `Linear(384, d+1)`
4. 前 `d` 維成 `H`
5. 最後 1 維經 sigmoid/softplus 變成 `sigma`
6. 用 `H` 建 `D`
7. 用 `sigma_q, sigma_s` 建 additive `Σ`
8. 用 `D ⊙ Σ†` 建 uDTW path
9. 畫 `A_sdtw`, `A_udtw`, `A_udtw_bin ⊙ Σ`   

---

# 9. 你應該固定的參數設定

guide 建議一開始：

* sDTW / uDTW 都先用 `gamma=0.01`
* 再測 `0.03 / 0.05`
* 原因是 gamma 太大，patch sequence 的 path 會更糊 

如果你是要**重做論文 Fig.2 風格**，那你應該至少畫：

* `gamma=0.01`
* `gamma=0.1`

因為論文本身就是這兩組。

---

# 10. 很重要：哪些 matrix 用錯會直接讓圖「看起來不像 Fig.2」

## 錯法 1：把 `D` 當 path 圖

這只會得到一張 cost heatmap，不是 path。

## 錯法 2：把 `Σ` 當 (c)/(d)

這只是在畫 uncertainty field，不是 uDTW path。

## 錯法 3：把 `D ⊙ Σ†` 當 (c)/(d)

這還是 cost map，不是 path。

## 錯法 4：把 (e) 做成 `A_udtw ⊙ Σ†`

錯。論文明確是乘 `Σ`。

## 錯法 5：把 `Ω` 畫成熱圖

`Ω(Σ)` 是 scalar penalty，不是 `T×T` matrix。論文定義它是沿 soft-selected path 聚合的 log-variance regularizer。 

---

# 11. 我建議你最後輸出的實作介面

你可以把視覺化 function 明確做成下面這樣：

```python
vis = make_udtw_fig2_visualization(
    Hq, Hs, sigma_q, sigma_s,
    gammas=[0.01, 0.1],
    binarize_ratio=0.05
)
```

它內部應輸出：

```python
{
    "D": D,
    "Sigma": Sigma,
    "Sigma_inv": Sigma_inv,
    "C": C,
    "A_sdtw_001": A_sdtw_001,
    "A_sdtw_01": A_sdtw_01,
    "A_udtw_001": A_udtw_001,
    "A_udtw_01": A_udtw_01,
    "A_udtw_001_bin": A_udtw_001_bin,
    "U_path": U_path,   # = A_udtw_001_bin * Sigma
}
```

這樣你之後 debug 很方便，也不會再混淆：

* 哪些是 cost matrix
* 哪些是 uncertainty matrix
* 哪些才是 path matrix

---

# 12. 最後給你一版最精簡但正確的流程總結

### 論文 Fig.2 (a)–(d)

畫的是：

* `A_sdtw(gamma=0.01)` from `D`
* `A_sdtw(gamma=0.1)` from `D`
* `A_udtw(gamma=0.01)` from `D ⊙ Σ†`
* `A_udtw(gamma=0.1)` from `D ⊙ Σ†`

### 論文 Fig.2 (e)

畫的是：

* `A_udtw(gamma=0.01)` 先二值化
* 再乘上 `Σ`
* 即 `A_udtw_bin ⊙ Σ`

### 不是這些

* 不是 `D`
* 不是 `Σ`
* 不是 `Σ†`
* 不是 `D ⊙ Σ†`
* 不是 `Ω`

---

如果你要，我下一則可以直接補你一份**PyTorch 版偽程式 / 真程式骨架**，把：

1. `build_D_and_Sigma()`
2. `alignment_from_cost()`
3. `make_fig2_panels()`

這三個函式直接寫出來。

---

# 13. 為什麼有時候只看到「一條很明顯的路徑」

即使你用的是 soft alignment matrix，也可能因為抽到的 query-support pair 太「容易對齊」，結果視覺上近似單一路徑。  
這不一定是實作錯，而是 pair selection 的問題。

## 建議做法（已在 `src/mvp_vit_fsl.py` 套用）

1. 不要固定畫 `query[0]` vs `support[0]`
2. 在同類 pair 裡，先算 `A_udtw(gamma=0.1)`
3. 計算 path 分叉分數（branch score），例如：
   * `row_branch = (sum(A) - sum(row_max(A))) / sum(A)`
   * `col_branch = (sum(A) - sum(col_max(A))) / sum(A)`
   * `branch_score = 0.5 * (row_branch + col_branch)`
4. 視覺化 branch score 最高的那組 pair

這樣可以穩定看到「soft path 的多路徑結構」，而不是被單一容易樣本主導。
