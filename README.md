# EigenSketch — A Linear-Algebraic Scoring Engine for Freehand Drawings

> **Abstract.** EigenSketch evaluates how closely a human-drawn sketch matches a target template by solving classical alignment problems (Orthogonal/Similarity Procrustes, Affine LSQ, PCA alignment) and aggregating error terms (RMSE, curvature-weighted Chamfer, corner proximity) into a 0–100 score. The project demonstrates how foundational linear algebra translates directly into an interactive web application with robust input handling and responsive rendering.

---

## Table of Contents
1. Motivation & Learning Goals  
2. Mathematical Formulation  
   2.1 Pre-normalization  
   2.2 Alignment models  
   2.3 Error terms & scoring (0–100)  
3. Algorithms & Pseudocode  
4. System Architecture (Web)  
5. Template Format (`.lal.json`)  
6. Reproducibility, Complexity & Performance Notes  
7. Running, Developing, and Extending  
8. Limitations & Future Work  
9. References (short)

---

## 1) Motivation & Learning Goals

- **Linear Algebra in practice.** Connect abstract concepts—centroids, covariance, eigensystems, orthogonal/affine transforms—to an immediate, visual task: “Does this drawing match the target?”
- **Robustness to pose.** Remove confounds of translation, scale, and rotation using closed-form solutions.
- **Model comparison.** Switch among Orthogonal, Similarity, Affine, and PCA alignment to see how different constraints affect residuals and scores.
- **Programming discipline.** Clean, framework-free architecture (HTML/CSS/JS), numerical stability (coalesced pointer events, DPR-aware rendering), and simple data interchange (JSON templates).

---

## 2) Mathematical Formulation

### 2.1 Pre-normalization (unit space)
Given a polyline point set \( P = \{p_i\}_{i=1}^n,\ p_i\in\mathbb{R}^2 \):

- **Centroid** \( \mu_P = \frac{1}{n}\sum_i p_i \)
- **Centered set** \( P' = \{p_i - \mu_P\} \)
- **RMS radius** \( r_P = \sqrt{\frac{1}{n}\sum_i \|p_i - \mu_P\|^2} \)
- **Unit set** \( \widehat{P} = \{(p_i-\mu_P)/\max(r_P,\varepsilon)\} \)

Both the **template** \(T\) and **user strokes** \(U\) are resampled and mapped to unit space. This removes dependence on absolute position and scale before alignment.

### 2.2 Alignment Models

Let \(X\) be user points and \(Y\) template points (both in unit space, paired via resampling).

**Cross-covariance accumulators**  
Let \( M = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \), where
\[
a=\sum x_i u_i, \quad b=\sum x_i v_i, \quad c=\sum y_i u_i, \quad d=\sum y_i v_i,
\]
with \(X_i=(x_i,y_i)\) and \(Y_i=(u_i,v_i)\).
Rotation angle:
\[
\theta = \operatorname{atan2}(b-c,\ a+d), \quad R(\theta) =
\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}.
\]

- **Orthogonal (R,t)**: \( s=1 \). Translation \( t = \mu_Y - R\,\mu_X \).
- **Similarity (sR,t)** (Procrustes/Kabsch in 2D):  
  \( s = \dfrac{\operatorname{tr}(R M)}{\sum_i \|X_i\|^2} \), \( t = \mu_Y - sR\,\mu_X \).
- **Affine (A,t)**: Solve LSQ for \( A\in\mathbb{R}^{2\times2},\, t\in\mathbb{R}^2 \) minimizing \( \sum_i \|A X_i + t - Y_i\|^2 \) via normal equations.
- **PCA alignment (sR,t)**: Align principal axes by eigenvectors of covariance matrices; scale by RMS radii.

**Optional ICP refinement**:  
Nearest-neighbor pairs between \( sR\,X+t \) and \( Y \) → small Procrustes update; iterate \(k\) times.

### 2.3 Error Terms & Composite Score

Let \( \widetilde{X} \) denote the aligned user set.

1) **RMSE** (pointwise pairing via resampling)
\[
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_i \| \widetilde{X}_i - Y_i \|^2}.
\]
Normalize by tolerance \( \tau \) (UI “Strictness” and difficulty):
\[
q_{\text{RMSE}} = \mathrm{clip}\left(1 - \frac{\mathrm{RMSE}}{\tau},\,0,\,1\right).
\]

2) **Curvature-weighted Chamfer coverage**  
For each template point \(y\), let \(d(y)=\min_{x\in \widetilde{X}} \|x-y\|\). Weights \(w(y)\ge 1\) emphasize high curvature (bend points).  
Coverage fraction within \(\tau\):
\[
q_{\text{cov}} = \frac{\sum_y w(y)\,\mathbf{1}[d(y)\le\tau]}{\sum_y w(y)}.
\]

3) **Corner proximity**  
For template corners \(C\), average nearest-neighbor distance \(\bar d\), then
\[
q_{\text{corn}} = \mathrm{clip}\!\left(1 - \frac{\bar d}{0.8\,\tau},\,0,\,1\right).
\]

**Final score (0–100)**  
\[
\text{Score} = 100\cdot \big(0.5\,q_{\text{RMSE}} + 0.3\,q_{\text{cov}} + 0.2\,q_{\text{corn}}\big).
\]

---

## 3) Algorithms & Pseudocode

**Similarity Procrustes (2D):**
```text
Input: X[1..n], Y[1..n]  // centered or unit-space
Compute M = [[a,b],[c,d]] from sums over pairs
θ = atan2(b - c, a + d)
R = [[cosθ, -sinθ], [sinθ, cosθ]]
xx = Σ ||X_i||^2
s  = (trace(R * M)) / max(xx, ε)
t  = μY - s R μX
Output: s, R, t
