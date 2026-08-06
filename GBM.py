# ============================================================
# Gradient Boosting Machine (GBM) - Complete Journal Style
# Combined VI Plot & 2x2 Prediction Matrix
# Candidate Fits: Quadratic (2nd), Cubic (3rd), Quartic (4th)
# Fonts: Times New Roman (12pt everywhere)
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns
from types import SimpleNamespace

# ============================================================
# JOURNAL STANDARD FONTS & GRAPHICS CONFIGURATION (12pt & Times New Roman)
# ============================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.edgecolor'] = '#000000'
plt.rcParams['axes.linewidth'] = 1.2

try:
    from alibi.explainers import ALE  # type: ignore[import]
    HAS_ALIBI = True
except Exception:
    ALE = None
    HAS_ALIBI = False
    print("alibi package not available. Using robust fallback ALE implementation.")

# ============================================================
# ROBUST ALE FALLBACK FUNCTION
# ============================================================
def compute_ale_fallback(model, X_df, feature_names, bins=10):
    feature_values = []
    ale_values = []
    X_working = X_df.copy().reset_index(drop=True)

    for feature in feature_names:
        x = X_working[feature].to_numpy(dtype=float)
        quantiles = np.unique(np.nanpercentile(x, np.linspace(0, 100, bins + 1)))

        if len(quantiles) < 2:
            feature_values.append(np.array([np.nan]))
            ale_values.append(np.array([0.0]))
            continue

        bin_edges = quantiles
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        local_effects = np.zeros(len(bin_centers))

        for k in range(len(bin_centers)):
            low = bin_edges[k]
            high = bin_edges[k + 1]

            if k < len(bin_centers) - 1:
                mask = (x >= low) & (x < high)
            else:
                mask = (x >= low) & (x <= high)

            if not np.any(mask):
                continue

            X_low = X_working.loc[mask].copy()
            X_high = X_low.copy()
            X_low[feature] = low
            X_high[feature] = high

            preds_low = model.predict(X_low)
            preds_high = model.predict(X_high)
            local_effects[k] = np.mean(preds_high - preds_low)

        ale = np.cumsum(local_effects)
        ale = ale - np.nanmean(ale)

        feature_values.append(bin_centers)
        ale_values.append(ale)

    return SimpleNamespace(data={'feature_values': feature_values, 'ale_values': ale_values})

# ============================================================
# ADVANCED POLYNOMIAL SELECTOR WITH OVERFITTING DETECTION
# Compares Linear, Quadratic, Cubic, Quartic using Adjusted R²
# ============================================================
def select_best_poly_fit(x, y, overfitting_threshold=0.05):
    """
    Fit polynomials of degree 1-4, select best using Adjusted R².
    
    Args:
        x: predictor values (array-like)
        y: target values (array-like)
        overfitting_threshold: if improvement in Adj R² < this, prefer simpler model
    
    Returns:
        dict with keys: degree, model_name, r2, adj_r2, aic, bic, coefs, poly_func, 
                        residuals, rmse, overfitting_flag, candidates
    """
    degree_names = {1: "Linear", 2: "Quadratic", 3: "Cubic", 4: "Quartic"}
    n = len(x)
    
    if n < 5:
        # fallback: simple linear
        coefs = np.polyfit(x, y, 1)
        poly_func = np.poly1d(coefs)
        y_fit = poly_func(x)
        residuals = y - y_fit
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_fit)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 1 - 1) if n > 2 else r2
        k = 1
        aic = n * np.log(mse) + 2 * (k + 1)
        bic = n * np.log(mse) + (k + 1) * np.log(n)
        return {
            'degree': 1, 'model_name': 'Linear', 'r2': r2, 'adj_r2': adj_r2,
            'aic': aic, 'bic': bic, 'coefs': coefs, 'poly_func': poly_func,
            'residuals': residuals, 'rmse': rmse, 'overfitting_flag': False,
            'candidates': []
        }
    
    candidates = []
    max_degree = min(4, n - 1)
    allowed_degrees = list(range(1, max_degree + 1))
    
    for d in allowed_degrees:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            coefs = np.polyfit(x, y, d)
            poly_func = np.poly1d(coefs)
            y_fit = poly_func(x)
            residuals = y - y_fit
            mse = np.mean(residuals ** 2)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_fit)
            
            # Adjusted R²: accounts for number of parameters
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - d - 1) if n > d + 1 else r2
            
            # AIC and BIC for model comparison
            aic = n * np.log(mse) + 2 * (d + 1)
            bic = n * np.log(mse) + (d + 1) * np.log(n)
            
            candidates.append({
                'degree': d,
                'model_name': degree_names.get(d, f"Degree {d}"),
                'r2': r2,
                'adj_r2': adj_r2,
                'aic': aic,
                'bic': bic,
                'coefs': coefs,
                'poly_func': poly_func,
                'residuals': residuals,
                'rmse': rmse
            })
    
    # Select best model using Adjusted R² with overfitting detection
    best_idx = 0
    best_adj_r2 = candidates[0]['adj_r2']
    overfitting_flag = False
    
    for idx in range(1, len(candidates)):
        current_adj_r2 = candidates[idx]['adj_r2']
        improvement = current_adj_r2 - best_adj_r2
        
        # If improvement is significant, update best; otherwise prefer simpler model
        if improvement > overfitting_threshold:
            best_adj_r2 = current_adj_r2
            best_idx = idx
        elif idx == len(candidates) - 1 and candidates[idx]['degree'] == 4:
            # Quartic was selected last; check if it's marginally better than Cubic
            if candidates[best_idx]['degree'] < 4 and improvement < 0.02:
                overfitting_flag = True
    
    best = candidates[best_idx].copy()
    best['overfitting_flag'] = overfitting_flag
    best['candidates'] = candidates
    
    return best

# ============================================================
# CONFIDENCE INTERVAL CALCULATION FOR POLYNOMIAL FITS
# ============================================================
def compute_confidence_intervals(x, y, poly_func, residuals, confidence=0.95):
    """
    Compute 95% confidence intervals for polynomial fit using residual standard error.
    
    Args:
        x: predictor values
        y: observed values
        poly_func: fitted numpy poly1d object
        residuals: residuals from fit
        confidence: confidence level (default 0.95 for 95%)
    
    Returns:
        x_smooth, y_fit, ci_lower, ci_upper (all arrays)
    """
    from scipy import stats
    
    n = len(x)
    dof = n - poly_func.order - 1
    
    if dof <= 0:
        # not enough degrees of freedom; return empty CI
        x_smooth = np.linspace(x.min(), x.max(), 100)
        y_fit = poly_func(x_smooth)
        return x_smooth, y_fit, y_fit, y_fit
    
    # residual standard error
    rse = np.sqrt(np.sum(residuals ** 2) / dof)
    
    # t-value for confidence interval
    t_val = stats.t.ppf((1 + confidence) / 2, dof)
    
    x_smooth = np.linspace(x.min(), x.max(), 100)
    y_fit = poly_func(x_smooth)
    
    # leverage / hat matrix diagonal (simplified for polynomial)
    # For each point in x_smooth, estimate the prediction standard error
    X_design = np.column_stack([x_smooth ** i for i in range(poly_func.order + 1)])
    X_orig = np.column_stack([x ** i for i in range(poly_func.order + 1)])
    
    try:
        X_inv = np.linalg.pinv(X_orig)
        # variance of prediction
        pred_var = rse ** 2 * np.sum(X_design * (X_inv.T @ X_design.T), axis=1)
        pred_se = np.sqrt(np.abs(pred_var))  # abs to avoid numerical issues
    except Exception:
        pred_se = rse * np.ones_like(y_fit)
    
    margin = t_val * pred_se
    ci_lower = y_fit - margin
    ci_upper = y_fit + margin
    
    return x_smooth, y_fit, ci_lower, ci_upper

# ============================================================
# FILE PATH & RESULT FOLDER
# ============================================================
file_path = "1.xlsx"
result_folder = "GBM_Results"
os.makedirs(result_folder, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

# INPUT VARIABLES (14 Features)
X = df[[
    'Inf. As', 'Inf. Fe', 'Inf. P', 'Inf. Mn', 'Inf. Mg', 
    'Inf. Ca', 'Inf. Si', 'Inf. TOC', 'Flow rate', 
    'Inf. pH', 'Inf. ORP', 'Inf. EC', 'Inf. Temp', 'Inf. DO'
]]

# OUTPUT TARGETS (4 Parameters)
targets = ['Eff. As', 'Eff. Fe', 'Eff. P', 'Eff. Mn']
performance = []


fig_pred, axes_pred = plt.subplots(2, 2, figsize=(11, 9), facecolor='white')
axes_pred = axes_pred.flatten()


vi_list = []
# ALE values per target collector
ale_data_dict = {}
# Collect all ALE model fits for summary table
ale_model_summary = []

# ============================================================
# LOOP THROUGH EACH TARGET
# ============================================================
for idx, target in enumerate(targets):
    print(f"Running GBM for {target}...")
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    performance.append([target, r2, rmse])

    # --------------------------------------------------------

    # --------------------------------------------------------
    ax = axes_pred[idx]
    ax.set_facecolor('white')
    ax.scatter(y_test, y_pred, alpha=0.75, color='#1f77b4', edgecolors='k', linewidth=0.6)
    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)

    ax.set_xlabel(f"Observed {target}", fontsize=18, fontweight='bold')
    ax.set_ylabel(f"Predicted {target}", fontsize=18, fontweight='bold')
    ax.set_title(f"R² = {r2:.3f} | RMSE = {rmse:.3f}", fontweight='bold', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, linestyle=':', alpha=0.6, color='#cccccc')

    # --------------------------------------------------------

    # --------------------------------------------------------
    vi_target = pd.DataFrame({
        'Variable': X.columns,
        'Importance': model.feature_importances_ * 100,
        'Target': target
    })
    vi_list.append(vi_target)

    # --------------------------------------------------------

    # --------------------------------------------------------
    if HAS_ALIBI:
        predict_fn = model.predict
        ale = ALE(predict_fn, feature_names=X.columns.tolist())
        exp = ale.explain(X_train.values)
    else:
        exp = compute_ale_fallback(model, X_train, X.columns.tolist(), bins=10)

    n_features = len(X.columns)
    ncols = 3
    nrows = int(np.ceil(n_features / ncols))

    fig_ale, axes_ale = plt.subplots(nrows, ncols, figsize=(16, 14), facecolor='white')
    axes_ale = axes_ale.flatten()

    for i in range(n_features):
        axes_ale[i].set_facecolor('white')
        x_vals = exp.data['feature_values'][i]
        y_vals = exp.data['ale_values'][i]

        # collect ALE rows for Excel export
        # store long-form rows: Feature | BinCenter | ALE
        if idx == 0:
            # initialize ale_rows for this target once
            ale_rows = []
        
        axes_ale[i].plot(x_vals, y_vals, color='#1f77b4', linewidth=2.2, label='ALE', marker='o', markersize=4)
        axes_ale[i].axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        
        if len(x_vals) > 2 and not np.isnan(x_vals).any():
            # Use new advanced fitting function
            fit_result = select_best_poly_fit(x_vals, y_vals, overfitting_threshold=0.05)
            
            x_smooth = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_poly = fit_result['poly_func'](x_smooth)
            
            # Compute confidence intervals
            x_ci, y_ci, ci_lower, ci_upper = compute_confidence_intervals(
                x_vals, y_vals, fit_result['poly_func'], fit_result['residuals'], confidence=0.95
            )
            
            # Plot best-fit curve
            axes_ale[i].plot(x_smooth, y_poly, color='#ff7f0e', linestyle='-', linewidth=2.5, 
                        label=f"{fit_result['model_name']}: R²={fit_result['r2']:.3f}, Adj.R²={fit_result['adj_r2']:.3f}")
            
            # Plot 95% confidence interval band
            axes_ale[i].fill_between(x_ci, ci_lower, ci_upper, alpha=0.2, color='#ff7f0e', label='95% CI')
            
            # Collect model info for summary table
            overfitting_flag_str = "Yes" if fit_result['overfitting_flag'] else "No"
            ale_model_summary.append({
                'Target': target,
                'Predictor': X.columns[i],
                'Selected_Model': fit_result['model_name'],
                'R2': fit_result['r2'],
                'Adj_R2': fit_result['adj_r2'],
                'AIC': fit_result['aic'],
                'BIC': fit_result['bic'],
                'RMSE': fit_result['rmse'],
                'Overfitting_Flag': overfitting_flag_str
            })
            
            legend = axes_ale[i].legend(loc='best', frameon=False, handlelength=1.5, fontsize=10)
            for txt in legend.get_texts():
                txt.set_fontsize(10)
        
        # append ALE values to rows (ensure arrays are same length)
        try:
            for bc, av in zip(x_vals, y_vals):
                ale_rows.append({'Feature': X.columns[i], 'BinCenter': float(bc), 'ALE': float(av)})
        except Exception:
            pass
        axes_ale[i].set_title(X.columns[i], fontweight='bold', fontsize=14)
        axes_ale[i].set_xlabel('Predictor Value', fontsize=11)
        axes_ale[i].set_ylabel('ALE Effect', fontsize=11)
        axes_ale[i].tick_params(axis='both', which='major', labelsize=10)
        axes_ale[i].grid(True, linestyle=':', alpha=0.5, color='#cccccc')

    for j in range(n_features, len(axes_ale)):
        axes_ale[j].axis('off')

    plt.suptitle(f"Accumulated Local Effects (ALE) with Targeted Poly-Fit - {target}", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(result_folder, f"{target}_ALE.png"), dpi=300)
    plt.close()

    # Save collected ALE rows into dict as DataFrame
    try:
        ale_df = pd.DataFrame(ale_rows)
    except Exception:
        ale_df = pd.DataFrame(columns=['Feature', 'BinCenter', 'ALE'])
    ale_data_dict[target] = ale_df


fig_pred.suptitle("Model Prediction Performance (2x2 Matrix)", fontsize=18, fontweight='bold', y=0.98)
fig_pred.tight_layout(rect=[0, 0, 1, 0.96])
fig_pred.savefig(os.path.join(result_folder, "Combined_Prediction_2x2.png"), dpi=300)
plt.close()

# ============================================================

# ============================================================
combined_vi = pd.concat(vi_list, ignore_index=True)
order = combined_vi.groupby('Variable')['Importance'].mean().sort_values(ascending=False).index

targets_vi = ['Eff. As', 'Eff. Fe', 'Eff. P', 'Eff. Mn']
fig_vi, axes_vi = plt.subplots(2, 2, figsize=(14, 12), facecolor='white')
axes_vi = axes_vi.flatten()

for ax_idx, target_name in enumerate(targets_vi):
    ax = axes_vi[ax_idx]
    ax.set_facecolor('white')
    target_data = combined_vi[combined_vi['Target'] == target_name]
    sns.barplot(
        data=target_data,
        x='Importance',
        y='Variable',
        order=order,
        palette='viridis',
        ax=ax,
        edgecolor='k',
        linewidth=0.6
    )
    ax.set_title(f"{target_name} Variable Importance", fontsize=20, fontweight='bold')
    ax.set_xlabel("Importance (%)", fontsize=18, fontweight='bold')
    ax.set_ylabel("")
    ax.grid(True, axis='x', linestyle=':', alpha=0.5, color='#cccccc')
    ax.set_xlim(0, combined_vi['Importance'].max() * 1.05)
    ax.tick_params(axis='y', labelsize=20)


for idx in range(len(targets_vi), len(axes_vi)):
    axes_vi[idx].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.suptitle("Variable Importance by Target (2x2 Matrix)", fontsize=18, fontweight='bold', y=0.99)
plt.savefig(os.path.join(result_folder, "Combined_Variable_Importance_2x2.png"), dpi=300)
plt.close()

# ============================================================
# WRITE VI AND ALE VALUES TO AN EXCEL WORKBOOK
# ============================================================
excel_path = os.path.join(result_folder, "GBM_VI_ALE.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    # per-target VI sheets
    for vi_df in vi_list:
        try:
            tname = vi_df['Target'].iloc[0]
        except Exception:
            tname = 'UnknownTarget'
        sheet_name = f"{tname}_VI"
        vi_df[['Variable', 'Importance']].to_excel(writer, sheet_name=sheet_name, index=False)

    # per-target ALE sheets (long format)
    for tname, ale_df in ale_data_dict.items():
        sheet_name = f"{tname}_ALE"
        # if sheet name too long or contains invalid chars, pandas will handle it
        ale_df.to_excel(writer, sheet_name=sheet_name, index=False)

    # also write combined VI
    try:
        combined_vi.to_excel(writer, sheet_name='Combined_VI', index=False)
    except Exception:
        pass
    
    # Write ALE model summary with model selection info
    if ale_model_summary:
        ale_summary_df = pd.DataFrame(ale_model_summary)
        ale_summary_df.to_excel(writer, sheet_name='ALE_Model_Summary', index=False)

# ============================================================
# SAVE ALE MODEL SUMMARY AND PERFORMANCE TABLE
# ============================================================
perf_df = pd.DataFrame(performance, columns=["Target", "R2", "RMSE"])
perf_df.to_csv(os.path.join(result_folder, "GBM_Performance.csv"), index=False)

# Save ALE model summary as CSV for easy reference
if ale_model_summary:
    ale_summary_df = pd.DataFrame(ale_model_summary)
    ale_summary_df.to_csv(os.path.join(result_folder, "ALE_Model_Summary.csv"), index=False)
    print(f"\nALE Model Summary saved: ALE_Model_Summary.csv")
    print(f"  - Shows selected model type, R², Adjusted R², and overfitting flags")
    print(f"  - Contains {len(ale_model_summary)} model fits ({len(targets)} targets × {n_features} predictors)")

print("\n============================================================")
print("SUCCESS: Unified Plots Generated Successfully (Publication-Ready)!")
print(f"Check files: 'Combined_Prediction_2x2.png' and 'Combined_Variable_Importance_2x2.png'")
print("=============================================================")
