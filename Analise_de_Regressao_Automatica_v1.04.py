#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, BooleanVar

# Configurações de inicialização
warnings.filterwarnings('ignore')
matplotlib.use('TkAgg')

# ── Imports estatísticos e Machine Learning ──────────────────────────────────
from scipy import stats
from scipy.stats import shapiro, kstest
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import confusion_matrix, log_loss, f1_score, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.discrete.discrete_model import MNLogit, NegativeBinomial

from numpy.linalg import det

# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES DE DIAGNÓSTICO CATEGÓRICA/MULTINOMIAL
# ─────────────────────────────────────────────────────────────────────────────

def exibir_heatmap(df, x_cols):
    """Gera um heatmap de correlação para os preditores numéricos."""
    num_cols = [c for c in x_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        messagebox.showinfo("Heatmap", "Heatmap requer ao menos 2 variáveis numéricas entre os preditores.")
        return
    plt.figure(figsize=(8, 6))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de Correlação dos Preditores")
    plt.tight_layout()
    plt.show()


def plotar_diagnostico_multinomial_completo(y_real, y_prob, titulo):
    """
    Gera diagnósticos avançados para modelos multinomiais em layout 3x2.
    Removeu-se a magnitude do erro para otimização do espaço.
    """
    n_classes = y_prob.shape[1]
    y_pred_cls = np.argmax(y_prob, axis=1)
    
    # Configuração da figura 3x2
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    fig.suptitle(f"Diagnóstico Multinomial: {titulo}", fontsize=16, fontweight='bold', y=0.98)

    # Estilo e Paleta
    ACCENT = '#4f46e5'
    palette = sns.color_palette("husl", n_classes)

    # --- 1. Matriz de Confusão [0, 0] ---
    cm = confusion_matrix(y_real, y_pred_cls)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False)
    axes[0, 0].set_title("Matriz de Confusão", fontsize=12)
    axes[0, 0].set_xlabel("Predito")
    axes[0, 0].set_ylabel("Real")

    # --- 2. Densidade de Probabilidades [0, 1] ---
    for i in range(n_classes):
        sns.kdeplot(y_prob[:, i], ax=axes[0, 1], label=f'Classe {i}', color=palette[i], fill=True, alpha=0.1)
    axes[0, 1].set_title("Densidade de Probabilidades por Classe", fontsize=12)
    axes[0, 1].set_xlabel("Probabilidade Predita")
    axes[0, 1].legend(fontsize='small')

    # --- 3. Confiança do Modelo (Boxplot) [1, 0] ---
    prob_real = np.array([y_prob[i, val] for i, val in enumerate(y_real)])
    df_aux = pd.DataFrame({'Real': y_real, 'Prob': prob_real})
    sns.boxplot(x='Real', y='Prob', data=df_aux, ax=axes[1, 0], palette="Set3")
    axes[1, 0].set_title("Confiança na Classe Real (Boxplot)", fontsize=12)
    axes[1, 0].set_ylabel("Probabilidade Atribuída")

    # --- 4. Curva ROC Multiclasse (OvR) [1, 1] ---
    y_real_bin = label_binarize(y_real, classes=np.arange(n_classes))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_real_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        axes[1, 1].plot(fpr, tpr, color=palette[i], lw=2, label=f'Cl {i} (AUC={roc_auc:.2f})')
    
    axes[1, 1].plot([0, 1], [0, 1], color='gray', ls='--')
    axes[1, 1].set_title("Curva ROC (One-vs-Rest)", fontsize=12)
    axes[1, 1].set_xlabel("FPR")
    axes[1, 1].set_ylabel("TPR")
    axes[1, 1].legend(fontsize='x-small', loc='lower right')

    # --- 5. Dispersão de Confiança (Stripplot) [2, 0] ---
    sns.stripplot(x='Real', y='Prob', data=df_aux, ax=axes[2, 0], alpha=0.3, jitter=True, color='gray')
    axes[2, 0].set_title("Dispersão de Confiança por Classe", fontsize=12)
    axes[2, 0].set_xlabel("Classe Real")
    axes[2, 0].set_ylabel("Probabilidade")

    # --- 6. Sumário de Métricas [2, 1] ---
    ll = log_loss(y_real, y_prob)
    f1_m = f1_score(y_real, y_pred_cls, average='macro')
    f1_w = f1_score(y_real, y_pred_cls, average='weighted')
    acc = accuracy_score(y_real, y_pred_cls)

    texto_metricas = (
        f"Métricas Globais\n\n"
        f"Log Loss: {ll:.4f}\n"
        f"Acurácia: {acc:.4f}\n"
        f"F1-Score (Macro): {f1_m:.4f}\n"
        f"F1-Score (Weighted): {f1_w:.4f}"
    )
    axes[2, 1].text(0.5, 0.5, texto_metricas, ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=1.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9))
    axes[2, 1].set_title("Sumário de Performance", fontsize=12, fontweight='bold')
    axes[2, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES DE APOIO GERAIS
# ─────────────────────────────────────────────────────────────────────────────


def identificar_tipo_variavel(serie: pd.Series) -> str:
    serie = serie.dropna()

    if pd.api.types.is_bool_dtype(serie):
        return "binaria"

    if pd.api.types.is_numeric_dtype(serie):
        serie_num = pd.to_numeric(serie, errors="coerce").dropna()
        n_unique = serie_num.nunique()  # ← recalcula APÓS conversão numérica
        if n_unique == 2:
            return "binaria"
        try:
            is_int = (serie_num == serie_num.astype(int)).all()
            is_pos = (serie_num >= 0).all()
            if is_int and is_pos and n_unique > 2:
                if n_unique <= 20 or serie_num.max() <= 50:
                    return "contagem"
        except (ValueError, TypeError):
            pass
        return "continua"

    # Não numérica
    n_unique = serie.nunique()  # ← recalcula para texto
    return "binaria" if n_unique == 2 else "categorica"


def calcular_vif(X: pd.DataFrame):
    import numpy as np
    from numpy.linalg import inv, det

    X_encoded_parts = []
    mapa_grupos = {} # Mapeia quais colunas pertencem a qual variável original

    # 1. Preparação e Mapeamento
    for col in X.columns:
        tipo = identificar_tipo_variavel(X[col])
        is_str = X[col].dtype == object or pd.api.types.is_string_dtype(X[col])

        if tipo == "categorica" or (tipo == "binaria" and is_str):
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True, dtype=float)
            X_encoded_parts.append(dummies)
            mapa_grupos[col] = list(dummies.columns)
        else:
            col_num = X[[col]].copy()
            col_num[col] = pd.to_numeric(col_num[col], errors="coerce")
            X_encoded_parts.append(col_num)
            mapa_grupos[col] = [col]

    X_enc = pd.concat(X_encoded_parts, axis=1).dropna()
    all_cols = list(X_enc.columns)
    col_to_idx = {c: i for i, c in enumerate(all_cols)}

    # 2. Cálculos Matriciais (Inversa para Estabilidade)
    R = X_enc.corr().values
    try:
        R_inv = inv(R)
    except np.linalg.LinAlgError:
        return pd.DataFrame({"Variável": X.columns, "Status": "Erro: Colinearidade Perfeita"})

    vif_data = []

    # 3. Consolidação por Variável Original
    for var_original, colunas_dummy in mapa_grupos.items():
        indices = [col_to_idx[c] for c in colunas_dummy]
        df_j = len(indices)
        
        # Para variáveis contínuas (df=1), o GVIF é o elemento da diagonal da inversa
        # Para categóricas, usamos a submatriz da inversa para calcular o GVIF consolidado
        sub_R = R[np.ix_(indices, indices)]
        sub_R_inv = R_inv[np.ix_(indices, indices)]
        
        # GVIF = det(R_sub) * det(R_inv_sub)
        gvif = det(sub_R) * det(sub_R_inv)
        
        # GVIF Ajustado para permitir comparação entre diferentes Df
        gvif_adj = gvif ** (1 / (2 * df_j))
        
        # Status baseado em critérios de rigor metodológico
        status = "✔ OK" if gvif_adj < 3.16 else ("⚠ Moderado" if gvif_adj < 3.87 else "✘ Alto")

        vif_data.append({
            "Variável": var_original,
            "Df": df_j,
            "GVIF": round(gvif, 4),
            "GVIF^(1/2Df)": round(gvif_adj, 4),
            "Status": status
        })


    vif_data = pd.DataFrame(vif_data)
    return vif_data
        


    

def teste_normalidade(serie: pd.Series):
    """Shapiro-Wilk (n<=50) ou Kolmogorov-Smirnov."""
    serie = serie.dropna()
    if len(serie) < 3:
        return None, None, "Amostra insuficiente"
    if len(serie) <= 50:
        stat, p = shapiro(serie)
        return stat, p, "Shapiro-Wilk"
    else:
        stat, p = kstest((serie - serie.mean()) / serie.std(), 'norm')
        return stat, p, "Kolmogorov-Smirnov"

def teste_equidispersao(y: pd.Series, y_pred: np.ndarray):
    """Razão variância/média."""
    ratio = y.var() / y.mean()
    return ratio

def excesso_de_zeros(y: pd.Series):
    """Proporção de zeros vs esperado por Poisson."""
    prop_obs  = (y == 0).mean()
    lambda_   = y.mean()
    prop_esp  = np.exp(-lambda_)
    return prop_obs, prop_esp

# ─────────────────────────────────────────────────────────────────────────────
# PLOTAGEM DE DIAGNÓSTICO 2x2
# ─────────────────────────────────────────────────────────────────────────────


def plotar_diagnostico_2x2(y_real, y_pred, residuos, model, titulo, tipo="linear"):
    # Grid 3x2 para manter a consistência visual
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    fig.suptitle(f"Diagnóstico: {titulo}", fontsize=16, fontweight='bold', y=0.98)

    # Cores de destaque
    ACCENT = '#4f46e5'
    ACCENT2 = '#7c3aed'
    DANGER = '#dc2626'
    WARNING = '#d97706'
    SUCCESS = '#10b981' 

    # --- PAINEL 0,0: Resíduos vs Preditos (Comum a todos) ---
    axes[0, 0].scatter(y_pred, residuos, color=ACCENT, alpha=0.5, s=25)
    axes[0, 0].axhline(0, color=DANGER, linestyle='--')
    axes[0, 0].set_title("Resíduos vs Preditos")
    axes[0, 0].set_xlabel("Valores Preditos")
    axes[0, 0].set_ylabel("Resíduos")
    
    # Identificação de outliers nos resíduos (2 desvios padrão)
    outliers_res = np.where(np.abs(residuos) > 2 * np.std(residuos))[0]
    for i in outliers_res:
        label = y_real.index[i] if hasattr(y_real, 'index') else i
        axes[0, 0].annotate(label, (y_pred[i], residuos[i]), color=DANGER, fontsize=8)

    # --- PAINEL 0,1: Distância de Cook (Comum a todos) ---
    try:
        influence = model.get_influence()
        (c, _) = influence.cooks_distance
        axes[0, 1].stem(np.arange(len(c)), c, markerfmt=",", linefmt=ACCENT)
        
        limiar_4n = 4 / len(y_real)
        limiar_3media = 3 * np.mean(c)
        
        axes[0, 1].axhline(limiar_4n, color=WARNING, linestyle='--', label=f'Limiar 4/n ({limiar_4n:.3f})')
        axes[0, 1].axhline(limiar_3media, color=SUCCESS, linestyle='-.', label=f'3x Média ({limiar_3media:.3f})')
        
        outliers_cook = np.where(c > min(limiar_4n, limiar_3media))[0]
        for i in outliers_cook:
            label = y_real.index[i] if hasattr(y_real, 'index') else i
            axes[0, 1].annotate(label, (i, c[i]), xytext=(0, 5), textcoords='offset points', 
                                ha='center', color=DANGER, fontsize=8, fontweight='bold')
        
        axes[0, 1].set_title("Influência (Distância de Cook)")
        axes[0, 1].legend(fontsize='x-small', loc='upper right')
    except Exception:
        axes[0, 1].text(0.5, 0.5, "Distância de Cook indisponível", ha='center', va='center', style='italic')

    # --- LÓGICA DE CONDICIONAL POR TIPO ---
    if tipo == "linear":
        # Painel 1,0: Distribuição
        sns.histplot(residuos, kde=True, color=ACCENT2, ax=axes[1, 0])
        axes[1, 0].set_title("Distribuição dos Resíduos")
        
        # Painel 1,1: QQ-Plot
        import statsmodels.api as sm
        sm.qqplot(np.array(residuos), line='s', ax=axes[1, 1])
        axes[1, 1].set_title("QQ-Plot dos Resíduos")

        # Painel 2,0: Scale-Location
        residuos_std = (residuos - np.mean(residuos)) / np.std(residuos)
        sqrt_abs_residuos = np.sqrt(np.abs(residuos_std))
        sns.scatterplot(x=y_pred, y=sqrt_abs_residuos, color=ACCENT, alpha=0.5, ax=axes[2, 0])
        sns.regplot(x=y_pred, y=sqrt_abs_residuos, scatter=False, ci=None, lowess=True, 
                    line_kws={'color': DANGER, 'lw': 2}, ax=axes[2, 0])
        axes[2, 0].set_title("Scale-Location (Homocedasticidade)")

        # Painel 2,1: Testes Estatísticos
        stat_sh, p_sh = stats.shapiro(residuos)
        from statsmodels.stats.diagnostic import het_breuschpagan
        try:
            exog = model.model.exog if hasattr(model.model, 'exog') else model.model.data.exog
            _, p_bp, _, _ = het_breuschpagan(residuos, exog)
        except:
            p_bp = np.nan

        texto_resumo = (f"Testes Estatísticos\n\nShapiro-Wilk (p): {p_sh:.4f}\n"
                        f"Breusch-Pagan (p): {p_bp:.4f}")
        cor_f = "green" if (p_sh > 0.05 and (np.isnan(p_bp) or p_bp > 0.05)) else DANGER
        axes[2, 1].text(0.5, 0.5, texto_resumo, ha='center', va='center', fontsize=11,
                        bbox=dict(boxstyle="round,pad=1.5", facecolor="white", edgecolor="gray", alpha=0.8))
        axes[2, 1].text(0.5, 0.1, "Pressupostos OK" if cor_f == "green" else "Violação Detectada", 
                        ha='center', color=cor_f, fontweight='bold')
        axes[2, 1].axis('off')

    elif tipo == "binaria":
        y_class = (np.array(y_pred) > 0.5).astype(int)
        sns.heatmap(confusion_matrix(y_real, y_class), annot=True, fmt='d', cmap='Purples', ax=axes[1, 0], cbar=False)
        axes[1, 0].set_title("Matriz de Confusão")

        fpr, tpr, _ = roc_curve(y_real, y_pred)
        axes[1, 1].plot(fpr, tpr, color=ACCENT2, lw=2, label=f'AUC = {auc(fpr, tpr):.2f}')
        axes[1, 1].plot([0, 1], [0, 1], color='gray', ls='--')
        axes[1, 1].set_title("Curva ROC")
        axes[1, 1].legend(loc='lower right')

        # Binned Residual Plot
        df_b = pd.DataFrame({'p': y_pred, 'r': residuos})
        df_b['bin'] = pd.qcut(df_b['p'], q=min(10, df_b['p'].nunique()), duplicates='drop')
        b = df_b.groupby('bin', observed=True).agg({'p': 'mean', 'r': 'mean', 'bin': 'count'})
        axes[2, 0].errorbar(b['p'], b['r'], yerr=2*(np.std(residuos)/np.sqrt(b['bin'])), fmt='o', color=ACCENT)
        axes[2, 0].axhline(0, color=DANGER, linestyle='--')
        axes[2, 0].set_title("Resíduos Agrupados (Binned)")

        # Hosmer-Lemeshow
        data_hl = pd.DataFrame({'y': y_real, 'p': y_pred})
        data_hl['bin'] = pd.qcut(data_hl['p'], 10, duplicates='drop')
        hl_df = data_hl.groupby('bin', observed=True).agg(obs_1=('y', 'sum'), total=('y', 'count'), prob=('p', 'mean'))
        hl_df['exp_1'], hl_df['exp_0'] = hl_df['total']*hl_df['prob'], hl_df['total']*(1-hl_df['prob'])
        hl_df['obs_0'] = hl_df['total'] - hl_df['obs_1']
        chi_sq = (((hl_df['obs_1']-hl_df['exp_1'])**2/hl_df['exp_1']) + ((hl_df['obs_0']-hl_df['exp_0'])**2/hl_df['exp_0'])).sum()
        p_val = 1 - stats.chi2.cdf(chi_sq, df=max(1, len(hl_df)-2))
        
        t_hl = f"Hosmer-Lemeshow\n\nChi-sq: {chi_sq:.2f}\np-valor: {p_val:.4f}"
        axes[2, 1].text(0.5, 0.5, t_hl, ha='center', va='center', fontsize=11, bbox=dict(boxstyle="round,pad=1", facecolor="white", alpha=0.8))
        axes[2, 1].text(0.5, 0.2, "Modelo Adequado" if p_val > 0.05 else "Modelo Inadequado", ha='center', color="green" if p_val > 0.05 else "red", fontweight='bold')
        axes[2, 1].axis('off')

    elif tipo == "contagem":
        sns.histplot(residuos, kde=True, color=ACCENT2, ax=axes[1, 0])
        axes[1, 0].set_title("Distribuição (Pearson)")

        axes[1, 1].scatter(y_real, y_pred, color=ACCENT, alpha=0.5)
        lims = [np.min([y_real, y_pred]), np.max([y_real, y_pred])]
        axes[1, 1].plot(lims, lims, color=DANGER, ls='--')
        axes[1, 1].set_title("Predito vs Real")

        max_v = int(min(max(y_real), 20))
        obs = np.bincount(y_real.astype(int), minlength=max_v+1)[:max_v+1]
        pred = np.bincount(np.round(y_pred).astype(int).clip(0, max_v), minlength=max_v+1)[:max_v+1]
        axes[2, 0].bar(np.arange(len(obs))-0.2, obs, width=0.4, label='Real', color=ACCENT, alpha=0.7)
        axes[2, 0].bar(np.arange(len(pred))+0.2, pred, width=0.4, label='Predito', color=ACCENT2, alpha=0.7)
        axes[2, 0].set_title("Frequências")
        axes[2, 0].legend()

        # --- PAINEL 2,1: Diagnóstico de Dispersão e Ajuste ---
        pearson_chi2 = np.sum(residuos**2)
        disp_ratio = pearson_chi2 / model.df_resid
        p_fit = 1 - stats.chi2.cdf(getattr(model, 'deviance', pearson_chi2), model.df_resid)
        
        # Lógica de Interpretação Automática
        # 1. Avaliação da Dispersão
        if disp_ratio > 1.2:
            desc_disp = "Superdispersão (Var > Média)"
            cor_status = DANGER
        elif disp_ratio < 0.8:
            desc_disp = "Subdispersão (Var < Média)"
            cor_status = WARNING
        else:
            desc_disp = "Equidispersão (Ideal)"
            cor_status = "green"

        # 2. Avaliação do Ajuste Global
        desc_fit = "Ajuste Adequado (p > 0.05)" if p_fit > 0.05 else "Ajuste Inadequado (p <= 0.05)"

        # Montagem do Texto Final
        t_cont = (
            f"Diagnóstico de Ajuste\n\n"
            f"Dispersion Ratio: {disp_ratio:.4f}\n"
            f"({desc_disp})\n\n"
            f"Goodness-of-Fit (p): {p_fit:.4f}\n"
            f"({desc_fit})"
        )

        # Renderização no Gráfico
        axes[2, 1].text(0.5, 0.5, t_cont, ha='center', va='center', fontsize=11, 
                        linespacing=1.5, # Melhora a leitura do bloco de texto
                        bbox=dict(boxstyle="round,pad=1.5", facecolor="white", edgecolor="gray", alpha=0.9))
        
        # Indicador visual rápido na parte inferior
        axes[2, 1].text(0.5, 0.1, "VALIDAÇÃO FINALIZADA", ha='center', 
                        color=cor_status, fontweight='bold', fontsize=10)
        
        axes[2, 1].axis('off')
        

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.show()
    
# ─────────────────────────────────────────────────────────────────────────────
# MOTORES DE ANÁLISE
# ─────────────────────────────────────────────────────────────────────────────

"""
Função analisar_continua — versão atualizada
Adiciona, quando a normalidade da VD é violada:
  1. Tentativa de transformação log1p e sqrt
  2. Seleção automática da melhor transformação (pelo p-valor de normalidade)
  3. Re-ajuste do modelo OLS com a variável transformada
  4. Regressão Robusta (RLM / MM-estimator) como alternativa
  5. Comparação AIC/BIC entre OLS original, OLS transformado e RLM
  6. Recomendação explícita ao usuário
Nenhuma outra funcionalidade foi removida.
"""

# ─── Função auxiliar: testa transformação e retorna série + info ──────────────

def _tentar_transformacao(y: pd.Series, nome: str, func):
    """
    Aplica func(y), verifica viabilidade (ex: log requer y > 0),
    roda teste de normalidade e retorna dict com resultados.
    """
    try:
        y_t = func(y)
        if y_t.isnull().any() or np.isinf(y_t).any():
            return None
        stat, p, teste = teste_normalidade(y_t)
        return {
            "nome":   nome,
            "serie":  y_t,
            "stat":   stat,
            "p":      p,
            "teste":  teste,
            "normal": (p is not None and p > 0.05),
        }
    except Exception:
        return None


# ─── Função principal ─────────────────────────────────────────────────────────

def analisar_continua(df, y_col, x_cols, norm_y=False, app=None):
    linhas = []
    add = linhas.append

    # ── 1. Preparação e Sincronização de Dados ────────────────────────────────
    y = df[y_col].dropna()
    X_df = df[x_cols].loc[y.index].dropna()
    y = y.loc[X_df.index]

    if norm_y:
        y = pd.Series(
            StandardScaler().fit_transform(y.values.reshape(-1, 1)).flatten(),
            index=y.index
        )
        add("[!] Z-Score aplicado na variável desfecho (Y).")

    norm_x = app.flag_norm.get() if (app and hasattr(app, 'flag_norm')) else True
    X_enc = pd.get_dummies(X_df, drop_first=True).astype(float)
    if norm_x:
        X_scaled = pd.DataFrame(
            StandardScaler().fit_transform(X_enc),
            columns=X_enc.columns, index=X_enc.index
        )
        add("[!] Z-Score aplicado nos preditores (X).")
    else:
        X_scaled = X_enc.copy()

    X_const = sm.add_constant(X_scaled)

    # ── 2. Ajuste do Modelo OLS Original ─────────────────────────────────────
    model = sm.OLS(y, X_const).fit()

    add("=" * 62)
    add("  ANÁLISE COMPLETA — REGRESSÃO LINEAR")
    add("=" * 62)

    # ── [1] Normalidade da VD ─────────────────────────────────────────────────
    stat, p_norm, nome_t = teste_normalidade(y)
    normalidade_ok = (p_norm is not None and p_norm > 0.05)

    add(f"\n[1] Normalidade da VD ({nome_t}): stat = {stat:.4f} | p = {p_norm:.4f} {'✔' if normalidade_ok else '✘'}")
    if normalidade_ok:
        add("    → Distribuição normal. Pressuposto ATENDIDO.")
    else:
        add("    → Distribuição não-normal. Pressuposto VIOLADO.")
        add("      O sistema irá automaticamente:")
        add("        (a) Testar transformações matemáticas (log1p e sqrt)")
        add("        (b) Ajustar regressão robusta (RLM — M/MM-estimator)")
        add("        (c) Comparar os modelos e recomendar o mais adequado.")
        add("      Veja a seção [1b] para os resultados dessas etapas.")

    # ── [1b] BLOCO DE REMEDIAÇÃO DA NÃO-NORMALIDADE ──────────────────────────
    modelo_recomendado   = "OLS original"
    model_final          = model        # modelo que será usado nos demais blocos
    y_final              = y            # série que será usada nos blocos seguintes
    X_const_final        = X_const
    transformacao_usada  = None         # None | "log1p" | "sqrt"
    rlm_model            = None

    if not normalidade_ok:
        add("\n" + "─" * 62)
        add("  [1b] REMEDIAÇÃO DA NÃO-NORMALIDADE")
        add("─" * 62)

        # ── (a) Tentativa de transformações ───────────────────────────────────
        add("\n  (a) Testando transformações matemáticas na variável desfecho")
        add(f"      Distribuição original  — média = {y.mean():.4f} | "
            f"mín = {y.min():.4f} | máx = {y.max():.4f}")

        candidatos = []

        # log1p: só aplicável se todos y >= 0
        if (y >= 0).all():
            r = _tentar_transformacao(y, "log1p  [log(Y + 1)]", np.log1p)
            if r:
                candidatos.append(r)
                add(f"\n      • log1p [log(Y+1)]: stat = {r['stat']:.4f} | "
                    f"p = {r['p']:.4f} {'✔ Normal' if r['normal'] else '✘ Não-normal'}")
            else:
                add("\n      • log1p: não aplicável (valores inválidos após transformação).")
        else:
            add("\n      • log1p: não aplicável — variável contém valores negativos.")

        # sqrt: só aplicável se todos y >= 0
        if (y >= 0).all():
            r = _tentar_transformacao(y, "sqrt   [√Y]", np.sqrt)
            if r:
                candidatos.append(r)
                add(f"      • sqrt  [√Y]   : stat = {r['stat']:.4f} | "
                    f"p = {r['p']:.4f} {'✔ Normal' if r['normal'] else '✘ Não-normal'}")
            else:
                add("      • sqrt: não aplicável.")
        else:
            add("      • sqrt: não aplicável — variável contém valores negativos.")

        # Seleciona melhor transformação (maior p-valor dentre as normais)
        candidatos_normais = [c for c in candidatos if c["normal"]]
        candidatos_todos   = sorted(candidatos, key=lambda c: c["p"], reverse=True)
        melhor_transf      = candidatos_normais[0] if candidatos_normais else (
                             candidatos_todos[0]    if candidatos_todos   else None)

        model_ols_transf = None
        aic_ols_orig     = model.aic
        aic_ols_transf   = None

        if melhor_transf:
            transformacao_usada = melhor_transf["nome"].split()[0]  # "log1p" ou "sqrt"
            y_t   = melhor_transf["serie"]
            X_t   = sm.add_constant(X_scaled)
            model_ols_transf = sm.OLS(y_t, X_t).fit()
            aic_ols_transf   = model_ols_transf.aic

            add(f"\n      Melhor transformação selecionada: {melhor_transf['nome']}")
            if melhor_transf["normal"]:
                add("      ✔ Normalidade ATENDIDA após transformação.")
            else:
                add(f"      ✘ Normalidade ainda violada (p = {melhor_transf['p']:.4f}), "
                    "mas esta é a menos assimétrica disponível.")

            add(f"\n      OLS com {melhor_transf['nome']}:")
            add(f"        R²           = {model_ols_transf.rsquared:.4f}")
            add(f"        R² Ajustado  = {model_ols_transf.rsquared_adj:.4f}")
            add(f"        AIC          = {model_ols_transf.aic:.4f}")
            add(f"        BIC          = {model_ols_transf.bic:.4f}")
            add(f"        F-stat       = {model_ols_transf.fvalue:.4f}  "
                f"(p = {model_ols_transf.f_pvalue:.4f})")
            add(f"\n      ⚠ Interpretação: os coeficientes referem-se à escala "
                f"{melhor_transf['nome']}.")
            add("        Para interpretar na escala original, aplique a "
                "transformação inversa (exp ou ²).")
        else:
            add("\n      Nenhuma transformação foi aplicável para este conjunto de dados.")

        # ── (b) Regressão Robusta (RLM) ───────────────────────────────────────
        add("\n  (b) Regressão Robusta (RLM — M-estimator / Huber T)")
        add("      A regressão robusta é indicada quando a normalidade é violada")
        add("      por outliers ou caudas pesadas. Ela pondera menos as observações")
        add("      discrepantes, sem exigir normalidade dos resíduos.")

        try:
            rlm_model  = sm.RLM(y, X_const, M=sm.robust.norms.HuberT()).fit()
            add(f"\n      RLM (Huber T):")
            add(f"        Coeficientes estimados com ponderação robusta.")
            convergiu = getattr(rlm_model, 'converged', 'N/D')
            add(f"        Critério de convergência: {convergiu}")

            # RLM não produz AIC/BIC nativamente — usamos desvio residual como proxy
            rlm_scale = rlm_model.scale
            add(f"        Escala robusta dos resíduos (σ̂) = {rlm_scale:.4f}")

            add("\n      Coeficientes RLM vs OLS original:")
            add(f"        {'Variável':<25} | {'OLS':>10} | {'RLM':>10} | {'Δ':>10}")
            add("        " + "-" * 60)
            for coef in model.params.index:
                ols_c = model.params[coef]
                rlm_c = rlm_model.params.get(coef, float('nan'))
                delta = rlm_c - ols_c
                add(f"        {coef[:25]:<25} | {ols_c:>10.4f} | {rlm_c:>10.4f} | {delta:>+10.4f}")

            add("\n      ✔ RLM ajustado com sucesso.")
        except Exception as e:
            add(f"\n      ✘ Erro ao ajustar RLM: {e}")
            rlm_model = None

        # ── (c) Comparação e Recomendação ─────────────────────────────────────
        add("\n  (c) Comparação entre modelos e Recomendação")
        add("      ┌─────────────────────────────────────────────────────────┐")
        add(f"      │ OLS original           AIC = {aic_ols_orig:>10.2f}              │")
        if aic_ols_transf is not None:
            add(f"      │ OLS {melhor_transf['nome'][:15]:<15}  AIC = {aic_ols_transf:>10.2f}              │")
        if rlm_model:
            add(f"      │ RLM (Huber T)          σ̂  = {rlm_scale:>10.4f}  (sem AIC/BIC)  │")
        add("      └─────────────────────────────────────────────────────────┘")

        # Lógica de recomendação
        # Prioridade 1: OLS transformado se normalidade atendida E AIC melhor
        if (melhor_transf and melhor_transf["normal"] and
                aic_ols_transf is not None and aic_ols_transf < aic_ols_orig):
            modelo_recomendado  = f"OLS com {melhor_transf['nome']}"
            model_final         = model_ols_transf
            y_final             = melhor_transf["serie"]
            X_const_final       = sm.add_constant(X_scaled)
            add(f"\n      ✔ RECOMENDAÇÃO: OLS com {melhor_transf['nome']}")
            add("        Justificativa: normalidade restaurada e AIC inferior ao OLS original.")
            add(f"        Redução no AIC: {aic_ols_orig - aic_ols_transf:.2f} pontos.")

        # Prioridade 2: RLM se transformação não normalizou
        elif rlm_model is not None:
            modelo_recomendado = "RLM (Huber T)"
            model_final        = rlm_model
            add("\n      ✔ RECOMENDAÇÃO: Regressão Robusta (RLM — Huber T)")
            add("        Justificativa: as transformações testadas não normalizaram a VD.")
            add("        A RLM é robusta a outliers e caudas pesadas sem exigir normalidade.")
            add("        Os coeficientes são interpretados na escala original de Y.")

        # Prioridade 3: OLS transformado mesmo sem normalidade (se AIC melhor)
        elif (melhor_transf and aic_ols_transf is not None and
              aic_ols_transf < aic_ols_orig):
            modelo_recomendado  = f"OLS com {melhor_transf['nome']} (melhor AIC)"
            model_final         = model_ols_transf
            y_final             = melhor_transf["serie"]
            X_const_final       = sm.add_constant(X_scaled)
            add(f"\n      ⚠ RECOMENDAÇÃO PARCIAL: OLS com {melhor_transf['nome']}")
            add("        Justificativa: normalidade não atendida, mas AIC inferior.")
            add("        Interprete os resultados com cautela.")

        # Prioridade 4: mantém OLS original
        else:
            modelo_recomendado = "OLS original (nenhuma alternativa superior)"
            add("\n      ⚠ RECOMENDAÇÃO: OLS original mantido.")
            add("        Nenhuma transformação ou modelo alternativo produziu resultado superior.")
            add("        Considere verificar outliers, coletar mais dados ou usar GLM.")

        add(f"\n      Modelo utilizado nos resultados finais: {modelo_recomendado}")
        add("─" * 62)

    # ── [2] Multicolinearidade (VIF) ──────────────────────────────────────────
    vif = calcular_vif(X_df)
    add("\n[2] Multicolinearidade (GVIF):")
    add(f"    {'Variável':<25} {'GVIF^(1/2Df)':>14}  {'Status'}")
    add("    " + "-" * 50)
    for _, row in vif.iterrows():
        v = row["GVIF^(1/2Df)"]
        status = "✔ OK" if v < 3.16 else ("⚠ Moderado" if v < 3.87 else "✘ Alto")
        add(f"    {str(row['Variável'])[:25]:<25} {v:>14.4f}  {status}")
    add("    Referência: GVIF^(1/2Df) < 3.16 = OK | 3.16-3.87 = Moderado | > 3.87 = Problemático.")
    add("    O GVIF foi introduzido para resolver a limitação das variáveis categóricas, ")
    add("    o GVIF avalia o subconjunto de coeficientes associados a uma única variável (ou fator).")

    # ── [3] Homocedasticidade (Breusch-Pagan) ─────────────────────────────────
    # Aplica sempre sobre o modelo OLS original (BP requer OLS)
    try:
        bp_lm, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
        add(f"\n[3] Homocedasticidade (Breusch-Pagan): LM = {bp_lm:.4f} | p = {bp_p:.4f} {'✔' if bp_p > 0.05 else '✘'}")
        if bp_p > 0.05:
            add("    → Variância dos resíduos homogênea. Pressuposto ATENDIDO.")
        else:
            add("    → Heterocedasticidade detectada. Pressuposto VIOLADO.")
            add("      Considere erros padrão robustos (HC3) ou transformação de Y.")
    except Exception as e:
        add(f"\n[3] Homocedasticidade: erro no cálculo ({e}).")

    # ── [4] Independência dos Resíduos (Durbin-Watson) ────────────────────────
    dw = durbin_watson(model.resid)
    dw_ok = 1.5 < dw < 2.5
    add(f"\n[4] Independência dos Resíduos (Durbin-Watson): DW = {dw:.4f} {'✔' if dw_ok else '✘'}")
    if dw_ok:
        add("    → Sem autocorrelação detectada. Pressuposto ATENDIDO.")
    elif dw < 1.5:
        add("    → Autocorrelação positiva detectada. Pressuposto VIOLADO.")
    else:
        add("    → Autocorrelação negativa detectada. Pressuposto VIOLADO.")

    # ── [5] Análise dos Resíduos ──────────────────────────────────────────────
    resid = model.resid
    add(f"\n[5] Análise Descritiva dos Resíduos (OLS original):")
    add(f"    Média       : {resid.mean():.6f}  (esperado ≈ 0)")
    add(f"    Desvio Padr.: {resid.std():.4f}")
    add(f"    Mín / Máx   : {resid.min():.4f} / {resid.max():.4f}")
    add(f"    Assimetria  : {stats.skew(resid):.4f}")
    add(f"    Curtose     : {stats.kurtosis(resid):.4f}")
    z_resid = np.abs((resid - resid.mean()) / resid.std())
    n_outliers = (z_resid > 3).sum()
    add(f"    Outliers (|z| > 3): {n_outliers} observação(ões)")

    if not normalidade_ok and rlm_model is not None:
        resid_rlm = rlm_model.resid
        add(f"\n    Resíduos do modelo RLM (para comparação):")
        add(f"    Média       : {resid_rlm.mean():.6f}")
        add(f"    Desvio Padr.: {resid_rlm.std():.4f}")
        add(f"    Assimetria  : {stats.skew(resid_rlm):.4f}")
        add(f"    Curtose     : {stats.kurtosis(resid_rlm):.4f}")
        z_rlm = np.abs((resid_rlm - resid_rlm.mean()) / resid_rlm.std())
        add(f"    Outliers RLM (|z| > 3): {(z_rlm > 3).sum()} observação(ões)")

    # ── [6] Regularização (Lasso / Ridge) ────────────────────────────────────
    lasso = Lasso(alpha=0.1).fit(X_scaled, y)
    ridge = Ridge(alpha=1.0).fit(X_scaled, y)
    add("\n[6] Comparação de Coeficientes (Regularização):")
    add(f"    {'Variável':<20} | {'Original':>10} | {'Lasso':>10} | {'Ridge':>10}")
    add("    " + "-" * 60)
    for i, col in enumerate(X_scaled.columns):
        add(f"    {col[:20]:<20} | {model.params[col]:>10.3f} | {lasso.coef_[i]:>10.3f} | {ridge.coef_[i]:>10.3f}")

    # ── [7] Qualidade do Ajuste ───────────────────────────────────────────────
    add(f"\n[7] Qualidade do Ajuste — OLS original:")
    add(f"    R²            : {model.rsquared:.4f}")
    add(f"    R² Ajustado   : {model.rsquared_adj:.4f}")
    add(f"    AIC           : {model.aic:.4f}")
    add(f"    BIC           : {model.bic:.4f}")
    add(f"    F-stat        : {model.fvalue:.4f}  (p = {model.f_pvalue:.4f})")

    if not normalidade_ok and model_ols_transf is not None:
        add(f"\n    Qualidade do Ajuste — OLS {transformacao_usada}:")
        add(f"    R²            : {model_ols_transf.rsquared:.4f}")
        add(f"    R² Ajustado   : {model_ols_transf.rsquared_adj:.4f}")
        add(f"    AIC           : {model_ols_transf.aic:.4f}")
        add(f"    BIC           : {model_ols_transf.bic:.4f}")
        add(f"    F-stat        : {model_ols_transf.fvalue:.4f}  (p = {model_ols_transf.f_pvalue:.4f})")

    # ── [8] Sumário do modelo recomendado ─────────────────────────────────────
    add("\n" + "=" * 62)
    add(f"  RESULTADO FINAL — {modelo_recomendado.upper()}")
    add("=" * 62)
    add(model_final.summary().as_text())

    # Se RLM foi recomendado, também mostra o OLS original para referência
    if modelo_recomendado.startswith("RLM"):
        add("\n" + "─" * 62)
        add("  REFERÊNCIA — OLS ORIGINAL (para comparação)")
        add("─" * 62)
        add(model.summary().as_text())

    texto_final = "\n".join(linhas)

    # ── Saída para o app (Tkinter) ────────────────────────────────────────────
    if app:
        try:
            if hasattr(app, 'txt_resultado') and app.txt_resultado.winfo_exists():
                app._escrever(texto_final, limpar=True)
                app.update_idletasks()
                app.update()
                # Plota diagnóstico do modelo final escolhido
                app.after(200, lambda: plotar_diagnostico_2x2(
                    y_final,
                    model_final.fittedvalues if hasattr(model_final, 'fittedvalues') else model_final.predict(),
                    model_final.resid,
                    model_final,
                    f"Regressão Linear — {modelo_recomendado}",
                    "linear"
                ))
                return texto_final
        except Exception:
            pass

    plotar_diagnostico_2x2(
        y_final,
        model_final.fittedvalues if hasattr(model_final, 'fittedvalues') else model_final.predict(),
        model_final.resid,
        model_final,
        f"Regressão Linear — {modelo_recomendado}",
        "linear"
    )
    return texto_final

def analisar_binaria(df, y_col, x_cols, app=None):
    linhas = []
    add = linhas.append

    # ── 1. Preparação dos Dados ───────────────────────────────────────────────
    y_raw = df[y_col].dropna()
    X_df  = df[x_cols].loc[y_raw.index].dropna()
    y = (
        (y_raw == sorted(y_raw.unique())[1]).astype(int)
        if not pd.api.types.is_numeric_dtype(y_raw)
        else y_raw.astype(int)
    )
    y = y.loc[X_df.index]

    X_enc    = pd.get_dummies(X_df, drop_first=True).astype(float)
    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X_enc),
        columns=X_enc.columns, index=X_enc.index
    )
    X_const = sm.add_constant(X_scaled)

    # ── 2. Ajuste do Modelo Logístico ─────────────────────────────────────────
    model   = sm.Logit(y, X_const).fit(disp=False)
    y_pred  = model.predict(X_const)
    residuos = y.values - y_pred.values

    add("=" * 62)
    add("  ANÁLISE COMPLETA — REGRESSÃO LOGÍSTICA BINÁRIA")
    add("=" * 62)

    # ── [1] Multicolinearidade (VIF) ──────────────────────────────────────────
    vif = calcular_vif(X_df)
    add("\n[2] Multicolinearidade (GVIF):")
    add(f"    {'Variável':<25} {'GVIF^(1/2Df)':>14}  {'Status'}")
    add("    " + "-" * 50)
    for _, row in vif.iterrows():
        v = row["GVIF^(1/2Df)"]
        status = "✔ OK" if v < 3.16 else ("⚠ Moderado" if v < 3.87 else "✘ Alto")
        add(f"    {str(row['Variável'])[:25]:<25} {v:>14.4f}  {status}")
    add("    Referência: GVIF^(1/2Df) < 3.16 = OK | 3.16-3.87 = Moderado | > 3.87 = Problemático.")
    add("    O GVIF foi introduzido para resolver a limitação das variáveis categóricas, ")
    add("    o GVIF avalia o subconjunto de coeficientes associados a uma única variável (ou fator).")
    
    # ── [2] Separação Perfeita (Hosmer-Lemeshow simplificado) ─────────────────
    # Verifica se o modelo convergiu
    add(f"\n[2] Convergência do Modelo: {'✔ Convergiu' if model.mle_retvals.get('converged', True) else '✘ Não convergiu'}")

    # ── [3] Independência dos Resíduos (Durbin-Watson sobre resíduos de Pearson) ──
    pearson_resid = (y.values - y_pred.values) / np.sqrt(y_pred.values * (1 - y_pred.values) + 1e-10)
    dw = durbin_watson(pearson_resid)
    dw_ok = 1.5 < dw < 2.5
    add(f"\n[3] Independência dos Resíduos (Durbin-Watson): DW = {dw:.4f} {'✔' if dw_ok else '✘'}")
    if not dw_ok:
        add("    → Possível autocorrelação. Avalie se as observações são independentes.")

    # ── [4] Tamanho amostral por preditor ─────────────────────────────────────
    n_eventos = y.sum()
    n_preditores = X_scaled.shape[1]
    epp = n_eventos / n_preditores if n_preditores > 0 else np.inf
    add(f"\n[4] Eventos por Preditor (EPP): {epp:.1f} {'✔' if epp >= 10 else '✘ Amostral insuficiente'}")
    add(f"    Eventos = {n_eventos} | Preditores = {n_preditores}")
    add(f"    Referência: mínimo de 10 EPP recomendado.")

    # ── [5] Análise dos Resíduos de Pearson ───────────────────────────────────
    add(f"\n[5] Análise dos Resíduos de Pearson:")
    add(f"    Média       : {pearson_resid.mean():.6f}  (esperado ≈ 0)")
    add(f"    Desvio Padr.: {pearson_resid.std():.4f}")
    add(f"    Mín / Máx   : {pearson_resid.min():.4f} / {pearson_resid.max():.4f}")
    n_out = (np.abs(pearson_resid) > 3).sum()
    add(f"    Outliers (|z| > 3): {n_out} observação(ões)")

    # ── [6] Distância de Cook e Influência ────────────────────────────────────
    try:
        influence = model.get_influence()
        (cook, _) = influence.cooks_distance
        limiar_cook = 4 / len(y)
        n_influentes = (cook > limiar_cook).sum()
        add(f"\n[6] Pontos Influentes (Cook > 4/n = {limiar_cook:.4f}): {n_influentes} observação(ões)")
        if n_influentes > 0:
            add("    → Avalie a exclusão de outliers extremos e re-ajuste o modelo.")
    except:
        add("\n[6] Distância de Cook: não disponível para este modelo.")

    # ── [7] Qualidade do Ajuste ───────────────────────────────────────────────
    # Pseudo-R²
    ll_null  = model.llnull
    ll_model = model.llf
    mcf_r2   = 1 - (ll_model / ll_null)
    nagelkerke_r2 = (1 - np.exp((ll_null - ll_model) * 2 / len(y))) / (1 - np.exp(ll_null * 2 / len(y)))

    # AUC
    fpr, tpr, _ = roc_curve(y.values, y_pred.values)
    roc_auc = auc(fpr, tpr)

    # Acurácia
    y_class = (y_pred > 0.5).astype(int)
    acuracia = (y_class == y.values).mean()

    add(f"\n[7] Qualidade do Ajuste:")
    add(f"    Log-Likelihood   : {ll_model:.4f}")
    add(f"    AIC              : {model.aic:.4f}")
    add(f"    BIC              : {model.bic:.4f}")
    add(f"    Pseudo-R² (McF.) : {mcf_r2:.4f}")
    add(f"    Pseudo-R² (Nag.) : {nagelkerke_r2:.4f}")
    add(f"    AUC-ROC          : {roc_auc:.4f} {'✔' if roc_auc >= 0.7 else '⚠ Discriminação fraca'}")
    add(f"    Acurácia (0.5)   : {acuracia:.4f}")

    # ── [8] Odds Ratios com Intervalo de Confiança ────────────────────────────
    add(f"\n[8] Odds Ratios (exp(β)) com IC 95%:")
    ci = model.conf_int()
    or_df = pd.DataFrame({
        "OR"     : np.exp(model.params),
        "IC 2.5%": np.exp(ci[0]),
        "IC 97.5%": np.exp(ci[1]),
        "p-valor": model.pvalues
    })
    add(or_df.to_string())

    add("\n" + "=" * 62)
    add(model.summary().as_text())

    # ── Exibição: texto ANTES da figura ──────────────────────────────────────
    texto_final = "\n".join(linhas)

    if app:
        try:
            if hasattr(app, 'txt_resultado') and app.txt_resultado.winfo_exists():
                app._escrever(texto_final, limpar=True)
                app.update_idletasks()
                app.update()
                app.after(200, lambda: plotar_diagnostico_2x2(
                    y, y_pred, residuos, model, "Logística Binária", "binaria"
                ))
                return texto_final
        except:
            pass

    # Fallback sem interface
    plotar_diagnostico_2x2(
    y_real=y, 
    y_pred=y_pred, 
    residuos=residuos, 
    model=model, 
    titulo="Logística Binária", 
    tipo="binaria"
    )
    return texto_final


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISE DE CONTAGEM — SELEÇÃO AUTOMÁTICA DE MODELO
# ─────────────────────────────────────────────────────────────────────────────

def analisar_contagem(df, y_col, x_cols, norm_y=False, app=None):
    """
    Ajusta Poisson primeiro, avalia superdispersão via Cameron-Trivedi e
    excesso de zeros. Seleciona automaticamente entre:
      • Poisson           — equidispersão, sem excesso de zeros
      • Quasi-Poisson     — superdispersão moderada (via statsmodels GLM com family=Poisson + scale)
      • Binomial Negativa — superdispersão acentuada ou confirmada
    Todos os blocos de resultados originais são mantidos para os três modelos.
    """
    linhas = []
    add = linhas.append

    # ── 1. Preparação dos Dados ───────────────────────────────────────────────
    y    = df[y_col].dropna().astype(int)
    X_df = df[x_cols].loc[y.index].dropna()
    y    = y.loc[X_df.index]

    norm_x = app.flag_norm.get() if (app and hasattr(app, 'flag_norm')) else False
    X_enc = pd.get_dummies(X_df, drop_first=True).astype(float)
    if norm_x:
        X_scaled = pd.DataFrame(
            StandardScaler().fit_transform(X_enc),
            columns=X_enc.columns, index=X_enc.index
        )
        add("[!] Z-Score aplicado nos preditores (X).")
    else:
        X_scaled = X_enc.copy()

    X_const = sm.add_constant(X_scaled)

    # ── 2. FASE DE DIAGNÓSTICO PRELIMINAR (sempre com Poisson) ───────────────
    poisson_model = sm.Poisson(y, X_const).fit(disp=False)
    mu_poisson    = poisson_model.predict()

    # Razão Var/Média
    ratio = teste_equidispersao(y, mu_poisson)

    # Teste Cameron-Trivedi
    try:
        mu_arr = mu_poisson
        aux    = ((y - mu_arr) ** 2 - mu_arr) / mu_arr
        ct_model  = sm.OLS(aux, mu_arr).fit()
        alpha_ct  = ct_model.params[0]
        p_ct      = ct_model.pvalues[0]
        ct_ok     = p_ct > 0.05        # p > 0.05 → equidisperso
    except:
        alpha_ct, p_ct, ct_ok = np.nan, np.nan, True

    # Excesso de zeros
    prop_obs, prop_esp = excesso_de_zeros(y)
    excesso_zeros = prop_obs > prop_esp * 1.5

    # ── 3. DECISÃO DE MODELO ──────────────────────────────────────────────────
    #   Regra: superdispersão confirmada (ratio > 1.2 E p_ct < 0.05)?
    #     → BN se ratio > 2 ou alpha_ct > 1; senão Quasi-Poisson.
    #   Caso contrário: Poisson.
    superdispersao = (ratio > 1.2) and (not ct_ok)

    if not superdispersao:
        modelo_escolhido = "Poisson"
    elif ratio > 2.0 or (not np.isnan(alpha_ct) and alpha_ct > 1.0):
        modelo_escolhido = "Binomial Negativa"
    else:
        modelo_escolhido = "Quasi-Poisson"

    add("=" * 62)
    add("  ANÁLISE COMPLETA — REGRESSÃO DE CONTAGEM")
    add("  SELEÇÃO AUTOMÁTICA DE MODELO")
    add("=" * 62)

    # ── [1] Verificação da Variável Desfecho ──────────────────────────────────
    n_neg  = (y < 0).sum()
    n_frac = (~(y == y.astype(int))).sum()
    add(f"\n[1] Verificação da Variável Desfecho:")
    add(f"    Valores negativos   : {n_neg}  {'✔' if n_neg == 0 else '✘ Não esperado em modelos de contagem'}")
    add(f"    Valores não-inteiros: {n_frac}  {'✔' if n_frac == 0 else '✘ Verifique os dados'}")
    add(f"    Média     : {y.mean():.4f}")
    add(f"    Variância : {y.var():.4f}")
    add(f"    Mín / Máx : {y.min()} / {y.max()}")

    # ── [2] Equidispersão ─────────────────────────────────────────────────────
    add(f"\n[2] Equidispersão (Var/Média = {ratio:.4f}):")
    if 0.8 <= ratio <= 1.2:
        add(f"    ✔ Próximo de 1 — Equidispersão. Poisson é adequado.")
    elif ratio > 1.2:
        add(f"    ✘ Superdispersão detectada (Var/Média = {ratio:.2f}).")
    else:
        add(f"    ⚠ Subdispersão detectada (Var/Média < 1).")

    if not np.isnan(p_ct):
        add(f"    Teste Cameron-Trivedi: α = {alpha_ct:.4f} | p = {p_ct:.4f} "
            f"{'✔ Equidisperso' if ct_ok else '✘ Superdispersão confirmada'}")
    else:
        add("    Teste Cameron-Trivedi: não calculado.")

    # ── [3] Excesso de Zeros ──────────────────────────────────────────────────
    add(f"\n[3] Excesso de Zeros:")
    add(f"    Proporção observada         : {prop_obs:.4f} ({prop_obs*100:.1f}%)")
    add(f"    Proporção esperada (Poisson): {prop_esp:.4f} ({prop_esp*100:.1f}%)")
    if excesso_zeros:
        add("    ✘ Excesso de zeros detectado. Considere Zero-Inflated Poisson (ZIP) se persistir.")
    else:
        add("    ✔ Proporção de zeros compatível com Poisson.")

    # ── [4] DECISÃO E AJUSTE DO MODELO SELECIONADO ───────────────────────────
    add(f"\n{'='*62}")
    add(f"  MODELO SELECIONADO: {modelo_escolhido}")
    add(f"{'='*62}")

    if modelo_escolhido == "Poisson":
        add("  Critério: Var/Média ≈ 1 e teste de Cameron-Trivedi não significativo.")
        add("  Modelo de Poisson padrão é adequado.")
        model    = poisson_model
        y_pred   = mu_poisson
        nome_plot = "Poisson"

    elif modelo_escolhido == "Quasi-Poisson":
        add("  Critério: Superdispersão moderada (Var/Média entre 1.2 e 2.0).")
        add("  Quasi-Poisson corrige os erros-padrão pelo fator de escala φ.")
        # GLM Poisson com escala estimada (equivalente ao Quasi-Poisson)
        glm_pois  = sm.GLM(y, X_const, family=sm.families.Poisson()).fit(scale='X2')
        model     = glm_pois
        y_pred    = glm_pois.predict()
        nome_plot = "Quasi-Poisson"
        add(f"  Fator de escala φ estimado: {glm_pois.scale:.4f}")

    else:  # Binomial Negativa
        add("  Critério: Superdispersão acentuada (Var/Média > 2 ou α > 1).")
        add("  Binomial Negativa modela a superdispersão com parâmetro de dispersão α.")
        try:
            nb_model  = sm.NegativeBinomial(y, X_const).fit(disp=False, maxiter=100)
            model     = nb_model
            y_pred    = nb_model.predict()
            nome_plot = "Binomial Negativa"
            add(f"  Parâmetro de dispersão α: {nb_model.params.get('alpha', np.nan):.4f}")
        except Exception as e:
            add(f"  ⚠ Binomial Negativa falhou ({e}). Usando Quasi-Poisson como fallback.")
            glm_pois  = sm.GLM(y, X_const, family=sm.families.Poisson()).fit(scale='X2')
            model     = glm_pois
            y_pred    = glm_pois.predict()
            nome_plot = "Quasi-Poisson (fallback)"
            modelo_escolhido = "Quasi-Poisson"

    # ── Resíduos do modelo selecionado ────────────────────────────────────────
    y_pred_arr       = np.array(y_pred)
    residuos_brutos  = y - y_pred_arr
    residuos_pearson = residuos_brutos / np.sqrt(y_pred_arr + 1e-10)
    df_resid         = max(len(y) - X_const.shape[1], 1)

    # ── [4b] Comparação dos três modelos (AIC/BIC/Deviance) ──────────────────
    add(f"\n[4] Comparação de Modelos (referência Poisson):")
    add(f"    {'Modelo':<22} {'AIC':>10} {'BIC':>10} {'Log-Lik':>12} {'Deviance':>10}")
    add("    " + "-" * 68)

    # Poisson (sempre disponível)
    try:
        add(f"    {'Poisson':<22} {poisson_model.aic:>10.2f} {poisson_model.bic:>10.2f} "
            f"{poisson_model.llf:>12.2f} {poisson_model.deviance:>10.2f}")
    except:
        add(f"    {'Poisson':<22} {'N/D':>10} {'N/D':>10} {'N/D':>12} {'N/D':>10}")

    # Quasi-Poisson (não tem AIC/BIC formal, apresenta φ)
    try:
        glm_ref = sm.GLM(y, X_const, family=sm.families.Poisson()).fit(scale='X2')
        add(f"    {'Quasi-Poisson':<22} {'N/A*':>10} {'N/A*':>10} "
            f"{'N/A*':>12} {glm_ref.deviance:>10.2f}  (* φ = {glm_ref.scale:.3f})")
    except:
        add(f"    {'Quasi-Poisson':<22} {'N/D':>10}")

    # Binomial Negativa
    try:
        nb_ref = sm.NegativeBinomial(y, X_const).fit(disp=False, maxiter=100)
        add(f"    {'Binomial Negativa':<22} {nb_ref.aic:>10.2f} {nb_ref.bic:>10.2f} "
            f"{nb_ref.llf:>12.2f} {'N/A':>10}")
    except:
        add(f"    {'Binomial Negativa':<22} {'Não convergiu':>10}")

    add(f"\n  ► Modelo utilizado para os resultados abaixo: {modelo_escolhido}")

    # ── [5] Multicolinearidade (VIF) ──────────────────────────────────────────
    vif = calcular_vif(X_df)
    add("\n[2] Multicolinearidade (GVIF):")
    add(f"    {'Variável':<25} {'GVIF^(1/2Df)':>14}  {'Status'}")
    add("    " + "-" * 50)
    for _, row in vif.iterrows():
        v = row["GVIF^(1/2Df)"]
        status = "✔ OK" if v < 3.16 else ("⚠ Moderado" if v < 3.87 else "✘ Alto")
        add(f"    {str(row['Variável'])[:25]:<25} {v:>14.4f}  {status}")
    add("    Referência: GVIF^(1/2Df) < 3.16 = OK | 3.16-3.87 = Moderado | > 3.87 = Problemático.")
    add("    O GVIF foi introduzido para resolver a limitação das variáveis categóricas, ")
    add("    o GVIF avalia o subconjunto de coeficientes associados a uma única variável (ou fator).")
    



    # ── [6] Independência dos Resíduos (Durbin-Watson) ────────────────────────
    dw = durbin_watson(residuos_pearson)
    dw_ok = 1.5 < dw < 2.5
    add(f"\n[6] Independência dos Resíduos (Durbin-Watson): DW = {dw:.4f} {'✔' if dw_ok else '✘'}")
    if not dw_ok:
        add("    → Possível autocorrelação nos resíduos.")

    # ── [7] Análise Descritiva dos Resíduos de Pearson ───────────────────────
    add(f"\n[7] Análise dos Resíduos de Pearson ({modelo_escolhido}):")
    add(f"    Média       : {residuos_pearson.mean():.6f}  (esperado ≈ 0)")
    add(f"    Desvio Padr.: {residuos_pearson.std():.4f}")
    add(f"    Mín / Máx   : {residuos_pearson.min():.4f} / {residuos_pearson.max():.4f}")
    add(f"    Assimetria  : {stats.skew(residuos_pearson):.4f}")
    add(f"    Curtose     : {stats.kurtosis(residuos_pearson):.4f}")
    n_out = (np.abs(residuos_pearson) > 3).sum()
    add(f"    Outliers (|z| > 3): {n_out} observação(ões)")

    # Qui-quadrado de Pearson (bondade de ajuste)
    chi2_pearson = np.sum(residuos_pearson ** 2)
    p_chi2 = 1 - stats.chi2.cdf(chi2_pearson, df_resid)
    add(f"\n    Qui-quadrado de Pearson: χ² = {chi2_pearson:.4f} | df = {df_resid} | p = {p_chi2:.4f}")
    add(f"    {'✔ Ajuste adequado.' if p_chi2 > 0.05 else '✘ Ajuste inadequado — verifique superdispersão ou zeros.'}")

    # ── [8] Qualidade do Ajuste ───────────────────────────────────────────────
    add(f"\n[8] Qualidade do Ajuste ({modelo_escolhido}):")
    try:
        add(f"    Log-Likelihood  : {model.llf:.4f}")
    except:
        add(f"    Log-Likelihood  : N/A (Quasi)")
    try:
        add(f"    AIC             : {model.aic:.4f}")
        add(f"    BIC             : {model.bic:.4f}")
    except:
        add(f"    AIC / BIC       : N/A (Quasi-Poisson não possui AIC formal)")
    try:
        pseudo_r2 = 1 - (model.llf / poisson_model.llnull)
        add(f"    Pseudo-R² (McF.): {pseudo_r2:.4f}")
    except:
        pass
    try:
        add(f"    Deviance        : {model.deviance:.4f}")
        add(f"    Deviance/df     : {model.deviance / df_resid:.4f}  "
            f"{'✔' if model.deviance / df_resid < 1.5 else '✘ Superdispersão residual'}")
    except:
        pass
    try:
        add(f"    Pseudo-R² (McF.): {1 - (poisson_model.llf / poisson_model.llnull):.4f}  (ref. Poisson)")
    except:
        pass

    # ── [9] Coeficientes com IRR (Incidence Rate Ratios) ─────────────────────
    add(f"\n[9] Incidence Rate Ratios — IRR = exp(β) com IC 95% ({modelo_escolhido}):")
    try:
        ci = model.conf_int()
        irr_df = pd.DataFrame({
            "IRR"     : np.exp(model.params),
            "IC 2.5%" : np.exp(ci.iloc[:, 0]),
            "IC 97.5%": np.exp(ci.iloc[:, 1]),
            "p-valor" : model.pvalues
        })
        # Remove linha 'alpha' da BN da tabela IRR (não é um preditor)
        irr_df = irr_df[~irr_df.index.str.lower().isin(['alpha'])]
        add(irr_df.to_string())
    except Exception as e:
        add(f"    IRR não calculado: {e}")

    add("\n" + "=" * 62)
    add(f"  RESUMO ESTATÍSTICO COMPLETO — {modelo_escolhido}")
    add("=" * 62)
    add(model.summary().as_text())

    # ── Exibição: texto ANTES da figura ──────────────────────────────────────
    texto_final = "\n".join(linhas)

    if app:
        try:
            if hasattr(app, 'txt_resultado') and app.txt_resultado.winfo_exists():
                app._escrever(texto_final, limpar=True)
                app.update_idletasks()
                app.update()
                app.after(200, lambda: plotar_diagnostico_2x2(
                    y, y_pred_arr, residuos_pearson, model,
                    f"Regressão de Contagem — {nome_plot}", "contagem"
                ))
                return texto_final
        except:
            pass
    
    plotar_diagnostico_2x2(
        y_real=y, 
        y_pred=y_pred_arr, 
        residuos=residuos_pearson, 
        model=model, 
        titulo=f"Regressão de Contagem — {nome_plot}", 
        tipo="contagem"
    )

    return texto_final


# ── FUNÇÃO CATEGÓRICA ─────────────────────────────────────────────────────────

def analisar_categorica(df, y_col, x_cols, app=None):
    linhas = []
    add = linhas.append

    try:
        y_raw = df[y_col].dropna()
        X_df  = df[x_cols].loc[y_raw.index].dropna()
        idx   = y_raw.index.intersection(X_df.index)
        y_raw, X_df = y_raw.loc[idx], X_df.loc[idx]

        all_cats     = sorted(y_raw.unique())
        ref_cat      = all_cats[0]
        outras       = [c for c in all_cats if c != ref_cat]
        cats_ordenadas = [ref_cat] + list(outras)

        y_enc = pd.Categorical(y_raw, categories=cats_ordenadas).codes
        X_enc = pd.get_dummies(X_df, drop_first=True).astype(float)

        norm_x = app.flag_norm.get() if (app and hasattr(app, 'flag_norm')) else False
        if norm_x:
            X_enc = pd.DataFrame(
                StandardScaler().fit_transform(X_enc),
                columns=X_enc.columns, index=X_enc.index
            )
            add("[!] Z-Score aplicado nos preditores (X).")

        X_const = sm.add_constant(X_enc)

        add("=" * 62)
        add("  RESULTADOS — REGRESSÃO LOGÍSTICA MULTINOMIAL")
        add("=" * 62)
        add(f"  Referência (Baseline): '{ref_cat}'")
        add(f"  Comparando as outras categorias contra '{ref_cat}'\n")

        model = MNLogit(y_enc, X_const).fit(method='bfgs', maxiter=100, disp=False)

        add("[ TABELA DE INTERPRETAÇÃO: RRR (exp(coef)) ]")
        params = model.params
        conf   = model.conf_int()

        for i in range(params.shape[1]):
            cat_label = cats_ordenadas[i + 1]
            add(f"\n--- Comparação: '{cat_label}' vs '{ref_cat}' ---")
            rrr_vals = np.exp(params.iloc[:, i])
            ic_low   = np.exp(conf.iloc[i * len(params):(i + 1) * len(params), 0].values)
            ic_high  = np.exp(conf.iloc[i * len(params):(i + 1) * len(params), 1].values)
            p_vals   = model.pvalues.iloc[:, i]
            rrr_df = pd.DataFrame({
                "RRR (Odds)": rrr_vals,
                "IC 2.5%"   : ic_low,
                "IC 97.5%"  : ic_high,
                "p-valor"   : p_vals
            })
            add(rrr_df.to_string())

        add("\n" + "-" * 62)
        add("Resumo Estatístico Completo (Escala Logarítmica):")
        add(model.summary().as_text())

    except Exception as e:
        add("\n" + "!" * 60)
        add(f"  ✘ ERRO NO AJUSTE: {str(e)}")
        add("!" * 60)

    texto_final = "\n".join(linhas)

    if app:
        try:
            if hasattr(app, 'txt_resultado') and app.txt_resultado.winfo_exists():
                app._escrever(texto_final, limpar=True)
                app.update_idletasks()
                app.update()
                if 'model' in locals():
                    app.after(200, lambda: plotar_diagnostico_multinomial_completo(
                        y_enc, model.predict(X_const).values, "Multinomial"
                    ))
                return texto_final
        except:
            pass

    if 'model' in locals():
        plotar_diagnostico_multinomial_completo(
            y_real=y_enc, 
            y_prob=model.predict(X_const).values, 
            titulo="Multinomial - Análise de Metodologia"
        )

    return texto_final


# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE GRÁFICA
# ─────────────────────────────────────────────────────────────────────────────

"""
Interface gráfica — customtkinter
Todas as funcionalidades da versão tkinter foram preservadas.

Dependências:
    pip install customtkinter pandas openpyxl matplotlib statsmodels scikit-learn scipy
"""


# ── Importações do restante do projeto (assumidas já existentes) ──────────────
# from analise import (
#     identificar_tipo_variavel, analisar_continua, analisar_binaria,
#     analisar_contagem, analisar_categorica, exibir_heatmap,
#     plotar_diagnostico_2x2
# )

# ─────────────────────────────────────────────────────────────────────────────
# TEMA GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# Paleta de cores (usada em widgets que aceitam hex direto)
BG_DARK   = "#1e1e2e"
BG_MID    = "#2a2a3e"
BG_CARD   = "#313145"
ACCENT    = "#7c6af7"
ACCENT2   = "#56cfb2"
TEXT_MAIN = "#e0e0f0"
TEXT_DIM  = "#8888aa"
DANGER    = "#e05c5c"
BORDER    = "#44446a"



# ─────────────────────────────────────────────────────────────────────────────
# CLASSE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Análise de Regressão Automática v1.04 - By Eduardo Borba Neves")
        self.geometry("1280x820")
        self.minsize(1100, 700)
        self.configure(fg_color=BG_DARK)

        self.df         = None
        self.var_checks = {}          # {col_name: BooleanVar}

        # Flags de controle (mesmo contrato da versão tkinter)
        self.var_norm_y = BooleanVar(value=False)
        self.flag_norm  = BooleanVar(value=False)
        self.flag_heat  = BooleanVar(value=False)

        self._build_ui()

    # ── CONSTRUÇÃO DA INTERFACE ───────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._build_sel_frame()
        self._build_painel()

    # ── Cabeçalho ─────────────────────────────────────────────────────────────
    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=BG_MID, corner_radius=0, height=56)
        header.pack(fill="x")
        header.pack_propagate(False)

        # Título
        ctk.CTkLabel(
            header,
            text="📊  Análise de Regressão Automática - By Eduardo Borba Neves",
            font=ctk.CTkFont("Segoe UI", 16, "bold"),
            text_color=TEXT_MAIN,
        ).pack(side="left", padx=20, pady=10)

        # Botão Fechar
        ctk.CTkButton(
            header, text="✕  Fechar",
            fg_color=DANGER, hover_color="#b84040",
            text_color="white", width=100, height=34,
            corner_radius=6, command=self.sair_seguro
        ).pack(side="right", padx=20, pady=10)

        # Botão Exportar
        ctk.CTkButton(
            header, text="💾 Exportar Relatório",
            fg_color=BG_CARD, hover_color=BORDER,
            text_color=TEXT_MAIN, width=160, height=34,
            corner_radius=6, command=self._exportar
        ).pack(side="right", padx=8, pady=10)

        # Botão Carregar
        ctk.CTkButton(
            header, text="📂 Carregar Planilha",
            fg_color=ACCENT, hover_color="#5a4ed1",
            text_color="white", width=160, height=34,
            corner_radius=6, command=self._carregar_arquivo
        ).pack(side="right", padx=8, pady=10)

        # Label arquivo
        self.lbl_arquivo = ctk.CTkLabel(
            header, text="Aguardando arquivo...",
            text_color=TEXT_DIM, font=ctk.CTkFont("Segoe UI", 9)
        )
        self.lbl_arquivo.pack(side="right", padx=10)

    # ── Barra de seleção da VD ────────────────────────────────────────────────
    def _build_sel_frame(self):
        sel = ctk.CTkFrame(self, fg_color=BG_DARK, corner_radius=0)
        sel.pack(fill="x", padx=20, pady=(10, 2))

        ctk.CTkLabel(
            sel, text="Y:", font=ctk.CTkFont("Segoe UI", 12, "bold"),
            text_color=TEXT_MAIN
        ).pack(side="left")

        # Combobox da variável desfecho
        self.combo_desfecho = ctk.CTkComboBox(
            sel, values=[], width=220,
            state="disabled",
            fg_color=BG_MID, border_color=BORDER,
            button_color=ACCENT, button_hover_color="#5a4ed1",
            dropdown_fg_color=BG_MID,
            text_color=TEXT_MAIN,
            command=self._atualizar_preditores   # chamado ao selecionar
        )
        self.combo_desfecho.pack(side="left", padx=8)

        # ── Norm Y ────────────────────────────────────────────────────────────
        self.check_norm_y = ctk.CTkCheckBox(
            sel, text="Norm. Y",
            variable=self.var_norm_y,
            fg_color=ACCENT, hover_color="#5a4ed1",
            text_color=TEXT_MAIN, state="disabled"
        )
        self.check_norm_y.pack(side="left", padx=12)

        # ── Norm X ────────────────────────────────────────────────────────────
        self.check_norm_x = ctk.CTkCheckBox(
            sel, text="Norm. X (Z-Score)",
            variable=self.flag_norm,
            fg_color=ACCENT, hover_color="#5a4ed1",
            text_color=TEXT_MAIN, state="disabled"
        )
        self.check_norm_x.pack(side="left", padx=12)

        # ── Heatmap ───────────────────────────────────────────────────────────
        self.check_heat = ctk.CTkCheckBox(
            sel, text="Exibir Heatmap",
            variable=self.flag_heat,
            fg_color=ACCENT, hover_color="#5a4ed1",
            text_color=TEXT_MAIN, state="disabled"
        )
        self.check_heat.pack(side="left", padx=12)

        # Label do tipo detectado
        self.lbl_tipo = ctk.CTkLabel(
            sel, text="",
            text_color=ACCENT2, font=ctk.CTkFont(family="Segoe UI", size=12, slant="italic")
        )
        self.lbl_tipo.pack(side="left", padx=10)

    # ── Painel principal (esquerda: checkboxes | direita: resultados) ─────────
    def _build_painel(self):
        painel = ctk.CTkFrame(self, fg_color=BG_DARK, corner_radius=0)
        painel.pack(fill="both", expand=True, padx=15, pady=(4, 12))

        # ── Frame esquerdo ────────────────────────────────────────────────────
        left = ctk.CTkFrame(painel, fg_color=BG_CARD, corner_radius=10, width=290)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        ctk.CTkLabel(
            left, text="Variáveis Preditoras (X)",
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
            text_color=TEXT_MAIN
        ).pack(pady=(12, 4), padx=12)

        # ScrollableFrame para os checkboxes
        self.check_scroll = ctk.CTkScrollableFrame(
            left, fg_color=BG_CARD, corner_radius=0
        )
        self.check_scroll.pack(fill="both", expand=True, padx=6, pady=4)

        # Botão Rodar
        self.btn_rodar = ctk.CTkButton(
            left, text="▶  RODAR ANÁLISE",
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
            fg_color=ACCENT2, hover_color="#3db89a",
            text_color=BG_DARK,
            height=44, corner_radius=8,
            state="disabled", command=self._rodar_analise
        )
        self.btn_rodar.pack(fill="x", padx=15, pady=14)

        # ── Frame direito ─────────────────────────────────────────────────────
        right = ctk.CTkFrame(painel, fg_color=BG_CARD, corner_radius=10)
        right.pack(side="left", fill="both", expand=True)

        ctk.CTkLabel(
            right, text="Resultados da Análise",
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
            text_color=TEXT_MAIN
        ).pack(pady=(12, 4), padx=12, anchor="w")

        # Caixa de texto com scroll (CTkTextbox substitui ScrolledText)
        self.txt_resultado = ctk.CTkTextbox(
            right,
            fg_color="#12121e",
            text_color=TEXT_MAIN,
            font=ctk.CTkFont("Courier New", 12),
            corner_radius=6,
            wrap="none",
            state="disabled"
        )

        self.txt_resultado.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # ── AÇÕES ─────────────────────────────────────────────────────────────────


    def _carregar_arquivo(self):
        path = filedialog.askopenfilename(
            title="Selecione a planilha",
            filetypes=[("Excel", "*.xlsx *.xls"), ("Todos", "*.*")]
        )
        if not path:
            return
        try:
            self.df = pd.read_excel(path)

            # Corrige colunas com vírgula como separador decimal
            for col in self.df.columns:
                if self.df[col].dtype == object:
                    tentativa = (
                        self.df[col]
                        .astype(str)
                        .str.replace(",", ".", regex=False)
                    )
                    convertida = pd.to_numeric(tentativa, errors="coerce")
                    # Só substitui se a conversão funcionou para a maioria dos valores
                    if convertida.notna().mean() >= 0.8:
                        self.df[col] = convertida

            cols = list(self.df.columns)
            self.combo_desfecho.configure(values=cols, state="readonly")
            self.combo_desfecho.set(cols[0])
            nome = os.path.basename(path)
            self.lbl_arquivo.configure(
                text=f"📄 {nome}  ({self.df.shape[0]} × {self.df.shape[1]})",
                text_color=ACCENT2
            )
            self._atualizar_preditores()
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível carregar o arquivo:\n{e}")

            


    def _atualizar_preditores(self, valor=None):
        """
        Atualiza checkboxes de preditores e estado dos controles
        conforme o tipo da variável desfecho selecionada.

        Regras de habilitação:
          • Norm Y  → apenas para 'continua' e 'contagem'
          • Norm X  → sempre habilitado após carregar arquivo
          • Heatmap → sempre habilitado após carregar arquivo
        """
        if self.df is None:
            return

        desfecho = self.combo_desfecho.get()
        tipo = identificar_tipo_variavel(self.df[desfecho])

        labels = {
            "continua":   "🔵 Contínua  → Regressão Linear",
            "binaria":    "🟢 Binária   → Logística Binária",
            "categorica": "🟣 Categórica → Log. Multinomial",
            "contagem":   "🟠 Contagem  → Poisson / Neg. Bin.",
        }
        self.lbl_tipo.configure(text=labels.get(tipo, tipo))

        # Norm Y: só faz sentido para contínua e contagem
        self.check_norm_y.configure(
            state="normal" if tipo in ["continua", "contagem"] else "disabled"
        )
        # Norm X e Heatmap: sempre disponíveis
        self.check_norm_x.configure(state="normal")
        self.check_heat.configure(state="normal")

        # Limpa checkboxes anteriores
        for w in self.check_scroll.winfo_children():
            w.destroy()
        self.var_checks = {}

        for col in self.df.columns:
            if col == desfecho:
                continue
            var = BooleanVar(value=False)
            self.var_checks[col] = var
            cb = ctk.CTkCheckBox(
                self.check_scroll,
                text=col,
                variable=var,
                fg_color=ACCENT, hover_color="#5a4ed1",
                text_color=TEXT_MAIN,
                font=ctk.CTkFont(family="Segoe UI", size=10)
            )

            cb.pack(fill="x", padx=6, pady=2)

        self.btn_rodar.configure(state="normal")

    def _rodar_analise(self):
        if self.df is None:
            return

        desfecho   = self.combo_desfecho.get()
        preditores = [c for c, v in self.var_checks.items() if v.get()]

        if not preditores:
            messagebox.showwarning("Aviso", "Selecione ao menos um preditor (X).")
            return

        tipo = identificar_tipo_variavel(self.df[desfecho])

        # Heatmap: exibido antes dos resultados, para qualquer tipo
        if self.flag_heat.get():
            exibir_heatmap(self.df, preditores)

        try:
            if tipo == "continua":
                analisar_continua(
                    self.df, desfecho, preditores,
                    self.var_norm_y.get(), self
                )
            elif tipo == "binaria":
                analisar_binaria(self.df, desfecho, preditores, self)
            elif tipo == "contagem":
                analisar_contagem(
                    self.df, desfecho, preditores,
                    self.var_norm_y.get(), self
                )
            elif tipo == "categorica":
                analisar_categorica(self.df, desfecho, preditores, self)
            else:
                self._escrever("Tipo de variável não reconhecido.", limpar=True)

            self._escrever_citacao()

        
        except Exception as e:
            messagebox.showerror("Erro na Análise", str(e))

    def _escrever(self, texto: str, limpar: bool = False):
        """Escreve no CTkTextbox de resultados (substitui ScrolledText)."""
        try:
            if not self.txt_resultado.winfo_exists():
                return
            self.txt_resultado.configure(state="normal")
            if limpar:
                self.txt_resultado.delete("1.0", "end")
            self.txt_resultado.insert("end", texto + "\n")
            self.txt_resultado.configure(state="disabled")
            self.txt_resultado.see("end")
        except Exception:
            pass

    def _escrever_citacao(self):
        citacao = """
══════════════════════════════════════════════════════════════
    COMO CITAR ESTE SOFTWARE
══════════════════════════════════════════════════════════════

    Neves, E. B. (2026). Análise de Regressão Automática (Versão 1.0)
    [Software]. Zenodo. DOI: https://doi.org/10.5281/zenodo.19653742

    Formato ABNT:
    NEVES, Eduardo Borba. Análise de Regressão Automática. Versão 1.0.
    [S.l.]: Zenodo, 2026. Software. DOI: https://doi.org/10.5281/zenodo.19653742 

    Formato APA:
    Neves, E. B. (2026). Análise de Regressão Automática (Version 1.0)
    [Computer software]. Zenodo. DOI: https://doi.org/10.5281/zenodo.19653742

══════════════════════════════════════════════════════════════
"""
        self._escrever(citacao)
        
    def _exportar(self):
        """Exporta o relatório em .txt e o gráfico ativo em .png."""
        try:
            conteudo = self.txt_resultado.get("1.0", "end-1c")
            if len(conteudo.strip()) < 5:
                messagebox.showwarning(
                    "Aviso", "O relatório está vazio. Rode uma análise primeiro."
                )
                return

            caminho = filedialog.asksaveasfilename(
                title="Exportar Relatório",
                defaultextension=".txt",
                filetypes=[
                    ("Arquivo de Texto", "*.txt"),
                    ("Todos os arquivos", "*.*")
                ]
            )
            if not caminho:
                return

            with open(caminho, "w", encoding="utf-8") as f:
                f.write(conteudo)

            if plt.get_fignums():
                caminho_img = caminho.replace(".txt", ".png")
                plt.savefig(caminho_img, dpi=150, bbox_inches="tight")
                messagebox.showinfo(
                    "Sucesso",
                    f"Relatório salvo em:\n{caminho}\n\nGráfico salvo em:\n{caminho_img}"
                )
            else:
                messagebox.showinfo("Sucesso", f"Relatório salvo em:\n{caminho}")

        except Exception as e:
            messagebox.showerror("Erro ao Exportar", f"Ocorreu um erro técnico:\n{e}")

    def sair_seguro(self):
        try:
            plt.close("all")
            self.quit()
            self.destroy()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()



# In[ ]:





# In[ ]:




