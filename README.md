# Análise de Regressão Automática v1.01

Uma ferramenta poderosa e intuitiva desenvolvida em Python para a realização de análises de regressão estatística complexas. O software automatiza a verificação de pressupostos e a seleção de modelos, sendo ideal para investigadores, académicos e analistas de dados.

**Autor:** Eduardo Borba Neves

---

<img width="1900" height="942" alt="image" src="https://github.com/user-attachments/assets/45b1b5b8-e628-4bae-a4ae-f85c6387f876" />


## 🚀 Funcionalidades Principais

Este App identifica automaticamente o tipo de variável desfecho (VD) e recomenda o modelo estatístico mais apropriado:

- **Regressão Linear (OLS):** Com remediação automática para não-normalidade através de transformações matemáticas (log1p, sqrt) ou Regressão Robusta (RLM - Huber T).
- **Regressão Logística Binária:** Para desfechos dicotómicos, incluindo cálculos de Odds Ratios (OR) e Curva ROC.
- **Regressão de Contagem:** Seleção automática entre Poisson, Quasi-Poisson e Binomial Negativa com base na superdispersão (Teste de Cameron-Trivedi).
- **Regressão Logística Multinomial:** Para variáveis categóricas com mais de dois níveis, utilizando RRR (Relative Risk Ratios).

## 📊 Diagnósticos Estatísticos Automáticos

O software executa uma bateria de testes para garantir o rigor científico:
- **Multicolinearidade:** Fator de Inflação da Variância (VIF).
- **Normalidade dos Resíduos:** Testes de Shapiro-Wilk ou Kolmogorov-Smirnov.
- **Homocedasticidade:** Teste de Breusch-Pagan.
- **Independência:** Estatística de Durbin-Watson.
- **Influência:** Distância de Cook e análise de outliers (Z-score).

## 🛠️ Requisitos Técnicos

Para executar a aplicação, necessita de ter o Python 3.10+ instalado e as seguintes bibliotecas:

```
bash

pip install customtkinter pandas openpyxl matplotlib statsmodels scikit-learn scipy seaborn

````

## 💻 Como Utilizar

1.  **Carregar Dados:** Importe a sua planilha Excel (.xlsx).

2.  **Configurar:** Selecione a variável desfecho (Y). O sistema detetará o tipo de análise automaticamente.

3.  **Selecionar Preditores:** Marque as variáveis independentes (X) que deseja incluir no modelo.

4.  **Executar:** Clique em "RODAR ANÁLISE" para gerar o relatório textual detalhado e os gráficos de diagnóstico 2x2.

5.  **Exportar:** Guarde o relatório em formato .txt e os gráficos em .png para a sua publicação.



## 🎓 Citação Académica

Se utilizar este software na sua investigação, por favor cite-o da seguinte forma:

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

-----

## 📄 Licença

Este projeto está licenciado sob a [Licença MIT](https://www.google.com/search?q=LICENSE).

"""

