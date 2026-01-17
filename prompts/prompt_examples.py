# Have additional docs to keep approved code, to better improve 
# The easiest way to get data into our RAG system?
FEW_SHOT_EXAMPLES = [
    {
        "question": "Show the distribution of tb_status by sex",
        "code": """\
```python
tbl = pd.crosstab(df['tb_status'], df['sex'])
tbl_percent = tbl.div(tbl.sum(axis=1), axis=0) * 100
print(f'The distribution of tb_status by sex is:\n{tbl_percent.applymap(lambda x: f"{x:.1f}%").round(1)}')
```
"""
    },
    {
        "question": "Fit a Kaplan-Meier survival curve for relapse event stratified by sex",
        "code": """\
```python
df_sub = df[~df['event_relapse'].isna()]
T = df_sub['time_to_relapse']
E = df_sub['event_relapse']
groups = df_sub['sex'].unique()

for g in groups:
    group = df_sub['sex'] == g
    
    # Check if the group is empty before fitting
    if group.sum() == 0:
        continue
        
    kmf = KaplanMeierFitter()
    kmf.fit(T[group], event_observed=E[group], label=f'{g}')
    kmf.plot_survival_function()
plt.show()
```
"""
    },
    {
        "question": "Calculate the proportion of participants with diabetes",
        "code": """\
```python
prop = df['has_diabetes'].mean()
print(f"Proportion of participants with diabetes is:\\n{prop:.2%}")
```
"""
    },
        {
        "question": "Calculate the distribution of participants' age",
        "code": """\
```python
out = df['age'].describe()
print(f'Distribution of participants' age is:\\n{out}")
```
"""
    },
        {
        "question": "What is the effect of smoke on active TB",
        "code": """\
```python
# Note this is a Binary vs Binary case, so we draw 2x2 contingency table
tbl = pd.crosstab(df['is_smoke'], df['ever_had_active_tb'])

# Choose Fisher's exact if any cell <5, else Chi-square
a, b, c, d = tbl.iloc[0,0], tbl.iloc[0,1], tbl.iloc[1,0], tbl.iloc[1,1]
if (a<5) or (b<5) or (c<5) or (d<5):
    from scipy.stats import fisher_exact
    OR, p = fisher_exact(tbl)
else:
    from scipy.stats import chi2_contingency
    chi2, p, _, _ = chi2_contingency(tbl)
    # Compute odds ratio manually
    OR = (d * a) / (b * c) if (b*c)!=0 else float('nan')

# 95% CI for OR (Woolf method)
import numpy as np
se = np.sqrt(1/a + 1/b + 1/c + 1/d)
log_or = np.log(OR)
ci_low = np.exp(log_or - 1.96*se)
ci_high = np.exp(log_or + 1.96*se)

print(
    f'Contigency table\\n{tbl}\\n'
    f'Odds Ratio (is_smoke vs ever_had_active_tb): {OR:.3f}\\n'
    f'95% CI: [{ci_low:.3f}, {ci_high:.3f}]\\n'
    f'P-value: {p:.4g}'
)
```
"""
    },
    {
        "question": "What is the relationsihp between hiv and active tb status",
        "code": """\
```python
# Note this is a binary x categorical, because tb status has more than two levels
tbl = pd.crosstab(df['is_smoke'], df['ever_had_active_tb'])

from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(tbl)
print(
    f'Conteigency table\\n{tbl}\\n'
    f'Chi-square: {chi2:.3f}, df={dof}, p={p:.4g}'
)
```
"""
    },
    {
        "question": "Help me to fit a Cox Proportional Hazards Model in relapse looking at covariate: smoking, diabetes, and hiv",
        "code": """\
```python
# Filter out NA cases in the event
df_sub = df[~df['event_relapse'].isna()]
# Select and preprocess covariates
covariates = df_sub[[
    'time_to_relapse', 'event_relapse', 'is_smoke', 
    'is_hiv_positive', 'has_diabetes'
]].copy()
# Convert categorical variables to dummies (numerical)
cph_data = pd.get_dummies(
    covariates, 
    columns=['is_smoke', 'is_hiv_positive', 'has_diabetes'],
    drop_first=True # Drop one category from each set to prevent multicollinearity
)
cph = CoxPHFitter()
cph.fit(cph_data, duration_col='time_to_relapse', event_col='event_relapse')
print(cph.print_summary())
```
"""
    }
]
