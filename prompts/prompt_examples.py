# Have additional docs to keep approved code, to better improve 
# The easiest way to get data into our RAG system?
FEW_SHOT_EXAMPLES = [
    {
        "question": "Show the distribution of tb_status by sex",
        "code": """\
{"response_type":"code_result","summary":"Builds a cross-tabulation of tb_status by sex and prints row percentages.","assumptions":"Dataset used: example-dataset.","code":"data = datasets[\"example-dataset\"]\\ntbl = pd.crosstab(data['tb_status'], data['sex'])\\ntbl_percent = tbl.div(tbl.sum(axis=1), axis=0) * 100\\nprint(f'The distribution of tb_status by sex is:\\n{tbl_percent.applymap(lambda x: f\"{x:.1f}%\").round(1)}')"}
"""
    },
    {
        "question": "Fit a Kaplan-Meier survival curve for relapse event stratified by sex",
        "code": """\
{"response_type":"code_result","summary":"Fits Kaplan-Meier relapse curves stratified by sex and plots the survival functions.","assumptions":"Dataset used: example-dataset.","code":"data = datasets[\"example-dataset\"]\\ndf_sub = data[~data['event_relapse'].isna()]\\nT = df_sub['time_to_relapse']\\nE = df_sub['event_relapse']\\ngroups = df_sub['sex'].unique()\\n\\nfor g in groups:\\n    group = df_sub['sex'] == g\\n    if group.sum() == 0:\\n        continue\\n    kmf = KaplanMeierFitter()\\n    kmf.fit(T[group], event_observed=E[group], label=f'{g}')\\n    kmf.plot_survival_function()\\nplt.show()"}
"""
    },
    {
        "question": "Calculate the proportion of participants with diabetes",
        "code": """\
{"response_type":"code_result","summary":"Calculates and prints the proportion of participants with diabetes.","assumptions":"Dataset used: example-dataset.","code":"data = datasets[\"example-dataset\"]\\nprop = data['has_diabetes'].mean()\\nprint(f\\"Proportion of participants with diabetes is:\\\\n{prop:.2%}\\")"}
"""
    },
        {
        "question": "Calculate the distribution of participants' age",
        "code": """\
{"response_type":"code_result","summary":"Computes descriptive statistics for participant age and prints them.","assumptions":"Dataset used: example-dataset.","code":"data = datasets[\"example-dataset\"]\\nout = data['age'].describe()\\nprint(f\\"Distribution of participants' age is:\\\\n{out}\\")"}
"""
    },
        {
        "question": "What is the effect of smoke on active TB",
        "code": """\
{"response_type":"code_result","summary":"Builds a 2x2 table for smoking and active TB, selects the appropriate association test, and prints the odds ratio with confidence interval.","assumptions":"Dataset used: example-dataset.","code":"data = datasets[\"example-dataset\"]\\ntbl = pd.crosstab(data['is_smoke'], data['ever_had_active_tb'])\\na, b, c, d = tbl.iloc[0,0], tbl.iloc[0,1], tbl.iloc[1,0], tbl.iloc[1,1]\\nif (a<5) or (b<5) or (c<5) or (d<5):\\n    from scipy.stats import fisher_exact\\n    OR, p = fisher_exact(tbl)\\nelse:\\n    from scipy.stats import chi2_contingency\\n    chi2, p, _, _ = chi2_contingency(tbl)\\n    OR = (d * a) / (b * c) if (b*c)!=0 else float('nan')\\nimport numpy as np\\nse = np.sqrt(1/a + 1/b + 1/c + 1/d)\\nlog_or = np.log(OR)\\nci_low = np.exp(log_or - 1.96*se)\\nci_high = np.exp(log_or + 1.96*se)\\nprint(\\n    f'Contigency table\\\\n{tbl}\\\\n'\\n    f'Odds Ratio (is_smoke vs ever_had_active_tb): {OR:.3f}\\\\n'\\n    f'95% CI: [{ci_low:.3f}, {ci_high:.3f}]\\\\n'\\n    f'P-value: {p:.4g}'\\n)"}
"""
    },
    {
        "question": "What is the relationsihp between hiv and active tb status",
        "code": """\
{"response_type":"code_result","summary":"Builds a contingency table for HIV and active TB status, runs a chi-square test, and prints the result.","assumptions":"Dataset used: example-dataset.","code":"data = datasets[\"example-dataset\"]\\ntbl = pd.crosstab(data['is_smoke'], data['ever_had_active_tb'])\\nfrom scipy.stats import chi2_contingency\\nchi2, p, dof, expected = chi2_contingency(tbl)\\nprint(\\n    f'Conteigency table\\\\n{tbl}\\\\n'\\n    f'Chi-square: {chi2:.3f}, df={dof}, p={p:.4g}'\\n)"}
"""
    },
    {
        "question": "Help me to fit a Cox Proportional Hazards Model in relapse looking at covariate: smoking, diabetes, and hiv",
        "code": """\
{"response_type":"code_result","summary":"Prepares relapse survival data, fits a Cox proportional hazards model with smoking, diabetes, and HIV covariates, and prints the model summary.","assumptions":"Dataset used: example-dataset.","code":"data = datasets[\"example-dataset\"]\\ndf_sub = data[~data['event_relapse'].isna()]\\ncovariates = df_sub[[\\n    'time_to_relapse', 'event_relapse', 'is_smoke', \\n    'is_hiv_positive', 'has_diabetes'\\n]].copy()\\ncph_data = pd.get_dummies(\\n    covariates, \\n    columns=['is_smoke', 'is_hiv_positive', 'has_diabetes'],\\n    drop_first=True\\n)\\ncph = CoxPHFitter()\\ncph.fit(cph_data, duration_col='time_to_relapse', event_col='event_relapse')\\nprint(cph.print_summary())"}
"""
    }
]
