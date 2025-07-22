import marimo

__generated_with = "0.12.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from scipy.stats import pearsonr
    from statsmodels.stats.multitest import multipletests
    return mo, multipletests, np, pd, pearsonr


@app.cell
def _(mo):
    mo.md(r"""## Person Correlation""")
    return


@app.cell
def _(pd):
    DMP=pd.read_csv('data/sample_MC.csv',index_col=0)
    metadata=pd.read_csv('data/metadata.csv',index_col=0)
    DMP['shareID']=metadata['shareID'].values.tolist()
    grouped_DMP = DMP.groupby('shareID').mean()
    qpcr=pd.read_csv('data/qpcr_ctvalue.csv',index_col=0)
    ctvalues_list=qpcr['Ct value '].values.tolist()
    reorder_grouped_DMP=grouped_DMP.loc[qpcr.index.tolist()]
    return DMP, ctvalues_list, grouped_DMP, metadata, qpcr, reorder_grouped_DMP


@app.cell(hide_code=True)
def _(multipletests, np, pd, pearsonr):
    def correlation_analysis(topic, alpha_289_list):
        results = []

        # Convert alpha_289_list to numpy array if it's not already
        alpha_array = np.array(alpha_289_list)

        # Calculate Pearson correlations, only considering non-zero values
        for column in topic.columns:
            # Filter rows where alpha_289_list is not 0
            valid_indices = np.where(alpha_array != 0)[0]

            # Extract the valid data (without 0s)
            filtered_alpha = alpha_array[valid_indices]
            filtered_column_values = topic.iloc[valid_indices][column]
            corr, p_value = pearsonr(filtered_column_values, filtered_alpha)
            results.append({
                'Column': column,
                'Correlation': corr,
                'P-value': p_value,
                'Valid_samples': len(filtered_alpha)
            })
        # Convert to DataFrame and apply Benjamini-Hochberg FDR correction
        df = pd.DataFrame(results)

        # Only apply multiple testing correction on valid p-values
        valid_p_values = ~df['P-value'].isna()
        if valid_p_values.any():
            df.loc[valid_p_values, 'BH_FDR'] = multipletests(df.loc[valid_p_values, 'P-value'], method='fdr_bh')[1]
        else:
            df['BH_FDR'] = float('nan')

        return df.round(6)
    return (correlation_analysis,)


@app.cell
def _(correlation_analysis, ctvalues_list, reorder_grouped_DMP):
    coorelation_result_ctvalue=correlation_analysis(reorder_grouped_DMP, ctvalues_list)
    return (coorelation_result_ctvalue,)


@app.cell
def _(mo):
    mo.md(r"""## PERMANOVA""")
    return


@app.cell
def _():
    from skbio.stats.distance import DistanceMatrix
    from skbio.stats.distance import permanova
    return DistanceMatrix, permanova


@app.cell
def _(metadata, pd):
    distance_matrix = pd.read_csv('data/distance-matrix(unweightedunifrac).tsv',sep='\t',index_col=0)
    grouping = metadata['LDA_group']
    return distance_matrix, grouping


@app.cell
def _(DistanceMatrix, distance_matrix, grouping, metadata, np, permanova):
    matrix_values = np.ascontiguousarray(distance_matrix.values)

    # Now create the DistanceMatrix
    dm = DistanceMatrix(matrix_values, ids=metadata.index)
    # Run PERMANOVA
    # You can adjust permutations (default is 999)
    permanova_results = permanova(dm, grouping, permutations=999)
    return dm, matrix_values, permanova_results


@app.cell
def _(mo):
    mo.md(r"""## MC Distribution Difference""")
    return


@app.cell
def _():
    import scipy.stats as stats
    from scikit_posthocs import posthoc_dunn
    return posthoc_dunn, stats


@app.cell
def _(pd, posthoc_dunn, stats):
    def analyze_topic_sets(dfs):
        """
        Analyze topic sets across all time periods and return two DataFrames:
        1. All results (regardless of significance)
        2. Only significant positive results (Dunn p-value < 0.05)
        """
        # Define time periods and their corresponding keys
        periods = {
            "First SampleTime": ['Infected_1st', 'Intermediate_1st', 'Healthy_1st'],
            "Early Stage": ['Infected_es', 'Intermediate_es', 'Healthy_es'],
            "Transition Stage": ['Infected_ts', 'Intermediate_ts', 'Healthy_ts'],
            "Late Stage": ['Infected_ls', 'Intermediate_ls', 'Healthy_ls']
        }

        all_results = []

        for period_name, period_keys in periods.items():
            # Get DataFrames for this period
            period_dfs = {k: dfs[k] for k in period_keys if k in dfs}

            if not period_dfs:  # Skip if no data for this period
                continue

            print(f"\nAnalyzing {period_name}")

            # Get subgroups (columns) from the first available DataFrame
            subgroups = list(period_dfs[list(period_dfs.keys())[0]].columns)

            for subgroup_name in subgroups:
                # Prepare data for this subgroup
                subgroup_data = []
                group_labels = []

                for group_name, df in period_dfs.items():
                    subgroup_data.extend(df[subgroup_name].values)
                    group_labels.extend([group_name] * len(df))

                analysis_df = pd.DataFrame({
                    'Group': group_labels,
                    'MC_Prob': subgroup_data
                })

                # Perform Kruskal-Wallis test
                groups = [group[1]['MC_Prob'].values for group in analysis_df.groupby('Group')]
                h_stat, kw_p_value = stats.kruskal(*groups)

                # Always perform post-hoc tests and calculate effect sizes
                if len(groups) > 1:  # Need at least 2 groups for comparison
                    try:
                        # Dunn's test
                        dunn_results = posthoc_dunn(analysis_df, val_col='MC_Prob', 
                                                  group_col='Group', p_adjust='bonferroni')

                        # Calculate effect sizes
                        effect_sizes = calculate_effect_sizes(analysis_df)

                        # Get all pairwise comparisons
                        unique_groups = sorted(analysis_df['Group'].unique())
                        for i, g1 in enumerate(unique_groups):
                            for g2 in unique_groups[i+1:]:
                                dunn_p = dunn_results.loc[g1, g2] if g1 in dunn_results.index and g2 in dunn_results.columns else float('nan')
                                effect_size = effect_sizes.loc[g1, g2] if g1 in effect_sizes.index and g2 in effect_sizes.columns else float('nan')

                                all_results.append({
                                    'Period': period_name,
                                    'MC': subgroup_name,
                                    'Group1': g1,
                                    'Group2': g2,
                                    'H_statistic': round(h_stat, 6),
                                    'KW_pvalue': round(kw_p_value, 6),
                                    'Dunn_pvalue': round(dunn_p, 6) if not pd.isna(dunn_p) else dunn_p,
                                    'Effect_size': round(effect_size, 6) if not pd.isna(effect_size) else effect_size,
                                    'Effect_interpretation': interpret_effect_size(effect_size) if not pd.isna(effect_size) else 'Unknown'
                                })

                    except Exception as e:
                        print(f"Error processing {period_name} - {subgroup_name}: {e}")
                        continue

        # Convert to DataFrame
        all_results_df = pd.DataFrame(all_results)

        # Create significant results DataFrame (Dunn p-value < 0.05)
        significant_results_df = all_results_df[
            (all_results_df['Dunn_pvalue'] < 0.05) & 
            (~all_results_df['Dunn_pvalue'].isna())
        ].copy()

        # Set display options for better formatting
        pd.set_option('display.float_format', lambda x: '{:.6f}'.format(x))

        return all_results_df, significant_results_df

    def calculate_effect_sizes(df):
        """
        Calculates Cliff's Delta effect size for all group pairs
        """
        groups = sorted(df['Group'].unique())
        effect_sizes = pd.DataFrame(index=groups, columns=groups, dtype=float)

        for g1 in groups:
            for g2 in groups:
                if g1 < g2:
                    x = df[df['Group'] == g1]['MC_Prob'].values
                    y = df[df['Group'] == g2]['MC_Prob'].values
                    delta = cliffs_delta(x, y)
                    effect_sizes.loc[g1, g2] = delta
                    effect_sizes.loc[g2, g1] = -delta
                elif g1 == g2:
                    effect_sizes.loc[g1, g2] = 0.0

        return effect_sizes

    def cliffs_delta(x, y):
        """
        Calculates Cliff's Delta effect size
        """
        if len(x) == 0 or len(y) == 0:
            return float('nan')

        nx = len(x)
        ny = len(y)
        dominance = 0

        for i in x:
            for j in y:
                if i > j:
                    dominance += 1
                elif i < j:
                    dominance -= 1

        return dominance / (nx * ny)

    def interpret_effect_size(delta):
        """
        Interprets Cliff's Delta effect size
        """
        if pd.isna(delta):
            return "Unknown"

        abs_delta = abs(delta)
        if abs_delta < 0.147:
            return "Negligible"
        elif abs_delta < 0.33:
            return "Small"
        elif abs_delta < 0.474:
            return "Medium"
        else:
            return "Large"
    return (
        analyze_topic_sets,
        calculate_effect_sizes,
        cliffs_delta,
        interpret_effect_size,
    )


@app.cell(hide_code=True)
def _(DMP, metadata):
    # data for small multiples
    infected_1 = metadata[
       (metadata['Greenhouse'].isin([1, 4, 5, 7])) & 
       (metadata['SampleTime'].isin([1]))
    ].index.tolist()

    infected_es = metadata[
       (metadata['Greenhouse'].isin([1, 4, 5, 7])) & 
       (metadata['SampleTime'].isin([1,2]))
    ].index.tolist()

    infected_ts = metadata[
       (metadata['Greenhouse'].isin([1, 4, 5, 7])) & 
       (metadata['SampleTime'].isin([3]))
    ].index.tolist()

    infected_ls = metadata[
       (metadata['Greenhouse'].isin([1, 4, 5, 7])) & 
       (metadata['SampleTime'].isin([4,5]))
    ].index.tolist()



    healthy_1 = metadata[
       (metadata['Greenhouse'].isin([8, 9, 10, 11])) & 
       (metadata['SampleTime'].isin([1]))
    ].index.tolist()

    healthy_es = metadata[
       (metadata['Greenhouse'].isin([8, 9, 10, 11])) & 
       (metadata['SampleTime'].isin([1,2]))
    ].index.tolist()

    healthy_ts = metadata[
       (metadata['Greenhouse'].isin([8, 9, 10, 11])) & 
       (metadata['SampleTime'].isin([3]))
    ].index.tolist()

    healthy_ls = metadata[
       (metadata['Greenhouse'].isin([8, 9, 10, 11])) & 
       (metadata['SampleTime'].isin([4,5]))
    ].index.tolist()


    intermediate_1 = metadata[
       (metadata['Greenhouse'].isin([2, 3, 6, 12])) & 
       (metadata['SampleTime'].isin([1]))
    ].index.tolist()

    intermediate_es = metadata[
       (metadata['Greenhouse'].isin([2, 3, 6, 12])) & 
       (metadata['SampleTime'].isin([1,2]))
    ].index.tolist()

    intermediate_ts = metadata[
       (metadata['Greenhouse'].isin([2, 3, 6, 12])) & 
       (metadata['SampleTime'].isin([3]))
    ].index.tolist()

    intermediate_ls = metadata[
       (metadata['Greenhouse'].isin([2, 3, 6, 12])) & 
       (metadata['SampleTime'].isin([4,5]))
    ].index.tolist()


    df_infected_1st = DMP.loc[infected_1]
    df_infected_es = DMP.loc[infected_es]
    df_infected_ts = DMP.loc[infected_ts]
    df_infected_ls = DMP.loc[infected_ls]

    df_healthy_1st = DMP.loc[healthy_1]
    df_healthy_es = DMP.loc[healthy_es]
    df_healthy_ts = DMP.loc[healthy_ts]
    df_healthy_ls = DMP.loc[healthy_ls]

    df_intermediate_1st = DMP.loc[intermediate_1]
    df_intermediate_es = DMP.loc[intermediate_es]
    df_intermediate_ts = DMP.loc[intermediate_ts]
    df_intermediate_ls = DMP.loc[intermediate_ls]

    dfs = {
        'Infected_1st': df_infected_1st,
        'Intermediate_1st': df_intermediate_1st,
        'Healthy_1st': df_healthy_1st,
        'Infected_es': df_infected_es,
        'Intermediate_es': df_intermediate_es,
        'Healthy_es': df_healthy_es,
        'Infected_ts': df_infected_ts,
        'Intermediate_ts': df_intermediate_ts,
        'Healthy_ts': df_healthy_ts,    
        'Infected_ls': df_infected_ls,
        'Intermediate_ls': df_intermediate_ls,
        'Healthy_ls': df_healthy_ls,
    }
    return (
        df_healthy_1st,
        df_healthy_es,
        df_healthy_ls,
        df_healthy_ts,
        df_infected_1st,
        df_infected_es,
        df_infected_ls,
        df_infected_ts,
        df_intermediate_1st,
        df_intermediate_es,
        df_intermediate_ls,
        df_intermediate_ts,
        dfs,
        healthy_1,
        healthy_es,
        healthy_ls,
        healthy_ts,
        infected_1,
        infected_es,
        infected_ls,
        infected_ts,
        intermediate_1,
        intermediate_es,
        intermediate_ls,
        intermediate_ts,
    )


@app.cell
def _(analyze_topic_sets, dfs):
    all_results, significant_results = analyze_topic_sets(dfs)
    return all_results, significant_results


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""## BCO-Pathogen Ratio""")
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import seaborn as sns
    return plt, sns


@app.cell(hide_code=True)
def _(np, pd):
    def calculate_ratio(df, numerator_col, denominator_col, index_list):

        # Filter dataframe to only include rows with indices in index_list
        filtered_df = df.loc[df.index.isin(index_list)]

        # Calculate ratio (handle division by zero)
        ratios = filtered_df[numerator_col] / filtered_df[denominator_col]

        # Convert to list and return
        return ratios.tolist()

    def calculate_ratio_pse(df, numerator_col, denominator_col, index_list, pseudocount=1e-6):
        # Filter dataframe to only include rows with indices in index_list
        filtered_df = df.loc[df.index.isin(index_list)].copy()

        # Add pseudocount to both numerator and denominator
        numerator_adjusted = filtered_df[numerator_col] + pseudocount
        denominator_adjusted = filtered_df[denominator_col] + pseudocount

        # Calculate ratio with adjusted values
        ratios = numerator_adjusted / denominator_adjusted

        # Convert to list and return
        return ratios.tolist()

    def create_violin_df(pre_list, inf_list, ratio_name):
        # Remove inf values but keep 0 values
        pre_clean = [x for x in pre_list if np.isfinite(x)]
        inf_clean = [x for x in inf_list if np.isfinite(x)]

        df = pd.DataFrame({
            'ratio': pre_clean + inf_clean,
            'condition': ['Pre'] * len(pre_clean) + ['Inf'] * len(inf_clean),
            'ratio_type': [ratio_name] * (len(pre_clean) + len(inf_clean))
        })
        return df

    def little_func(df,numerator,Denominator,sample_list_1, sample_list_2):
        con1_df=calculate_ratio(df,numerator,Denominator, sample_list_1)
        con2_df=calculate_ratio(df,numerator,Denominator, sample_list_2)
        concat_df=create_violin_df(con1_df, con2_df, f"{numerator}/{Denominator}")
        return concat_df

    def little_func_3(df,numerator,Denominator,sample_list_1, sample_list_2,sample_list_3):
        con1_df=calculate_ratio(df,numerator,Denominator, sample_list_1)
        con2_df=calculate_ratio(df,numerator,Denominator, sample_list_2)
        con3_df=calculate_ratio(df,numerator,Denominator, sample_list_3)
        concat_df=create_violin_df(con1_df, con2_df,con3_df, f"{numerator}/{Denominator}")
        return concat_df
    return (
        calculate_ratio,
        calculate_ratio_pse,
        create_violin_df,
        little_func,
        little_func_3,
    )


@app.cell
def _(pd):
    mod= pd.read_csv("data/modeled_asvtable.csv",index_col=0)
    rel=pd.read_csv("data/relative_asvtable.csv",index_col=0)
    return mod, rel


@app.cell
def _(mo):
    dropdown_numerator = mo.ui.dropdown(options=['Paenibacillus_11','Paenibacillus_15','Paenibacillus_16','Paenibacillus_4', 'Comamonadaceae_unknown_70', 'Comamonadaceae_unknown_65','Afipia_3','Caulobacter','Limnobacter_4'], label="choose numerator", value="Paenibacillus_11")
    dropdown_denominator = mo.ui.dropdown(options=['Rhizobium_complex_25','Rhizobium_complex_29','Limnobacter_4','Paenibacillus_11'], label="choose denominator", value="Rhizobium_complex_25")
    return dropdown_denominator, dropdown_numerator


@app.cell(hide_code=True)
def _(
    calculate_ratio,
    calculate_ratio_pse,
    dropdown_denominator,
    dropdown_numerator,
    metadata,
    mod,
    pd,
    rel,
):
    index_h_e = metadata.loc[(metadata['Diagnosis'] == 'healthy') & 
                                (metadata['Stage'].isin([0]))].index.tolist()
    index_inb_e = metadata.loc[(metadata['Diagnosis'] == 'intermediate') & 
                                (metadata['Stage'].isin([0]))].index.tolist()
    index_inf_e = metadata.loc[(metadata['Diagnosis'] == 'infected') & 
                                (metadata['Stage'].isin([0]))].index.tolist()

    rel_h_es_df=calculate_ratio(rel,dropdown_numerator.value,dropdown_denominator.value,index_h_e)
    rel_inb_es_df=calculate_ratio(rel,dropdown_numerator.value,dropdown_denominator.value,index_inb_e)
    rel_inf_es_df=calculate_ratio(rel,dropdown_numerator.value,dropdown_denominator.value,index_inf_e)


    rel_combined_es_df = pd.DataFrame({
        'ratio': rel_h_es_df + rel_inb_es_df + rel_inf_es_df,
        'Greenhouse': ['healthy'] * len(rel_h_es_df) + ['intermediate'] * len(rel_inb_es_df) + ['infected'] * len(rel_inf_es_df)
    })

    rel_h_es_df_pse = calculate_ratio_pse(rel, 'Paenibacillus_11', dropdown_denominator.value, index_h_e)
    rel_inb_es_df_pse = calculate_ratio_pse(rel, 'Paenibacillus_11', dropdown_denominator.value, index_inb_e)
    rel_inf_es_df_pse = calculate_ratio_pse(rel, 'Paenibacillus_11', dropdown_denominator.value, index_inf_e)

    # Create combined dataframe with pseudocount adjustment
    rel_combined_es_df_pse = pd.DataFrame({
        'ratio': rel_h_es_df_pse + rel_inb_es_df_pse + rel_inf_es_df_pse,
        'Greenhouse': ['healthy'] * len(rel_h_es_df_pse) + ['intermediate'] * len(rel_inb_es_df_pse) + ['infected'] * len(rel_inf_es_df_pse)
    })

    mod_h_es_df=calculate_ratio(mod,'Paenibacillus_11',dropdown_denominator.value,index_h_e)
    mod_inb_es_df=calculate_ratio(mod,'Paenibacillus_11',dropdown_denominator.value,index_inb_e)
    mod_inf_es_df=calculate_ratio(mod,'Paenibacillus_11',dropdown_denominator.value,index_inf_e)

    mod_combined_es_df = pd.DataFrame({
        'ratio': mod_h_es_df + mod_inb_es_df + mod_inf_es_df,
        'Greenhouse': ['healthy'] * len(mod_h_es_df) + ['intermediate'] * len(mod_inb_es_df) + ['infected'] * len(mod_inf_es_df)
    })
    return (
        index_h_e,
        index_inb_e,
        index_inf_e,
        mod_combined_es_df,
        mod_h_es_df,
        mod_inb_es_df,
        mod_inf_es_df,
        rel_combined_es_df,
        rel_combined_es_df_pse,
        rel_h_es_df,
        rel_h_es_df_pse,
        rel_inb_es_df,
        rel_inb_es_df_pse,
        rel_inf_es_df,
        rel_inf_es_df_pse,
    )


@app.cell(hide_code=True)
def _(
    calculate_ratio,
    calculate_ratio_pse,
    dropdown_denominator,
    dropdown_numerator,
    metadata,
    mod,
    pd,
    rel,
):
    index_h_t = metadata.loc[(metadata['Diagnosis'] == 'healthy') & 
                                (metadata['Stage'].isin([1]))].index.tolist()
    index_inb_t = metadata.loc[(metadata['Diagnosis'] == 'intermediate') & 
                                (metadata['Stage'].isin([1]))].index.tolist()
    index_inf_t = metadata.loc[(metadata['Diagnosis'] == 'infected') & 
                                (metadata['Stage'].isin([1]))].index.tolist()

    rel_h_ts_df=calculate_ratio(rel,dropdown_numerator.value,dropdown_denominator.value,index_h_t)
    rel_inb_ts_df=calculate_ratio(rel,dropdown_numerator.value,dropdown_denominator.value,index_inb_t)
    rel_inf_ts_df=calculate_ratio(rel,dropdown_numerator.value,dropdown_denominator.value,index_inf_t)

    rel_combined_ts_df = pd.DataFrame({
        'ratio': rel_h_ts_df + rel_inb_ts_df + rel_inf_ts_df,
        'Greenhouse': ['healthy'] * len(rel_h_ts_df) + ['intermediate'] * len(rel_inb_ts_df) + ['infected'] * len(rel_inf_ts_df)
    })

    rel_h_ts_df_pse = calculate_ratio_pse(rel, dropdown_numerator.value, dropdown_denominator.value, index_h_t)
    rel_inb_ts_df_pse = calculate_ratio_pse(rel, dropdown_numerator.value, dropdown_denominator.value, index_inb_t)
    rel_inf_ts_df_pse = calculate_ratio_pse(rel, dropdown_numerator.value, dropdown_denominator.value, index_inf_t)

    # Create combined dataframe with pseudocount adjustment for TS stage
    rel_combined_ts_df_pse = pd.DataFrame({
        'ratio': rel_h_ts_df_pse + rel_inb_ts_df_pse + rel_inf_ts_df_pse,
        'Greenhouse': ['healthy'] * len(rel_h_ts_df_pse) + ['intermediate'] * len(rel_inb_ts_df_pse) + ['infected'] * len(rel_inf_ts_df_pse)
    })

    mod_h_ts_df=calculate_ratio(mod,dropdown_numerator.value,dropdown_denominator.value,index_h_t)
    mod_inb_ts_df=calculate_ratio(mod,dropdown_numerator.value,dropdown_denominator.value,index_inb_t)
    mod_inf_ts_df=calculate_ratio(mod,dropdown_numerator.value,dropdown_denominator.value,index_inf_t)

    mod_combined_ts_df = pd.DataFrame({
        'ratio': mod_h_ts_df + mod_inb_ts_df + mod_inf_ts_df,
        'Greenhouse': ['healthy'] * len(mod_h_ts_df) + ['intermediate'] * len(mod_inb_ts_df) + ['infected'] * len(mod_inf_ts_df)
    })
    return (
        index_h_t,
        index_inb_t,
        index_inf_t,
        mod_combined_ts_df,
        mod_h_ts_df,
        mod_inb_ts_df,
        mod_inf_ts_df,
        rel_combined_ts_df,
        rel_combined_ts_df_pse,
        rel_h_ts_df,
        rel_h_ts_df_pse,
        rel_inb_ts_df,
        rel_inb_ts_df_pse,
        rel_inf_ts_df,
        rel_inf_ts_df_pse,
    )


@app.cell(hide_code=True)
def _(
    calculate_ratio,
    calculate_ratio_pse,
    dropdown_denominator,
    dropdown_numerator,
    metadata,
    mod,
    pd,
    rel,
):
    index_h_l = metadata.loc[(metadata['Diagnosis'] == 'healthy') & 
                                (metadata['Stage'].isin([2]))].index.tolist()
    index_inb_l = metadata.loc[(metadata['Diagnosis'] == 'intermediate') & 
                                (metadata['Stage'].isin([2]))].index.tolist()
    index_inf_l = metadata.loc[(metadata['Diagnosis'] == 'infected') & 
                                (metadata['Stage'].isin([2]))].index.tolist()

    rel_h_ls_df=calculate_ratio(rel,dropdown_numerator.value,dropdown_denominator.value,index_h_l)
    rel_inb_ls_df=calculate_ratio(rel,dropdown_numerator.value,dropdown_denominator.value,index_inb_l)
    rel_inf_ls_df=calculate_ratio(rel,dropdown_numerator.value,dropdown_denominator.value,index_inf_l)

    rel_combined_ls_df = pd.DataFrame({
        'ratio': rel_h_ls_df + rel_inb_ls_df + rel_inf_ls_df,
        'Greenhouse': ['healthy'] * len(rel_h_ls_df) + ['intermediate'] * len(rel_inb_ls_df) + ['infected'] * len(rel_inf_ls_df)
    })

    rel_h_ls_df_pse = calculate_ratio_pse(rel, dropdown_numerator.value, dropdown_denominator.value, index_h_l)
    rel_inb_ls_df_pse = calculate_ratio_pse(rel, dropdown_numerator.value, dropdown_denominator.value, index_inb_l)
    rel_inf_ls_df_pse = calculate_ratio_pse(rel, dropdown_numerator.value, dropdown_denominator.value, index_inf_l)

    # Create combined dataframe with pseudocount adjustment for LS stage
    rel_combined_ls_df_pse = pd.DataFrame({
        'ratio': rel_h_ls_df_pse + rel_inb_ls_df_pse + rel_inf_ls_df_pse,
        'Greenhouse': ['healthy'] * len(rel_h_ls_df_pse) + ['intermediate'] * len(rel_inb_ls_df_pse) + ['infected'] * len(rel_inf_ls_df_pse)
    })


    mod_h_ls_df=calculate_ratio(mod,dropdown_numerator.value,dropdown_denominator.value,index_h_l)
    mod_inb_ls_df=calculate_ratio(mod,dropdown_numerator.value,dropdown_denominator.value,index_inb_l)
    mod_inf_ls_df=calculate_ratio(mod,dropdown_numerator.value,dropdown_denominator.value,index_inf_l)

    mod_combined_ls_df = pd.DataFrame({
        'ratio': mod_h_ls_df + mod_inb_ls_df + mod_inf_ls_df,
        'Greenhouse': ['healthy'] * len(mod_h_ls_df) + ['intermediate'] * len(mod_inb_ls_df) + ['infected'] * len(mod_inf_ls_df)
    })
    return (
        index_h_l,
        index_inb_l,
        index_inf_l,
        mod_combined_ls_df,
        mod_h_ls_df,
        mod_inb_ls_df,
        mod_inf_ls_df,
        rel_combined_ls_df,
        rel_combined_ls_df_pse,
        rel_h_ls_df,
        rel_h_ls_df_pse,
        rel_inb_ls_df,
        rel_inb_ls_df_pse,
        rel_inf_ls_df,
        rel_inf_ls_df_pse,
    )


@app.cell
def _(dropdown_numerator):
    dropdown_numerator
    return


@app.cell
def _(dropdown_denominator):
    dropdown_denominator
    return


@app.cell(hide_code=True)
def _(np, pd, plt, sns):
    def create_boxplot_with_stats(_combined_es_df, _save_path, 
                                 _figsize=(8, 6), 
                                 x_col='Greenhouse', 
                                 y_col='ratio',
                                 _point_color='grey',
                                 _point_size=4,
                                 _box_alpha=0.8,
                                 _median_precision=2):
        """
        Create a boxplot with outlier filtering and comprehensive statistics.

        Parameters:
        - _combined_es_df: DataFrame with condition and ratio columns
        - _save_path: Path to save the figure
        - _figsize: Figure size tuple (width, height)
        - _x_col: Column name for x-axis (conditions)
        - _y_col: Column name for y-axis (values)
        - _point_color: Color for individual data points
        - _point_size: Size of individual data points
        - _box_alpha: Transparency of box plots
        - _median_precision: Decimal places for median values
        """
        # Clean data
        _combined_es_df_clean = _combined_es_df.replace([np.inf, -np.inf], np.nan).dropna()

        # Create figure
        _fig, _ax = plt.subplots(figsize=_figsize)

        # Create box plot without outliers
        sns.boxplot(data=_combined_es_df_clean, x=x_col, y=y_col, ax=_ax, 
                    boxprops=dict(alpha=_box_alpha), showfliers=False)

        # Function to remove outliers and count them
        def _remove_outliers_and_count(_group):
            _q1 = _group[y_col].quantile(0.25)
            _q3 = _group[y_col].quantile(0.75)
            _iqr = _q3 - _q1
            _lower = _q1 - 1.5 * _iqr
            _upper = _q3 + 1.5 * _iqr
            _non_outliers = _group[(_group[y_col] >= _lower) & (_group[y_col] <= _upper)]
            _outlier_count = len(_group) - len(_non_outliers)
            return _non_outliers, _outlier_count

        # Process each condition
        _filtered_data = []
        _outlier_counts = {}
        _conditions = _combined_es_df_clean[x_col].unique()

        for _condition in _conditions:
            _condition_data = _combined_es_df_clean[_combined_es_df_clean[x_col] == _condition]
            _non_outliers, _outlier_count = _remove_outliers_and_count(_condition_data)
            _filtered_data.append(_non_outliers)
            _outlier_counts[_condition] = _outlier_count

        # Combine filtered data and add to plot
        _filtered_df = pd.concat(_filtered_data, ignore_index=True)
        sns.stripplot(data=_filtered_df, x=x_col, y=y_col, ax=_ax, 
                      edgecolor=_point_color, facecolor='none', linewidth=1, 
                      size=_point_size, jitter=True)

        # Create labels and median annotations
        _new_labels = []
        for _i, _condition in enumerate(_conditions):
            # Get data for this condition
            _condition_data = _combined_es_df_clean[_combined_es_df_clean[x_col] == _condition][y_col]
            _condition_original = _combined_es_df[_combined_es_df[x_col] == _condition][y_col]

            # Calculate statistics
            _sample_count_clean = len(_condition_data)
            _sample_count_original = len(_condition_original)
            _inf_count = _sample_count_original - _sample_count_clean
            _outlier_count = _outlier_counts[_condition]
            _median_val = _condition_data.median()

            # Create label
            _new_labels.append(f'{_condition}\n(n={_sample_count_clean}/{_sample_count_original})\n{_inf_count} inf, {_outlier_count} outliers')

            # Add median annotation
            _ax.text(_i, _median_val, f'Median: {_median_val:.{_median_precision}f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='black'))

        # Finalize plot
        _ax.set_xticklabels(_new_labels)
        _ax.set_title(f'{y_col.title()} Distribution by {x_col.title()}')
        _ax.set_xlabel(x_col.title())
        _ax.set_ylabel(y_col.title())
        plt.tight_layout()

        # Save figure
        _fig.savefig(_save_path, format='svg', bbox_inches='tight')

        return _fig
    return (create_boxplot_with_stats,)


@app.cell(hide_code=True)
def _(
    dropdown_denominator,
    dropdown_numerator,
    mod_combined_es_df,
    mod_combined_ls_df,
    mod_combined_ts_df,
    np,
    pd,
    plt,
    rel_combined_es_df,
    rel_combined_es_df_pse,
    rel_combined_ls_df,
    rel_combined_ls_df_pse,
    rel_combined_ts_df,
    rel_combined_ts_df_pse,
    sns,
):
    # Define colors
    infected='#d7191c'
    intermediate='#4a1486'
    healthy='#abdda4'

    # Create color palette - assuming the order matches your conditions
    _color_palette = [healthy, intermediate, infected]

    # Create the combined figure
    _fig, _axes = plt.subplots(3, 3, figsize=(18, 12))
    _datasets = [
        (rel_combined_es_df, "ES"),
        (rel_combined_ts_df, "TS"), 
        (rel_combined_ls_df, "LS"),
        (rel_combined_es_df_pse, "ES"),
        (rel_combined_ts_df_pse, "TS"), 
        (rel_combined_ls_df_pse, "LS"),
        (mod_combined_es_df, "ES"),
        (mod_combined_ts_df, "TS"), 
        (mod_combined_ls_df, "LS")
    ]

    _row_labels = ["Relative Abundance", "Relative Abundance (add pseudocount)","Modeled Relative Abundance"]

    for _idx, (_df, _label) in enumerate(_datasets):
        # Calculate row and column indices
        _row = _idx // 3
        _col = _idx % 3
        _ax = _axes[_row, _col]

        # Clean data
        _df_clean = _df.replace([np.inf, -np.inf], np.nan).dropna()

        # Create box plot
        sns.boxplot(data=_df_clean, x='Greenhouse', y='ratio', ax=_ax, 
                    boxprops=dict(alpha=0.8), showfliers=False, palette=_color_palette)

        # Function to remove outliers
        def _remove_outliers_and_count(_group):
            _q1 = _group['ratio'].quantile(0.25)
            _q3 = _group['ratio'].quantile(0.75)
            _iqr = _q3 - _q1
            _lower = _q1 - 1.5 * _iqr
            _upper = _q3 + 1.5 * _iqr
            _non_outliers = _group[(_group['ratio'] >= _lower) & (_group['ratio'] <= _upper)]
            _outlier_count = len(_group) - len(_non_outliers)
            return _non_outliers, _outlier_count

        # Process conditions
        _filtered_data = []
        _outlier_counts = {}
        _conditions = _df_clean['Greenhouse'].unique()

        for _condition in _conditions:
            _condition_data = _df_clean[_df_clean['Greenhouse'] == _condition]
            _non_outliers, _outlier_count = _remove_outliers_and_count(_condition_data)
            _filtered_data.append(_non_outliers)
            _outlier_counts[_condition] = _outlier_count

        # Add filtered points
        _filtered_df = pd.concat(_filtered_data, ignore_index=True)
        sns.stripplot(data=_filtered_df, x='Greenhouse', y='ratio', ax=_ax, 
                      edgecolor='grey', facecolor='none', linewidth=1, size=3, jitter=True)

        # Create labels and annotations
        _new_labels = []
        for _i, _condition in enumerate(_conditions):
            _condition_data = _df_clean[_df_clean['Greenhouse'] == _condition]['ratio']
            _condition_original = _df[_df['Greenhouse'] == _condition]['ratio']

            _sample_count_clean = len(_condition_data)
            _sample_count_original = len(_condition_original)
            _inf_count = _sample_count_original - _sample_count_clean
            _outlier_count = _outlier_counts[_condition]
            _median_val = _condition_data.median()

            _new_labels.append(f'{_condition}\n(n={_sample_count_clean})\n{_inf_count}inf {_outlier_count}out')

            # Add median
            _ax.text(_i, _median_val, f'Median: {_median_val:.1f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))


        _current_ylim = _ax.get_ylim()
        _y_min, _y_max = _current_ylim
        _margin_factor = 0.15  # 15% margin above
        _y_range = _y_max - _y_min
        _new_y_max = _y_max + (_y_range * _margin_factor)
        _ax.set_ylim(_y_min, _new_y_max)

        # Customize each subplot
        _ax.set_xticklabels(_new_labels, fontsize=9)
        _ax.set_title(f'{_label}', fontsize=10)
        _ax.set_xlabel('Greenhouse', fontsize=9)
        _ax.set_ylabel('Ratio', fontsize=9)

    # Add row labels
    for _row in range(3):
        _axes[_row, 0].text(-0.2, 0.5, _row_labels[_row], 
                           transform=_axes[_row, 0].transAxes, 
                           fontsize=12, fontweight='bold', 
                           va='center', ha='right', rotation=90)

    _fig.suptitle(f'Abundance Ratio Analysis: {dropdown_numerator.value} to {dropdown_denominator.value}', 
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    _fig
    return healthy, infected, intermediate


@app.cell
def _(mo):
    mo.md(r"""### Statstics tests""")
    return


@app.cell
def kruskal_wallis_analysis():
    def kruskal_wallis_analysis(df, value_col='values', category_col='category'):
        """
        Perform Kruskal-Wallis test with post-hoc pairwise comparisons and multiple testing correction.
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        value_col : str, default 'values'
            Name of the column containing values
        category_col : str, default 'category'
            Name of the column containing categories
        Returns:
        --------
        dict : Dictionary containing test results
        """
        import pandas as pd
        import numpy as np
        from scipy import stats
        from scipy.stats import kruskal
        from statsmodels.stats.multitest import multipletests
    
        # Clean the data - remove infinite values
        df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
    
        # Get the three categories
        categories = df_clean[category_col].unique()
    
        # Prepare data for Kruskal-Wallis test
        group_data = []
        for category in categories:
            group_values = df_clean[df_clean[category_col] == category][value_col].values
            group_data.append(group_values)
    
        # Perform Kruskal-Wallis test
        kw_stat, kw_pvalue = kruskal(*group_data)
    
        # Initialize results dictionary
        results = {
            'kw_statistic': kw_stat,
            'kw_pvalue': kw_pvalue,
            'significant': kw_pvalue < 0.05,
            'pairwise_results': None,
            'summary_stats': None
        }
    
        # If significant, perform post-hoc pairwise comparisons
        if kw_pvalue < 0.05:
            # Pairwise comparisons
            pairwise_results = []
            pairs = []
            u_statistics = []
            for i in range(len(categories)):
                for j in range(i+1, len(categories)):
                    cat1, cat2 = categories[i], categories[j]
                    group1 = df_clean[df_clean[category_col] == cat1][value_col]
                    group2 = df_clean[df_clean[category_col] == cat2][value_col]
                
                    # Mann-Whitney U test
                    u_stat, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                    pairwise_results.append(p_val)
                    pairs.append(f"{cat1} vs {cat2}")
                    u_statistics.append(u_stat)
        
            # Multiple testing correction (Bonferroni)
            corrected_pvals = multipletests(pairwise_results, method='bonferroni')[1]
        
            pairwise_df = pd.DataFrame({
                'comparison': pairs,
                'U_statistic': u_statistics,
                'original_pvalue': pairwise_results,
                'corrected_pvalue': corrected_pvals,
                'significant': corrected_pvals < 0.05
            })
        
            results['pairwise_results'] = pairwise_df
    
        # Summary statistics
        summary_stats = df_clean.groupby(category_col)[value_col].agg(['count', 'median', 'mean', 'std']).round(3)
        results['summary_stats'] = summary_stats
    
        return results
    return (kruskal_wallis_analysis,)


@app.cell
def _(
    kruskal_wallis_analysis,
    mod_combined_es_df,
    mod_combined_ls_df,
    mod_combined_ts_df,
    multipletests,
    np,
    pd,
    rel_combined_es_df,
    rel_combined_es_df_pse,
    rel_combined_ls_df,
    rel_combined_ls_df_pse,
    rel_combined_ts_df,
    rel_combined_ts_df_pse,
    stats,
):
    # Define your dataframes and their stages
    datasets = [
        (rel_combined_es_df, "ES", "rel"),
        (rel_combined_ts_df, "TS", "rel"), 
        (rel_combined_ls_df, "LS", "rel"),
        (rel_combined_es_df_pse, "ES", "rel_pse"),
        (rel_combined_ts_df_pse, "TS", "rel_pse"), 
        (rel_combined_ls_df_pse, "LS", "rel_pse"),
        (mod_combined_es_df, "ES", "mod"),
        (mod_combined_ts_df, "TS", "mod"), 
        (mod_combined_ls_df, "LS", "mod")
    ]

    # Run KW-test for each dataset and collect ALL results
    all_kw_results = []
    all_pairwise_results = []

    for df, stage, dataset_type in datasets:
        # Clean data first to get accurate sample sizes
        df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
        conditions = df_clean['Greenhouse'].unique()

        # Run KW-test
        results = kruskal_wallis_analysis(df, value_col='ratio', category_col='Greenhouse')

        # Store main KW results
        main_result = {
            'dataset': dataset_type,
            'stage': stage,
            'dataset_stage': f"{dataset_type}_{stage}",
            'n_total': len(df_clean),
            'n_conditions': len(conditions),
            'conditions': ', '.join(sorted(conditions)),
            'kw_statistic': results['kw_statistic'],
            'kw_pvalue': results['kw_pvalue'],
            'kw_significant': results['significant'],
            'kw_effect_size': results['kw_statistic'] / (len(df_clean) - 1)  # eta-squared approximation
        }

        # Add sample sizes per condition
        for condition in sorted(conditions):
            n_condition = len(df_clean[df_clean['Greenhouse'] == condition])
            main_result[f'n_{condition}'] = n_condition

        all_kw_results.append(main_result)

        # Store pairwise results - ALWAYS, whether KW is significant or not
        if len(conditions) >= 2:  # Need at least 2 groups for pairwise comparisons
            import itertools

            # Calculate all pairwise comparisons manually to ensure we get everything
            pairwise_data = []
            raw_pvalues = []

            for cond1, cond2 in itertools.combinations(sorted(conditions), 2):
                group1 = df_clean[df_clean['Greenhouse'] == cond1]['ratio']
                group2 = df_clean[df_clean['Greenhouse'] == cond2]['ratio']

                # Mann-Whitney U test
                u_stat, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')

                # Effect size (r = Z / sqrt(N))
                z_score = stats.norm.ppf(1 - p_val/2)  # approximate z-score
                effect_size = abs(z_score) / np.sqrt(len(group1) + len(group2))

                pair_result = {
                    'dataset': dataset_type,
                    'stage': stage,
                    'dataset_stage': f"{dataset_type}_{stage}",
                    'comparison': f"{cond1} vs {cond2}",
                    'condition_1': cond1,
                    'condition_2': cond2,
                    'n_group1': len(group1),
                    'n_group2': len(group2),
                    'median_group1': group1.median(),
                    'median_group2': group2.median(),
                    'U_statistic': u_stat,
                    'original_pvalue': p_val,
                    'effect_size_r': effect_size
                }

                pairwise_data.append(pair_result)
                raw_pvalues.append(p_val)

            # Apply FDR multiple testing correction
            if raw_pvalues:
                # FDR correction only
                fdr_corrected = multipletests(raw_pvalues, method='fdr_bh')[1]

                # Add corrected p-values to results
                for i, pair_result in enumerate(pairwise_data):
                    pair_result['fdr_pvalue'] = fdr_corrected[i]
                    pair_result['fdr_significant'] = fdr_corrected[i] < 0.05
                    pair_result['uncorrected_significant'] = raw_pvalues[i] < 0.05

                    # Significance stars based on FDR-corrected p-values
                    fdr_p = fdr_corrected[i]
                    if fdr_p < 0.001:
                        pair_result['significance'] = '***'
                    elif fdr_p < 0.01:
                        pair_result['significance'] = '**'
                    elif fdr_p < 0.05:
                        pair_result['significance'] = '*'
                    else:
                        pair_result['significance'] = 'ns'

            all_pairwise_results.extend(pairwise_data)

    # 1. Main KW-test results
    kw_summary_df = pd.DataFrame(all_kw_results)

    # 2. ALL pairwise comparison results
    pairwise_summary_df = pd.DataFrame(all_pairwise_results)

    # Significant KW tests
    sig_kw = kw_summary_df[kw_summary_df['kw_significant'] == True]

    # Significant pairwise (FDR only)
    sig_pairs_fdr = pairwise_summary_df[pairwise_summary_df['fdr_significant'] == True]
    return (
        all_kw_results,
        all_pairwise_results,
        cond1,
        cond2,
        condition,
        conditions,
        dataset_type,
        datasets,
        df,
        df_clean,
        effect_size,
        fdr_corrected,
        fdr_p,
        group1,
        group2,
        i,
        itertools,
        kw_summary_df,
        main_result,
        n_condition,
        p_val,
        pair_result,
        pairwise_data,
        pairwise_summary_df,
        raw_pvalues,
        results,
        sig_kw,
        sig_pairs_fdr,
        stage,
        u_stat,
        z_score,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
