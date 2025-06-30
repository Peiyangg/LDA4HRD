import marimo

__generated_with = "0.11.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.colors as mcolors
    import random
    import string
    import subprocess
    import os
    from collections import defaultdict
    from gensim import corpora
    from gensim.models import LdaModel, CoherenceModel
    import gensim
    from gensim.corpora import Dictionary

    import little_mallet_wrapper as lmw
    # Path to Mallet binary
    path_to_mallet = '/Users/huopeiyang/Library/CloudStorage/OneDrive-KULeuven/py_working/LDA_workflow/Mallet-202108/bin/mallet'

    import json
    import umap
    import hdbscan

    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import jensenshannon
    from scipy.spatial import distance_matrix
    from pathlib import Path
    return (
        CoherenceModel,
        Dictionary,
        LdaModel,
        Path,
        corpora,
        defaultdict,
        distance_matrix,
        gensim,
        hdbscan,
        jensenshannon,
        json,
        linear_sum_assignment,
        lmw,
        mcolors,
        mo,
        np,
        os,
        path_to_mallet,
        pd,
        plt,
        random,
        sns,
        string,
        subprocess,
        umap,
        warnings,
    )


@app.cell(hide_code=True)
def _(
    CoherenceModel,
    Dictionary,
    defaultdict,
    jensenshannon,
    lmw,
    np,
    pd,
    random,
    string,
):
    # functions

    def generate_unique_ids(num_ids, min_length=5):
        ids = set()
        while len(ids) < num_ids:
            # Generate a random string with the minimal length
            new_id = ''.join(random.choices(string.ascii_lowercase, k=min_length))

            # If the ID already exists, increase the length until we get a unique ID
            while new_id in ids:
                new_id = ''.join(random.choices(string.ascii_lowercase, k=min_length))

            ids.add(new_id)

            # Gradually increase length if all combinations of the current length are used
            if len(ids) == 26**min_length:
                min_length += 1

        return list(ids)


    def load_mallet_model_output_2(topic_distributions_path, word_weights_path):
        # Load topic distributions
        topic_distributions = lmw.load_topic_distributions(topic_distributions_path)

        # Load word weights
        word_topics = []
        with open(word_weights_path, 'r') as f:
            for line in f:
                parts = line.split()
                try:
                    if len(parts) < 2:
                        raise ValueError("Line does not have enough parts")

                    word = parts[1]
                    topic_freq_pairs = parts[2:]

                    for pair in topic_freq_pairs:
                        topic_id, frequency = pair.split(':')
                        word_topics.append((int(topic_id), word, int(frequency)))

                except ValueError as e:
                    # Log or print the problematic line for debugging
                    print(f"Skipping line due to format issues: {line} - Error: {e}")

        return topic_distributions, word_topics


    def process_topic_distributions(topic_distributions, sample_data, desired_order=None):

        # Create a DataFrame from topic distributions
        df_topic_distributions = pd.DataFrame(topic_distributions)

        # Copy the DataFrame for further modifications
        plot_heatmap = df_topic_distributions.copy(deep=True)

        # Extract sample names from the index of sample_data
        samples = sample_data.index.values.tolist()
        sample_time = []
        greenhouse = []
        location = []
        rep = []

        # Extract metadata from sample names
        for name in samples:
            parts = name.split('-')
            sample_time.append(parts[0]) 
            greenhouse.append(parts[1][:-1])  
            location.append(parts[1][-1])  
            rep.append(parts[2]) 

        # Add extracted metadata to the DataFrame
        plot_heatmap['SampleTime'] = sample_time
        plot_heatmap['Greenhouse'] = greenhouse
        plot_heatmap['location'] = location
        plot_heatmap['rep'] = rep

        # Prepare the DataFrame for plotting, calculating median values
        plot_heatmap_mid = plot_heatmap.drop(columns=['location', 'rep'])
        plot_heatmap_mid = plot_heatmap_mid.groupby(['SampleTime', 'Greenhouse']).median().reset_index()

        # Reorder 'Greenhouse' categories if desired_order is provided
        if desired_order is not None:
            other_greenhouses = [g for g in plot_heatmap_mid['Greenhouse'].unique() if g not in desired_order]
            final_order = desired_order + other_greenhouses
            plot_heatmap_mid['Greenhouse'] = pd.Categorical(plot_heatmap_mid['Greenhouse'], categories=final_order, ordered=True)
            plot_heatmap_mid = plot_heatmap_mid.sort_values('Greenhouse')

        return plot_heatmap_mid


    def calculate_perplexity(topic_distributions, epsilon=1e-10):

        perplexities = []

        for distribution in topic_distributions:
            # Ensure the distribution doesn't have zero values by clipping
            distribution = np.clip(distribution, epsilon, 1.0)
            # Calculate the entropy for this distribution
            entropy = -np.sum(np.log(distribution) * distribution)
            # Calculate perplexity and store it
            perplexities.append(np.exp(entropy))

        # Return the average perplexity over all samples
        return np.mean(perplexities)


    def calculate_coherence(word_topics, texts):
        """Calculate coherence score using gensim's CoherenceModel."""
        # Ensure texts are in the correct format
        processed_texts = []
        for text in texts:
            if isinstance(text, str):
                processed_texts.append(text.split())
            elif isinstance(text, list):
                processed_texts.append(text)
            else:
                raise ValueError(f"Unexpected input type: {type(text)}. Expected string or list.")

        # Prepare the data for coherence calculation
        id2word = Dictionary(processed_texts)
        corpus = [id2word.doc2bow(text) for text in processed_texts]

        # Group word_topics by topic number
        topics_dict = defaultdict(list)
        for topic_num, word, freq in word_topics:
            topics_dict[topic_num].append((word, freq))

        # Extract top 10 words for each topic
        topics = []
        for topic_num, word_freqs in topics_dict.items():
            # Sort words by frequency and take the top 10
            top_words = [word for word, _ in sorted(word_freqs, key=lambda x: x[1], reverse=True)[:10]]
            topics.append(top_words)

        # Calculate coherence
        coherence_model = CoherenceModel(topics=topics, texts=processed_texts, dictionary=id2word, coherence='c_v')
        return coherence_model.get_coherence()


    def load_mallet_model_output_2(topic_distributions_path, word_weights_path):
        # Load topic distributions
        topic_distributions = lmw.load_topic_distributions(topic_distributions_path)

        # Load word weights
        word_topics = []
        with open(word_weights_path, 'r') as f:
            for line in f:
                parts = line.split()
                try:
                    if len(parts) < 2:
                        raise ValueError("Line does not have enough parts")

                    word = parts[1]
                    topic_freq_pairs = parts[2:]

                    for pair in topic_freq_pairs:
                        topic_id, frequency = pair.split(':')
                        word_topics.append((int(topic_id), word, int(frequency)))

                except ValueError as e:
                    # Log or print the problematic line for debugging
                    print(f"Skipping line due to format issues: {line} - Error: {e}")

        return topic_distributions, word_topics

    def compute_jsd_matrix_rows(df):
        m = df.shape[0]  # Number of rows
        jsd_matrix = np.zeros((m, m))

        for i in range(m):
            for j in range(m):
                if i != j:
                    # Calculate JSD between rows i and j
                    jsd_matrix[i, j] = jensenshannon(df.iloc[i, :], df.iloc[j, :])**2

        return pd.DataFrame(jsd_matrix, columns=df.index, index=df.index)
    return (
        calculate_coherence,
        calculate_perplexity,
        compute_jsd_matrix_rows,
        generate_unique_ids,
        load_mallet_model_output_2,
        process_topic_distributions,
    )


@app.cell(hide_code=True)
def _(pd):
    # def update_genus_rockwool(row, unknown_count):
    #     if row['Genus'] == 'uncultured':
    #         return f"{row['Family']}_uncultured" if pd.notna(row['Family']) else "unknown_uncultured"
    #     elif pd.isna(row['Genus']):
    #         if pd.notna(row['Family']):
    #             return f"{row['Family']}_unknown"
    #         else:
    #             unknown_count[0] += 1  # Update the count in a mutable list to keep state
    #             return f"unknown_{unknown_count[0]}_rockwool"
    #     return row['Genus']

    def update_genus_rockwool(row, unknown_count):
        if row['Genus'] == 'uncultured':
            if pd.notna(row['Family']) and row['Family'] != 'uncultured':
                return f"{row['Family']}_uncultured"
            else:
                unknown_count[0] += 1  # Update the count in a mutable list to keep state
                return f"unknown_{unknown_count[0]}_rockwool"
        elif pd.isna(row['Genus']):
            if pd.notna(row['Family']) and row['Family'] != 'uncultured':
                return f"{row['Family']}_unknown"
            else:
                unknown_count[0] += 1  # Update the count in a mutable list to keep state
                return f"unknown_{unknown_count[0]}_rockwool"
        return row['Genus']
    unknown_count = [0]

    def assign_unknown_labels_rockwool(df, column):
        unknown_count = 0
        def label_unknown(value):
            nonlocal unknown_count
            if pd.isna(value):
                unknown_count += 1
                return f'unknown_{unknown_count}'
            return value

        df[column] = df[column].apply(label_unknown)
        return df
    return assign_unknown_labels_rockwool, unknown_count, update_genus_rockwool


@app.cell
def _(os):
    asvtable_path           ='/Users/huopeiyang/Desktop/qiime_env/deblur_output/exported_data/feature-table.tsv'
    taxonomy_path           ='/Users/huopeiyang/Library/CloudStorage/OneDrive-KULeuven/py_working/LDA_workflow/Hans/intermediate/used_taxa.csv'
    base_directory          = '/Users/huopeiyang/Library/CloudStorage/OneDrive-KULeuven/py_working/LDA_workflow/Hans2'

    # Create new directory for LDA analysis
    intermediate_directory  =   os.path.join(base_directory, 'intermediate')
    loop_directory          =   os.path.join(base_directory, 'lda_loop')
    lda_directory           =   os.path.join(base_directory, 'lda_results')

    os.makedirs(intermediate_directory, exist_ok=True)
    os.makedirs(loop_directory, exist_ok=True)
    os.makedirs(lda_directory, exist_ok=True)


    # Now you can use lda_directory as your output path
    loop_output_directory_path = loop_directory
    Loop_2tables_directory_path = lda_directory

    path_to_training_data           = loop_output_directory_path + '/training.txt'
    path_to_formatted_training_data = loop_output_directory_path + '/mallet.training'

    DC_range = range(2, 21)
    return (
        DC_range,
        Loop_2tables_directory_path,
        asvtable_path,
        base_directory,
        intermediate_directory,
        lda_directory,
        loop_directory,
        loop_output_directory_path,
        path_to_formatted_training_data,
        path_to_training_data,
        taxonomy_path,
    )


@app.cell
def _():
    # sampletable = sampletable.iloc[:, :-1]
    # asvtable = sampletable.T
    # asv_id = pd.read_csv(taxonomy_path, index_col=0)
    return


@app.cell
def _(asvtable_path, pd, taxonomy_path):
    asvtable=pd.read_csv(asvtable_path, sep='\t', skiprows=1, index_col=0)
    sampletable = asvtable.T
    asv_id = pd.read_csv(taxonomy_path, index_col=0)
    return asv_id, asvtable, sampletable


@app.cell
def _(
    asv_id,
    asvtable,
    generate_unique_ids,
    intermediate_directory,
    json,
    sampletable,
):
    # replace complex qiime2 id / sequence to new id
    asvs = asvtable.index.values.tolist()
    new_ids = generate_unique_ids(len(asvs))
    genera = asv_id['Genus'].values.tolist()
    header_dict = dict(zip(new_ids, genera))
    asvid_dict = dict(zip(new_ids, asvs))

    with open(intermediate_directory + '/id_genera_mapping.json', 'w') as k:
        json.dump(header_dict, k)

    with open(intermediate_directory + '/id_asv_mapping.json', 'w') as k:
        json.dump(asvid_dict, k)

    sampletable.columns = new_ids
    sampletable.to_csv(intermediate_directory + '/inter_sampletable.csv', index=True)
    return asvid_dict, asvs, genera, header_dict, k, new_ids


@app.cell
def _(path_to_training_data, sampletable):
    # Create Mallet input documents
    doc_list=[]
    # Each sample becomes a document where ASVs are repeated based on their abundance
    for index, row in sampletable.iterrows():
        doc = []
        for asvs_id1, abundance in row.items():  # Using items() instead of iteritems()
            if abundance > 0:
                doc.extend([str(asvs_id1)] * int(abundance))
        doc_list.append(doc)

    flattened_nested_list = [' '.join(sublist) for sublist in doc_list]

    with open(path_to_training_data, 'w') as f:
        for document in flattened_nested_list:
            f.write(document + '\n')
    return (
        abundance,
        asvs_id1,
        doc,
        doc_list,
        document,
        f,
        flattened_nested_list,
        index,
        row,
    )


@app.cell
def _(
    DC_range,
    Loop_2tables_directory_path,
    asvid_dict,
    calculate_coherence,
    calculate_perplexity,
    flattened_nested_list,
    header_dict,
    lmw,
    load_mallet_model_output_2,
    loop_output_directory_path,
    path_to_formatted_training_data,
    path_to_mallet,
    path_to_training_data,
    pd,
    sampletable,
    subprocess,
):
    # Initialize empty DataFrames to store all results
    all_df_pivot_rel = pd.DataFrame()
    all_df_probabilities_rel = pd.DataFrame()
    all_metrics = pd.DataFrame(columns=['Num_Topics', 'Perplexity', 'Coherence'])

    # Define the range of topics
    for num_topics in DC_range:  # Loop from 2 to 20 topics
        # Define file paths based on the current number of topics
        path_to_model = loop_output_directory_path + f'/mallet.model.{num_topics}'
        path_to_topic_keys = loop_output_directory_path + f'/mallet.topic_keys.{num_topics}'
        path_to_topic_distributions = loop_output_directory_path + f'/mallet.topic_distributions.{num_topics}'
        path_to_word_weights = loop_output_directory_path + f'/mallet.word_weights.{num_topics}'
        path_to_diagnostics = loop_output_directory_path + f'/mallet.diagnostics.{num_topics}.xml'

        # Define paths for individual model results
        path_to_DirichletComponentProbabilities = Loop_2tables_directory_path + f'/DirichletComponentProbabilities_{num_topics}.csv'
        path_to_TaxaProbabilities = Loop_2tables_directory_path + f'/TaxaProbabilities_{num_topics}.csv'
        path_to_ASVProbabilities = Loop_2tables_directory_path + f'/ASVProbabilities_{num_topics}.csv'

        # Training model with optimizing
        lmw.import_data(path_to_mallet,
                        path_to_training_data,
                        path_to_formatted_training_data,
                        flattened_nested_list)

        # Construct the MALLET command
        mallet_command = [
            path_to_mallet,
            'train-topics',
            '--input', path_to_formatted_training_data,
            '--num-topics', str(num_topics),  # Change number of topics as needed
            '--output-state', path_to_model,
            '--output-topic-keys', path_to_topic_keys,
            '--output-doc-topics', path_to_topic_distributions,
            '--word-topic-counts-file', path_to_word_weights,
            '--diagnostics-file', path_to_diagnostics,
            '--optimize-interval', '10',  # Enable alpha optimization every 10 iterations
            '--num-iterations', '1000',  # Number of iterations, adjust as needed
            '--random-seed', '43'
        ]

        # Run the MALLET command
        print(f"Running MALLET for {num_topics} topics...")
        subprocess.run(mallet_command, check=True)
        print(f"Completed MALLET for {num_topics} topics.")

        # Create index names for the topics
        topic_index = []
        for a in range(1, num_topics + 1):
            topic_index.append(str(num_topics) + '_' + str(a))

        # Load the MALLET model output
        topic_distributions, word_topics = load_mallet_model_output_2(path_to_topic_distributions, path_to_word_weights)

        # Map term IDs to genus names and create a new list of renamed word topics
        rename_word_topics = []
        for topic, term, freq in word_topics:
            new_term = header_dict.get(term, term)  # Use header_dict to map term IDs to genus names
            rename_word_topics.append((topic, new_term, freq))

        rename_asv_topics = []
        for topic_asv, term_asv, freq_asv in word_topics:
            new_term_asv = asvid_dict.get(term_asv, term_asv)  # Use term_asv to map term IDs to ASV
            rename_asv_topics.append((topic_asv, new_term_asv, freq_asv))

        # Convert the renamed word topics into a DataFrame
        df = pd.DataFrame(rename_word_topics, columns=['Topic', 'Term', 'Frequency'])
        df_asv= pd.DataFrame(rename_asv_topics, columns=['Topic', 'Term', 'Frequency'])

        # Pivot the DataFrame to get the desired format (Term columns, Topic rows)
        df_pivot = df.pivot_table(index='Topic', columns='Term', values='Frequency', fill_value=0)
        df_asv_pivot = df_asv.pivot_table(index='Topic', columns='Term', values='Frequency', fill_value=0)

        # Merge columns with the same header by summing them
        df_pivot_grouped = df_pivot.groupby(level=0, axis=1).sum()

        # Normalize the DataFrame to get probabilities
        df_probabilities = df_pivot_grouped.div(df_pivot_grouped.sum(axis=1), axis=0)
        df_asv_probabilities = df_asv_pivot.div(df_asv_pivot.sum(axis=1), axis=0)

        # Rename the index with the generated index names
        df_pivot_grouped.index = topic_index
        df_probabilities.index = topic_index
        df_asv_probabilities.index = topic_index

        # Create and save the topic distribution DataFrame for this specific model
        df_topic_dist = pd.DataFrame(
            topic_distributions,  # Your nested list of topic probabilities
            index=sampletable.index,  # Use sampletable's index
            columns=topic_index  # Your list of topic names
        )

        # Save individual model results
        df_topic_dist.to_csv(path_to_DirichletComponentProbabilities, index=True)
        df_probabilities.to_csv(path_to_TaxaProbabilities, index=True)
        df_asv_probabilities.to_csv(path_to_ASVProbabilities, index=True)
        print(f"Saved individual model results for {num_topics} topics.")

        # Concatenate the results to the overall DataFrames (as in your original code)
        all_df_pivot_rel = pd.concat([all_df_pivot_rel, df_pivot_grouped])
        all_df_probabilities_rel = pd.concat([all_df_probabilities_rel, df_probabilities])

        perplexity = calculate_perplexity(topic_distributions)
        coherence = calculate_coherence(word_topics, flattened_nested_list)

        new_row = pd.DataFrame([{
            'Num_Topics': num_topics,
            'Perplexity': perplexity,
            'Coherence': coherence
        }])
        all_metrics = pd.concat([all_metrics, new_row], ignore_index=True)

        print(f"Processed and appended results for {num_topics} topics.")

    # After the loop, save the final combined DataFrames to CSV files
    all_df_pivot_rel.to_csv(loop_output_directory_path + '/all_topic_pivots_rel_2_20.csv')
    all_df_probabilities_rel.to_csv(loop_output_directory_path + '/all_topic_probabilities_rel_2_20.csv')
    all_metrics.to_csv(loop_output_directory_path + '/all_topic_metrics_2_20.csv')

    print("Saved final combined DataFrames.")
    return (
        a,
        all_df_pivot_rel,
        all_df_probabilities_rel,
        all_metrics,
        coherence,
        df,
        df_asv,
        df_asv_pivot,
        df_asv_probabilities,
        df_pivot,
        df_pivot_grouped,
        df_probabilities,
        df_topic_dist,
        freq,
        freq_asv,
        mallet_command,
        new_row,
        new_term,
        new_term_asv,
        num_topics,
        path_to_ASVProbabilities,
        path_to_DirichletComponentProbabilities,
        path_to_TaxaProbabilities,
        path_to_diagnostics,
        path_to_model,
        path_to_topic_distributions,
        path_to_topic_keys,
        path_to_word_weights,
        perplexity,
        rename_asv_topics,
        rename_word_topics,
        term,
        term_asv,
        topic,
        topic_asv,
        topic_distributions,
        topic_index,
        word_topics,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
