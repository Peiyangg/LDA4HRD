import marimo

__generated_with = "0.14.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import os
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.cluster import hierarchy
    from scipy.spatial import distance
    return (
        LinearSegmentedColormap,
        distance,
        hierarchy,
        mcolors,
        mo,
        np,
        pd,
        plt,
        sns,
    )


@app.cell
def _():
    path_to_DirichletComponentProbabilities            = 'data/DirichletComponentProbabilities_9.csv'
    path_to_TaxaProbabilities                          = 'data/9MC_ASV.csv'
    return (path_to_TaxaProbabilities,)


@app.cell
def _(pd):
    DMP = pd.read_csv('data/sample_MC.csv', index_col=0)
    metadata=pd.read_csv('data/metadata.csv',index_col=0)
    return DMP, metadata


@app.cell
def _(mo):
    mo.md(r"""## Visualizations of two heatmaps""")
    return


@app.cell
def _(mo):
    mo.md(r"""### MC-Sample Heatmap""")
    return


@app.cell(hide_code=True)
def _(pd):
    def prepare_heatmap_data(DM_distributions, metadata_df, id_column='averageID', headers_to_include=None):
        """
        Create a multi-index heatmap table from distributions and metadata

        Parameters:
        -----------
        DM_distributions : list of lists
            Nested list where each child list is one sample's topic distribution
        metadata_df : DataFrame
            DataFrame containing metadata for each sample
        id_column : str, default='averageID'
            Column name to use for matching/joining data
        headers_to_include : list, optional
            List of column names to include in the multi-index header
            If None, all columns in metadata_df will be used

        Returns:
        --------
        DataFrame with multi-index columns
        """
        # Convert nested list to DataFrame and transpose
        distributions_df = pd.DataFrame(DM_distributions)
        multiheader = distributions_df.T

        # If no headers specified, use all columns except id_column
        if headers_to_include is None:
            headers_to_include = [col for col in metadata_df.columns if col != id_column]

        # Fill NaN values in metadata with 0
        metadata_df_filled = metadata_df.fillna(0)

        # Get unique IDs from the metadata that match our samples
        sample_ids = metadata_df_filled[id_column].values[:len(DM_distributions)]

        # Create a list of tuples for the multi-index
        header_tuples = []
        for idx, id_val in enumerate(sample_ids):
            # Get the matching metadata row
            metadata_row = metadata_df_filled[metadata_df_filled[id_column] == id_val]

            if not metadata_row.empty:
                # Extract values for each header
                tuple_values = [metadata_row[col].values[0] for col in headers_to_include]
                # Add the ID as the last element
                tuple_values.append(id_val)
                header_tuples.append(tuple(tuple_values))

        # Create column names for the multi-index (headers_to_include + id_column)
        multi_columns = headers_to_include + [id_column]

        # Create a DataFrame for the multi-index
        header_df = pd.DataFrame(header_tuples, columns=multi_columns)

        # Create the MultiIndex
        multi_index = pd.MultiIndex.from_frame(header_df)

        # Set the multi-index on the columns
        multiheader.columns = multi_index

        return multiheader
    return (prepare_heatmap_data,)


@app.cell(hide_code=True)
def _(LinearSegmentedColormap, mcolors, np, pd, plt, sns):
    def create_clustered_heatmap(multiheader, id_column=None, headers_to_color=None, custom_colors=None, 
                                 continuous_headers=None, figsize=(8.27, 11.69), output_path=None, 
                                 legend_path=None, show_dendrograms=False, continuous_cmaps=None,
                                 continuous_colors=None, order_by='clustered'):
        """
        Create a clustermap with color annotations for specified headers, supporting both categorical
        and continuous color scales. Saves the legend to a separate file.

        Parameters:
        -----------
        multiheader : DataFrame
            DataFrame with multi-index columns to visualize
        id_column : str, optional
            Column name to use for x-axis labels
            If None, uses the last level of the MultiIndex
        headers_to_color : list, optional
            List of column headers to use for color annotations
            If None, uses all headers except the id_column
        custom_colors : dict, optional
            Dictionary mapping header names to dictionaries of value-color pairs
            Example: {'Diagnosis': {'healthy': '#7B8B6F', 'infested': '#965454'}}
        continuous_headers : list, optional
            List of headers that should use continuous color scales instead of categorical
            These headers should contain numeric data
        figsize : tuple, default=(8.27, 11.69)
            Figure size in inches (default is A4)
        output_path : str, optional
            Path to save the figure, if None, figure is not saved
        legend_path : str, optional
            Path to save the separate legend file, if None, legend is not saved
        show_dendrograms : bool, default=False
            Whether to show the dendrograms
        continuous_cmaps : dict, optional
            Dictionary mapping header names to specific colormap names or custom colormaps
            Example: {'Temperature': 'viridis', 'pH': 'coolwarm'}
        continuous_colors : dict, optional
            Dictionary mapping header names to pairs of colors for creating custom colormaps
            Example: {'Temperature': ['white', 'red'], 'pH': ['blue', 'yellow']}
        order_by : str, default='clustered'
            Controls how columns are ordered in the heatmap:
            - 'clustered': Use hierarchical clustering (default behavior)
            - Any header name from headers_to_color: Order columns by that metadata

        Returns:
        --------
        tuple
            (ClusterGrid object, Legend figure object)
        """
        # Set default values if not provided
        if id_column is None:
            id_column = multiheader.columns.names[-1]

        if headers_to_color is None:
            headers_to_color = [name for name in multiheader.columns.names if name != id_column]

        if continuous_headers is None:
            continuous_headers = []

        if continuous_cmaps is None:
            continuous_cmaps = {}

        if continuous_colors is None:
            continuous_colors = {}

        # Get unique values for each header and create color palettes
        color_maps = {}
        colors_dict = {}

        # Define gray color for missing values
        missing_color = '#D3D3D3'  # Light gray

        # Store any custom colormaps created during execution
        created_colormaps = {}

        # Create a unique palette for each header
        for header in headers_to_color:
            # Get unique values
            header_values = multiheader.columns.get_level_values(header)

            # Check if this header should use a continuous color scale
            if header in continuous_headers:
                # Filter out non-numeric, zero, and missing values for finding min/max
                numeric_values = pd.to_numeric(header_values, errors='coerce')
                valid_mask = ~np.isnan(numeric_values) & (numeric_values != 0)

                if not any(valid_mask):
                    # If no valid numeric values, fall back to categorical
                    print(f"Warning: Header '{header}' specified as continuous but contains no valid numeric values. Using categorical colors.")
                    is_continuous = False
                else:
                    is_continuous = True
                    # Get min and max for normalization
                    vmin = np.min(numeric_values[valid_mask])
                    vmax = np.max(numeric_values[valid_mask])

                    # Create a normalization function
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

                    # Determine which colormap to use for this header
                    if header in continuous_colors:
                        # User provided custom colors to create a colormap
                        color_pair = continuous_colors[header]
                        cmap_name = f"custom_{header.replace(' ', '_')}"

                        # Create a custom colormap from the provided colors
                        if isinstance(color_pair, (list, tuple)) and len(color_pair) >= 2:
                            cmap = LinearSegmentedColormap.from_list(cmap_name, color_pair)
                            created_colormaps[cmap_name] = cmap
                        else:
                            print(f"Warning: Invalid color pair for '{header}'. Expected [color1, color2]. Using default colormap.")
                            cmap = plt.cm.viridis

                    elif header in continuous_cmaps:
                        # User specified a specific colormap
                        cmap_name = continuous_cmaps[header]
                        if isinstance(cmap_name, str):
                            try:
                                # Try to get a matplotlib colormap
                                cmap = plt.get_cmap(cmap_name)
                            except:
                                print(f"Warning: Colormap '{cmap_name}' not found. Using default.")
                                cmap = plt.cm.viridis
                        else:
                            # Assume it's already a colormap object
                            cmap = cmap_name
                    else:
                        # Use default colormap for continuous data
                        # Use a default colormap from matplotlib's built-in options
                        standard_cmaps = [plt.cm.viridis, plt.cm.plasma, plt.cm.inferno, 
                                         plt.cm.magma, plt.cm.cividis, plt.cm.cool, 
                                         plt.cm.YlGnBu, plt.cm.YlOrRd]

                        # Assign based on position in continuous_headers list
                        cmap_index = continuous_headers.index(header) % len(standard_cmaps)
                        cmap = standard_cmaps[cmap_index]

                    # Store info for legend creation
                    color_maps[header] = {
                        'type': 'continuous',
                        'cmap': cmap,
                        'norm': norm,
                        'vmin': vmin,
                        'vmax': vmax
                    }

                    # Map values to colors
                    colors = []
                    for val in header_values:
                        try:
                            num_val = float(val)
                            if np.isnan(num_val) or num_val == 0:  # Treat zero as missing value
                                colors.append(missing_color)
                            else:
                                colors.append(cmap(norm(num_val)))
                        except (ValueError, TypeError):
                            colors.append(missing_color)

                    colors_dict[header] = pd.Series(colors, index=multiheader.columns)
                    continue  # Skip the categorical color assignment
            else:
                is_continuous = False

            # Categorical coloring (for non-continuous headers)
            # Filter out None, NaN, and empty string values for color assignment
            unique_values = header_values.unique()
            valid_values = [v for v in unique_values if pd.notna(v) and v != '']

            if pd.api.types.is_numeric_dtype(np.array(valid_values, dtype=object)):
                valid_values = sorted(valid_values)

            # Use custom colors if provided, otherwise generate a palette
            if custom_colors and header in custom_colors:
                # Use custom color dictionary for this header
                lut = custom_colors[header].copy()  # Make a copy to avoid modifying the original
            else:
                # Use default palette for categorical data
                palette = sns.color_palette("deep", len(valid_values))
                lut = dict(zip(valid_values, palette))

            # Add a color for missing values (None, NaN, or empty string)
            lut[None] = missing_color
            lut[np.nan] = missing_color
            lut[''] = missing_color

            # Store the color lookup table
            color_maps[header] = {
                'type': 'categorical',
                'lut': lut
            }

            # Map colors to columns, handling missing values
            colors = []
            for val in header_values:
                if pd.isna(val) or val == '':
                    colors.append(missing_color)
                elif val in lut:
                    colors.append(lut[val])
                else:
                    colors.append(missing_color)  # If value not in lut for some reason

            colors_dict[header] = pd.Series(colors, index=multiheader.columns)

        # Create a DataFrame of colors
        multi_colors = pd.DataFrame(colors_dict)

        # Determine column ordering based on order_by parameter
        if order_by == 'clustered':
            # Use clustering (original behavior)
            col_cluster = True
            column_order = None
        else:
            # Order by specific metadata
            if order_by not in headers_to_color:
                print(f"Warning: '{order_by}' not found in headers_to_color. Using clustered ordering.")
                col_cluster = True
                column_order = None
            else:
                col_cluster = False
                # Get the values for the specified header
                order_values = multiheader.columns.get_level_values(order_by)

                # Create a DataFrame to sort by
                sort_df = pd.DataFrame({
                    'index': range(len(multiheader.columns)),
                    'sort_key': order_values
                })

                # Sort the DataFrame
                if order_by in continuous_headers:
                    # For continuous data, convert to numeric and sort
                    sort_df['sort_key'] = pd.to_numeric(sort_df['sort_key'], errors='coerce')
                    # Handle NaN values by putting them at the end
                    sort_df = sort_df.sort_values('sort_key', na_position='last')
                else:
                    # For categorical data, sort alphabetically
                    sort_df = sort_df.sort_values('sort_key', na_position='last')

                column_order = sort_df['index'].tolist()

        # Create the clustermap
        if column_order is not None:
            # Reorder the data and colors based on the specified order
            ordered_data = multiheader.iloc[:, column_order]
            ordered_colors = multi_colors.iloc[column_order]

            g = sns.clustermap(
                ordered_data, 
                center=0, 
                cmap="vlag",
                col_colors=ordered_colors,
                dendrogram_ratio=(.1, .2),
                cbar_pos=(-.08, .50, .03, .2),
                linewidths=.75, 
                figsize=figsize,
                col_cluster=col_cluster, 
                row_cluster=True
            )

            # For ordered data, labels are already in the correct order
            new_labels = [ordered_data.columns.get_level_values(id_column)[i] for i in range(len(ordered_data.columns))]
        else:
            # Use original clustering approach
            g = sns.clustermap(
                multiheader, 
                center=0, 
                cmap="vlag",
                col_colors=multi_colors,
                dendrogram_ratio=(.1, .2),
                cbar_pos=(-.08, .50, .03, .2),
                linewidths=.75, 
                figsize=figsize,
                col_cluster=col_cluster, 
                row_cluster=True
            )

            # Get the specified ID column values for x-tick labels (reordered by clustering)
            new_labels = [multiheader.columns.get_level_values(id_column)[i] for i in g.dendrogram_col.reordered_ind]

        # Set the x-tick positions and labels
        g.ax_heatmap.set_xticks(np.arange(len(new_labels)) + 0.5)

        # Then set the labels for these positions
        g.ax_heatmap.set_xticklabels(new_labels, fontsize=4, rotation=45, ha='right')

        # Make tick marks thinner and shorter
        g.ax_heatmap.tick_params(axis='x', which='major', length=3, width=0.5, bottom=True)
        g.ax_heatmap.xaxis.set_tick_params(labeltop=False, top=False)

        # Show/hide dendrograms based on parameter
        g.ax_row_dendrogram.set_visible(show_dendrograms)
        g.ax_col_dendrogram.set_visible(show_dendrograms)

        # Save the figure if path is provided
        if output_path:
            g.savefig(output_path, dpi=300, format='svg')

        # Create separate legend file if path is provided
        legend_fig = None
        if legend_path:
            legend_fig = create_legend_file(
                color_maps=color_maps,
                headers_to_color=headers_to_color,
                continuous_headers=continuous_headers,
                missing_color=missing_color,
                output_path=legend_path
            )

        # Return the clustermap and legend figure
        return g, legend_fig
    return (create_clustered_heatmap,)


@app.function(hide_code=True)
def create_legend_file(color_maps, headers_to_color, continuous_headers, missing_color, output_path=None):
    """
    Create a separate legend file with vertical organization

    Parameters:
    -----------
    color_maps : dict
        Dictionary containing color mapping information
    headers_to_color : list
        List of header names to create legends for
    continuous_headers : list
        List of headers that use continuous color scales
    missing_color : str
        Hex color code for missing values
    output_path : str, optional
        Path to save the legend file, if None, figure is displayed but not saved

    Returns:
    --------
    matplotlib.figure.Figure
        The legend figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.cm import ScalarMappable

    # Create figure for the legends
    fig_height = 1 + 0.8 * len(headers_to_color)  # Dynamic height based on number of headers
    fig, ax = plt.subplots(figsize=(5, fig_height))
    ax.axis('off')  # Hide the axes

    # Configure background
    fig.patch.set_facecolor('white')

    # Vertical spacing parameters
    y_start = 0.95
    y_step = 0.9 / len(headers_to_color)

    legends = []

    # Track headers we've already seen to handle duplicates
    seen_headers = {}

    # Create legends in vertical stack
    for i, header in enumerate(headers_to_color):
        # Handle duplicate header names by creating unique titles
        if header in seen_headers:
            seen_headers[header] += 1
            display_title = f"{header} ({seen_headers[header]})"
        else:
            seen_headers[header] = 1
            display_title = header

        # Calculate y position
        y_pos = y_start - i * y_step

        # Check if this is a continuous or categorical header
        if header in continuous_headers and color_maps[header]['type'] == 'continuous':
            # Get colormap info
            cmap_info = color_maps[header]
            cmap = cmap_info['cmap']
            norm = cmap_info['norm']

            # Create a new axis for the colorbar
            cax_height = 0.02
            cax_width = 0.3
            cax = fig.add_axes([0.35, y_pos - cax_height/2, cax_width, cax_height])

            # Create the colorbar
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')

            # Add title
            cbar.set_label(display_title, fontsize=10, labelpad=8)
            cbar.ax.tick_params(labelsize=8)

            legends.append(cbar)
        else:
            # Categorical legend
            lut = color_maps[header]['lut']

            # Filter out None/NaN keys for the legend
            filtered_lut = {k: v for k, v in lut.items() if k is not None and not (isinstance(k, float) and np.isnan(k)) and k != ''}

            # Add a "Missing" entry if there were missing values
            has_missing = None in lut or np.nan in lut or '' in lut
            if has_missing:
                filtered_lut["Missing"] = missing_color

            # Create handles for the legend
            handles = [plt.Rectangle((0,0), 1.5, 1.5, color=color, ec="k") for label, color in filtered_lut.items()]
            labels = list(filtered_lut.keys())

            # Add legend
            num_items = len(filtered_lut)

            # Determine number of columns based on number of items
            legend_ncol = 1
            if num_items > 6:
                legend_ncol = 2
            if num_items > 12:
                legend_ncol = 3

            leg = ax.legend(
                handles, 
                labels, 
                title=display_title,
                loc="center", 
                bbox_to_anchor=(0.5, y_pos),
                ncol=legend_ncol,
                frameon=True, 
                fontsize=8,
                title_fontsize=10
            )

            # Need to manually add the legend
            ax.add_artist(leg)
            legends.append(leg)

    plt.tight_layout()

    # Save figure if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='svg')

    return fig


@app.cell
def _(DMP, metadata, prepare_heatmap_data):
    multiheader = prepare_heatmap_data(
        DM_distributions=DMP,
        metadata_df=metadata,
        id_column="sampleID",
        headers_to_include=['Diagnosis','SampleTime','Location_int']
    )

    group_colors = {
        'Diagnosis': {
            'healthy': '#abdda4',
            'pre-symptomatic': '#fdae61',  
            'infected': '#d7191c',
            'intermediate': '#4a1486'
        },
        'Location_int': {
            1: '#4a1486',
            2: '#807dba',
            3: '#bcbddc'
        },
    }
    return group_colors, multiheader


@app.cell
def _(create_clustered_heatmap, group_colors, multiheader):
    order_by='clustered' # other options: 'Diagnosis', 'SampleTime', 'Location_int'
    g, leg = create_clustered_heatmap(multiheader, 
                                    headers_to_color=['Diagnosis','SampleTime','Location_int'],
                                    continuous_headers=['SampleTime'],
                                    continuous_cmaps={'SampleTime': 'Blues'},
                                    custom_colors = group_colors,
                                    figsize=(12, 10),
                                    output_path=f"viz/clustered_selected_headers_{order_by}.svg",
                                    legend_path="viz/clustered-selected_headers-legend.svg",
                                      order_by=order_by
                                     )

    return


@app.cell
def _(mo):
    mo.md(r"""### MC-ASV Heatmap""")
    return


@app.cell(hide_code=True)
def _(distance, hierarchy, mcolors, pd, plt, sns):
    def create_clustered_heatmap_MC_ASV(top_tokens_df, 
                                figsize=(20, 16), 
                                vmin=0, 
                                vmax=0.4, 
                                cmap=plt.cm.PuBu, 
                                save_path='topic_word_heatmap.png',
                                epsilon=1e-10):


        # Make a copy and fill any NaN values with 0 to avoid distance computation issues
        top_tokens_df_customized = top_tokens_df.copy()
        df_for_clustering = top_tokens_df_customized.fillna(0)

        # Add epsilon to rows that are all zeros
        if (df_for_clustering.sum(axis=1) == 0).any():
            zero_rows = df_for_clustering.sum(axis=1) == 0
            df_for_clustering.loc[zero_rows] = epsilon

        # Add epsilon to columns that are all zeros
        if (df_for_clustering.sum(axis=0) == 0).any():
            zero_cols = df_for_clustering.sum(axis=0) == 0
            df_for_clustering.loc[:, zero_cols] = epsilon

        # Compute clustering for rows
        row_linkage = hierarchy.linkage(distance.pdist(df_for_clustering.values), method='ward')
        row_order = hierarchy.dendrogram(row_linkage, no_plot=True)['leaves']

        # Compute the clustering for columns (words)
        col_linkage = hierarchy.linkage(distance.pdist(df_for_clustering.values.T), method='ward')
        col_order = hierarchy.dendrogram(col_linkage, no_plot=True)['leaves']

        # Reorder the original dataframe according to the clustering
        df_clustered = top_tokens_df_customized.iloc[row_order, col_order]

        # For visualization, fill NaN values with zeros
        df_clustered_filled = df_clustered.fillna(0)

        # Set up the plot
        plt.figure(figsize=figsize)

        # Create a custom normalization
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Create the heatmap with annotations and continuous colors
        ax = sns.heatmap(df_clustered_filled, 
                        cmap=cmap,
                        norm=norm,
                        cbar_kws={'label': 'Probability',
                                 'ticks': [0, 0.02, 0.05, 0.1, vmax],
                                 'shrink': 0.5,
                                 'fraction': 0.046,
                                 'pad': 0.04,
                                 'aspect': 20},
                        annot=True,
                        fmt='.3f',
                        annot_kws={'size': 6},
                        square=True,
                        mask=pd.isna(df_clustered))

        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # Show the plot
        plt.show()

        return df_clustered
    return (create_clustered_heatmap_MC_ASV,)


@app.cell
def _(path_to_TaxaProbabilities, pd):
    TP = pd.read_csv(path_to_TaxaProbabilities, index_col=0)
    def get_top_tokens(row, top_n=8):
        return row.nlargest(top_n)
    top_token_df_id = TP.apply(get_top_tokens, axis=1,top_n=8)

    columns_to_keep = []

    for column in top_token_df_id.columns:
        # Get non-NaN values in the column
        non_nan_values = top_token_df_id[column].dropna()

        # Check if any non-NaN value is >= 0.2
        if (non_nan_values >= 0.010).any():
            columns_to_keep.append(column)

    # Filter the DataFrame to keep only columns with at least one value >= 0.01
    filtered_df = top_token_df_id[columns_to_keep]
    return (filtered_df,)


@app.cell
def _(create_clustered_heatmap_MC_ASV, filtered_df, plt):
    clustered_df = create_clustered_heatmap_MC_ASV(
        filtered_df.T,
        figsize=(16, 12),
        vmin=0,
        vmax=0.2,
        cmap = plt.cm.PuBu,
        save_path='viz/top8asv_0.01.svg',
        epsilon=1e-8
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
