import psychai.data_preparation.preprocessing
import seaborn as sns
from scipy.stats import ttest_1samp, ttest_ind, pearsonr, f_oneway  # For statistical tests
from itertools import combinations  # For generating combinations of values
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Chart:
    
    def __init__(self):
        """
        """
        # self.df = df  # The DataFrame that holds file paths or data identifiers
        


    def count_bar_chart(self, df, features_to_plot= [], group_label = "",  number_of_top_features = 0):

        if number_of_top_features > 0:
            features_to_plot = df.select_dtypes(include=['number']).columns[0:number_of_top_features]

        if len(features_to_plot) == 0:
            return
        
        # Count plot for target variable 'Group'
        # This helps check for class imbalance in the target variable
        sns.countplot(x=df[group_label], data=df[features_to_plot], palette='Set2')
        plt.title('Distribution of Target Variable - Group')
        plt.show()

    def draw_histogram(self, df,  features_to_plot= [], number_of_top_features = 0):

        if number_of_top_features > 0:
            features_to_plot = df.select_dtypes(include=['number']).columns[0:number_of_top_features]

        if len(features_to_plot) == 0:
            return

        df[features_to_plot].hist(figsize=(6, 4), bins=5, edgecolor='black')
        plt.suptitle("Histograms of Numerical Features", fontsize=20)
        plt.tight_layout()
        plt.show()

    def draw_kde_plot(self, df,  features_to_plot=[], number_of_top_features=0, hue='Group', palette_selection = "coolwarm", group_labels = {}):

        if len(group_labels)>0:
            df[hue] = df[hue].map(group_labels)

        # If specified, select the top N features to plot
        if number_of_top_features > 0:
            features_to_plot = self.df.select_dtypes(include=['number']).columns[:number_of_top_features]

        # Exit if no features to plot
        if len(features_to_plot) == 0:
            return

        # Set up the figure for KDE plots
        plt.figure(figsize=(16, 12))
        
        # Loop through features to plot KDE for each feature with a distinct color palette
        for i, feature in enumerate(features_to_plot):  # Plot the first 5 numerical features
            plt.subplot(3, 2, i + 1)
            
            # Using the 'Spectral' color palette for distinct group colors
            #palette = sns.color_palette('husl', n_colors=len(features_to_plot))  # Create a palette with 3 colors
            palette_selection=palette_selection # "coolwarm", "Spectral", "viridis", "plasma", "cubehelix","Set1","Accent"
            sns.kdeplot(data=df, 
                        x=feature, 
                        hue=hue, 
                        fill=True, 
                        #color=palette[i],
                        common_norm=True,  # Ensure total area under curve is 1 for each group
                        palette=palette_selection
                        )
            plt.title(f'{feature} Distribution by {hue}')
        
        plt.tight_layout()
        plt.show()


    def draw_scatter_plot(self, df, features_to_plot= [], number_of_top_features = 0, hue= None):

        if number_of_top_features > 0:
            features_to_plot = df.select_dtypes(include=['number']).columns[0:number_of_top_features]

        if len(features_to_plot) == 0:
            return

        sns.pairplot(df[features_to_plot], vars=features_to_plot, hue = hue, palette='Set2', diag_kind='kde')
        plt.suptitle("Pair Plot of Selected Features", fontsize=16)
        plt.show()

    def draw_heatmap_with_matrix(self, matrix, title = "", x_label= "", y_label = "", font_size = 12,
                                 cmap_color = "Blues", ticklabels = []):
        
        # Generate confusion matrix and plot
        ax = sns.heatmap(matrix, annot=True, fmt="d", cmap=cmap_color, 
                    xticklabels=ticklabels, yticklabels=ticklabels,
                    annot_kws={"size": font_size})  # Set annotation font size
        
        # Customize font sizes and add labels
        plt.title(title, fontsize=font_size)
        plt.xlabel(x_label, fontsize=font_size)
        plt.ylabel(y_label, fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)

        # Customize the legend font size
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=font_size)  # Set the font size of the legend
        plt.show()


    def plot_confusion_matrix(self, cm, ticklabels, show_percentage=False, font_size = 14, cmap_color = "Blues", task_name = "", title = ""):
        """
        Plot the confusion matrix.

        Parameters:
        - cm (np.ndarray): Confusion matrix.
        - labels (list): List of unique class labels.
        - show_percentage (bool): Whether to show percentages instead of raw numbers.
        """



        sns.set(font_scale=1.4)  # Scale fonts in the plot

        if show_percentage:
            cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            annot = np.array([[f"{value:.2f}%" for value in row] for row in cm_percentage])
            fmt = '.2f'
        else:
            annot = cm
            fmt = 'd'


        if len(title) == 0:
            if len(task_name)>0:
                title = f"{task_name} Confusion Matrix"
            else:
                title = f"Confusion Matrix"
        self.draw_heatmap_with_matrix(cm, title = title, ticklabels = ticklabels, font_size = font_size,
                                 cmap_color = cmap_color)
        # # Plot confusion matrix
        # plt.figure(figsize=(6, 4))
        # sns.heatmap(cm, annot=annot, fmt=fmt, cmap='Blues', xticklabels=labels, yticklabels=labels)

        # # Customize title and labels
        # plt.title('Confusion Matrix', fontsize=14)
        # plt.xlabel('Predicted', fontsize=14)
        # plt.ylabel('Actual', fontsize=14)
        # plt.xticks(fontsize=14)
        # plt.yticks(fontsize=14)
        # plt.show()

    def draw_heatmap(self, df, features_to_plot= [], number_of_top_features = 0):


        if number_of_top_features > 0:
            features_to_plot = df.select_dtypes(include=['number']).columns[0:number_of_top_features]

        if len(features_to_plot) == 0:
            return
        
        # Correlation matrix for numerical features
        # A correlation matrix shows the pairwise correlation between features
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[features_to_plot].corr()

        # Heatmap of the correlation matrix
        # Visualizes how closely related the features are to each other
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
        plt.title("Correlation Matrix of Features")
        plt.show()
        

    def draw_box_plot(self, df, features_to_plot= [], number_of_top_features = 0):
        
        if number_of_top_features > 0:
            features_to_plot = df.select_dtypes(include=['number']).columns[0:number_of_top_features]

        if len(features_to_plot) == 0:
            return

        plt.figure(figsize=(16, 12))
        for i, feature in enumerate(features_to_plot):
            plt.subplot(5, 4, i + 1)  # Use a 5x4 grid, allowing up to 20 plots
            sns.boxplot(y=df[feature])
            plt.title(f'Box Plot - {feature}')
            
        plt.tight_layout()
        plt.show()

    def draw_violin(self, df, features_to_plot, to_remove_outlier= False, outlier_multiplier=3, do_within_group_t_test=True, group_names = [], verbose= False):
        
        data_preprocessor = psychai.data_preparation.preprocessing.DataPreprocessor()

        font_size = 12  # Set font size for all axis labels

        # Create subplots for each variable in columns
        fig, axes = plt.subplots(len(features_to_plot), 1, figsize=(6, 6 * len(features_to_plot)))

        # Ensure axes is iterable even if there is only one subplot
        if len(features_to_plot) == 1:
            axes = [axes]

        # Initialize a list to store F-test results
        f_test_results = []

        for i, var in enumerate(features_to_plot):
            # Initialize list to store clean plot data
            plot_data_list = []

            # Remove rows with NaN values for the variable
            df_step1_cleaned = df.dropna(subset=[var])

            if to_remove_outlier:
                df_group_wt_outlier = data_preprocessor.remove_outliers(df_step1_cleaned, var, outlier_multiplier)
                plot_data_list.append(df_group_wt_outlier)
            else:
                plot_data_list.append(df_step1_cleaned)

            # Concatenate DataFrame for all groups
            plot_data = pd.concat(plot_data_list, ignore_index=True)

            # Replace group numbers with group names
            plot_data['Group'] = plot_data['Group'].map(group_names)

            # Specify the order for the groups based on group_names
            group_order = list(group_names.values())

            # Create violin plot to show variable by group
            sns.violinplot(ax=axes[i], data=plot_data, x='Group', y=var, palette='Set2', order=group_order)

            # Add scatter plot to show individual data points
            sns.stripplot(ax=axes[i], data=plot_data, x='Group', y=var, color='white', alpha=0.6, jitter=False, order=group_order)

            # Perform within-group t-test for each group and annotate only p-value
            if do_within_group_t_test:
                for group_num, group_name in group_names.items():
                    data = plot_data[plot_data['Group'] == group_name][var]

                    # Perform one-sample t-test (against zero)
                    t_stat, p_value = ttest_1samp(data, 0)

                    if verbose:
                        # Print the t-value and p-value for within-group comparison
                        print(f'Within-group t-test for {group_name} on {var}: t={t_stat:.3f}, p={p_value:.3f}')

                    # Format the p-value for annotation (t-value is omitted)
                    p_value_text = f'p={p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'

                    # Annotate only the p-value on the plot
                    y_min = data.min()
                    x_pos = list(group_names.values()).index(group_name)  # x-coordinate for annotation
                    y_pos = y_min * 1.1  # y-coordinate slightly below the minimum value

                    # Annotate p-value
                    axes[i].text(x_pos, y_pos, p_value_text, horizontalalignment='center', verticalalignment='top', 
                                fontsize=font_size, color='black')

            # Perform between-group independent t-tests and annotate only p-value
            group_combinations = list(combinations(group_names.values(), 2))
            y_max = plot_data[var].max()
            y_min = plot_data[var].min()

            for j, (group1, group2) in enumerate(group_combinations):
                data1 = plot_data[plot_data['Group'] == group1][var]
                data2 = plot_data[plot_data['Group'] == group2][var]

                # Perform independent two-sample t-test (Welch's t-test)
                t_stat, p_value = ttest_ind(data1, data2, equal_var=False)

                # Print the t-value and p-value for between-group comparison
                if verbose:
                    print(f'Between-group t-test for {group1} vs {group2} on {var}: t={t_stat:.3f}, p={p_value:.3f}')

                # Format the p-value for annotation (t-value is omitted)
                p_value_text = f'p={p_value:.3f}' if p_value >= 0.001 else 'p < 0.001'

                # Positioning line and p-value annotation between groups
                x1, x2 = list(group_names.values()).index(group1), list(group_names.values()).index(group2)
                y = y_max + 0.15 * (j + 1) * (y_max - y_min)  # Vertically stack lines

                # Draw comparison line
                axes[i].plot([x1, x2], [y, y], color='black', lw=1.5)

                # Annotate only p-value above the line
                axes[i].text((x1 + x2) * 0.5, y, p_value_text, horizontalalignment='center', verticalalignment='bottom', 
                            fontsize=font_size, color='black')

            # Calculate and format F-test results
            data_groups = [plot_data[plot_data['Group'] == group][var] for group in group_names.values()]
            f_stat, f_p_value = f_oneway(*data_groups)

            # Format F-test result
            f_p_value_text = f'p={f_p_value:.3f}' if f_p_value >= 0.001 else 'p < 0.001'
            f_test_text = f'F({len(group_names) - 1}, {len(plot_data) - len(group_names)}) = {f_stat:.3f}, {f_p_value_text}'

            # Store F-test result for later display
            f_test_results.append(f'{var}: {f_test_text}')

            # Set axis labels and font size
            axes[i].set_xlabel('Group', fontsize=font_size)
            axes[i].set_ylabel(var, fontsize=font_size)

            # Set tick label font size
            axes[i].tick_params(axis='both', which='major', labelsize=font_size)

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.show()

        # After the plot, print F-test results in the console
        for result in f_test_results:
            print(result)