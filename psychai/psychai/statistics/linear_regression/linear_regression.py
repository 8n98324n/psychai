import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import psychai.statistics.data_preparation.outlier as outlier

def explore_simple_linear_regression_all(valid_data, dependent_variables, independent_variables, outlier_threshold, apply_transform, plot_all, plot_significant, display_summary, significance_level):
        """
        线性回归分析函数，对目标变量和预测变量进行简单线性回归，支持对数据进行转换、绘图和结果输出。
        
        参数:
        valid_data: 有效数据集
        dependent_variables: 目标变量列表
        independent_variables: 预测变量列表
        outlier_threshold: 去除异常值的阈值
        apply_transform: 是否尝试对数据进行转换
        plot_all: 是否绘制所有回归结果的图
        plot_significant: 是否只绘制显著结果的图
        display_summary: 是否打印所有回归结果的摘要
        significance_level: 显著性水平
        """
        regression_results = []  # 存储回归结果的列表
        font_size = 14  # 图表字体大小
        
        for dependent_variable in dependent_variables:
            for independent_variable in independent_variables:
                # 去除异常值
                cleaned_data = outlier.remove_outliers(valid_data, independent_variable, outlier_threshold)
                cleaned_data = outlier.remove_outliers(cleaned_data, dependent_variable, outlier_threshold)

                # 数据归一化函数
                def normalize_data(raw_data):
                    min_val = raw_data.min()
                    max_val = raw_data.max()
                    return 0.01 + (raw_data - min_val) / (max_val - min_val)

                # 如果需要，应用数据转换
                if apply_transform:
                    cleaned_data['log_' + dependent_variable] = np.log(normalize_data(cleaned_data[dependent_variable]))
                    cleaned_data['log_' + independent_variable] = np.log(normalize_data(cleaned_data[independent_variable]))
                    cleaned_data['sqrt_' + dependent_variable] = np.sqrt(normalize_data(cleaned_data[dependent_variable]))
                    cleaned_data['sqrt_' + independent_variable] = np.sqrt(normalize_data(cleaned_data[independent_variable]))

                    # 需要测试的转换组合
                    transformations = [(dependent_variable, independent_variable), 
                                    ('log_' + dependent_variable, independent_variable), 
                                    ('log_' + dependent_variable, 'log_' + independent_variable), 
                                    ('sqrt_' + dependent_variable, independent_variable),
                                    ('sqrt_' + dependent_variable, 'sqrt_' + independent_variable)]

                    # 对每个转换进行回归分析
                    for transformed_target, transformed_predictor in transformations:
                        formula = f"{transformed_target} ~ {transformed_predictor}"
                        
                        transformed_data = outlier.remove_outliers(cleaned_data, transformed_predictor, outlier_threshold)
                        transformed_data = outlier.remove_outliers(transformed_data, transformed_target, outlier_threshold)

                        # 执行线性回归
                        model = smf.ols(formula, data=transformed_data).fit()
                        slope = model.params[transformed_predictor]
                        intercept = model.params.Intercept
                        r_squared = model.rsquared

                        # 计算预测值并基于预测值计算R²
                        predicted_values = model.fittedvalues
                        actual_values = transformed_data[transformed_target]
                        prediction_r_squared = 1 - (np.sum((actual_values - predicted_values) ** 2) / np.sum((actual_values - np.mean(actual_values)) ** 2))

                        # 存储回归结果，包括预测值的R²
                        regression_results.append([dependent_variable, independent_variable, transformed_target, transformed_predictor, formula, len(transformed_data), model.f_pvalue, slope, intercept, r_squared, prediction_r_squared])
                    
                        # # 存储回归结果
                        # regression_results.append([dependent_variable, independent_variable, transformed_target, transformed_predictor, formula, len(transformed_data), model.f_pvalue, slope, intercept, r_squared])

                        # 如果满足绘图条件，绘制回归图
                        if plot_all or (model.f_pvalue < significance_level and plot_significant):
                            plt.scatter(transformed_data[transformed_predictor], transformed_data[transformed_target])
                            plt.plot(transformed_data[transformed_predictor], model.fittedvalues, color='red')
                            plt.xlabel(transformed_predictor, fontsize=font_size)
                            plt.ylabel(transformed_target, fontsize=font_size)
                            plt.title(f'{formula}, R² = {r_squared:.4f}, p-value: {model.f_pvalue:.4f}', fontsize=font_size)
                            plt.show()

                # 进行未转换的数据回归分析
                else:
                    formula = f"{dependent_variable} ~ {independent_variable}"
                    model = smf.ols(formula, data=cleaned_data).fit()
                    slope = model.params[independent_variable]
                    intercept = model.params.Intercept
                    r_squared = model.rsquared

                    # 计算预测值并基于预测值计算R²
                    predicted_values = model.fittedvalues
                    actual_values = transformed_data[transformed_target]
                    prediction_r_squared = 1 - (np.sum((actual_values - predicted_values) ** 2) / np.sum((actual_values - np.mean(actual_values)) ** 2))

                    # 存储回归结果，包括预测值的R²
                    regression_results.append([dependent_variable, independent_variable, transformed_target, transformed_predictor, formula, len(transformed_data), model.f_pvalue, slope, intercept, r_squared, prediction_r_squared])
                    
                    # 存储回归结果
                    # regression_results.append([dependent_variable, independent_variable, dependent_variable, independent_variable, formula, len(cleaned_data), model.f_pvalue, slope, intercept, r_squared])

                    # 如果满足显著性要求，绘制回归图
                    if model.f_pvalue < significance_level and plot_significant:
                        plt.scatter(cleaned_data[independent_variable], cleaned_data[dependent_variable])
                        plt.plot(cleaned_data[independent_variable], model.fittedvalues, color='red')
                        plt.xlabel(independent_variable, fontsize=font_size)
                        plt.ylabel(dependent_variable, fontsize=font_size)
                        plt.title(f'{formula}, R² = {r_squared:.4f}, p-value: {model.f_pvalue:.4f}', fontsize=font_size)
                        plt.show()

        # 将回归结果转换为DataFrame
        result_df = pd.DataFrame(regression_results, columns=['目标变量', '预测变量', '转换后目标变量', '转换后预测变量', '回归公式', '数据长度', 'p值', '斜率B1', '截距B0', 'R²','Model-Evaluation R-Squared'])

        # 为每个目标变量和预测变量组合选出最大的R²
        idx = result_df.groupby(['目标变量', '预测变量'])['R²'].idxmax()
        largest_r_squared_df = result_df.loc[idx].reset_index(drop=True)

        # 如果需要，打印结果摘要
        if display_summary:
            print("所有回归结果:")
            print(result_df)
            print("\n每个目标和预测变量组合的最大R²:")
            print(largest_r_squared_df)

        return result_df, largest_r_squared_df

def simple_linear_regression(df, dep_var, indep_var):
    formula = f"{dep_var} ~ {indep_var}"
    model = smf.ols(formula, data=df).fit()
    return model

def explore_simple_linear_regression(data_valid, target_variables, other_variables, K, TryTransformation, PlotSigfinicant, PrintSummary, SignificantLevel):
    all_p_values = []

    for dep_var in target_variables:
        for indep_var in other_variables:
            # Removing outliers
            data = outlier.remove_outliers(data_valid, indep_var, K)
            data = outlier.remove_outliers(data, dep_var, K)

            # Normalizing data
            def normalize_data(raw_data):
                min_val = raw_data.min()
                max_val = raw_data.max()
                return 0.01 + (raw_data - min_val) / (max_val - min_val)

            # Apply transformations if required
            if TryTransformation:
                data['log_' + dep_var] = np.log(normalize_data(data[dep_var]))
                data['log_' + indep_var] = np.log(normalize_data(data[indep_var]))
                data['sqrt_' + dep_var] = np.sqrt(normalize_data(data[dep_var]))
                data['sqrt_' + indep_var] = np.sqrt(normalize_data(data[indep_var]))

                # List of transformations to test
                transformations = [(dep_var, indep_var), 
                                   ('log_' + dep_var, indep_var), 
                                   ('log_' + dep_var, 'log_' + indep_var), 
                                   ('sqrt_' + dep_var, indep_var),
                                   ('sqrt_' + dep_var, 'sqrt_' + indep_var)]

                # Apply transformations and perform regression
                for dep, indep in transformations:
                    formula = f"{dep} ~ {indep}"
                    model = smf.ols(formula, data=data).fit()
                    # Add results to all_p_values, plot and print if necessary

            # Regular regression
            else:
                formula = f"{dep_var} ~ {indep_var}"
                model = smf.ols(formula, data=data).fit()
                # Add results to all_p_values, plot and print if necessary

            # Plotting if significant
            if model.f_pvalue < SignificantLevel and PlotSigfinicant:
                plt.scatter(data[indep_var], data[dep_var])
                plt.plot(data[indep_var], model.fittedvalues, color='red')
                plt.xlabel(indep_var)
                plt.ylabel(dep_var)
                plt.title(f'p-value: {model.f_pvalue:.4f}')
                plt.show()

            # Printing summary if required
            if PrintSummary:
                print(formula)
                print(model.summary())

            # Add results to all_p_values
            all_p_values.append([1,2])

    return pd.DataFrame(all_p_values, columns=['Formula', 'Length', 'P-Value', 'B1', 'B0', 'Correlation'])

