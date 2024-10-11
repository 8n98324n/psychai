import numpy as np
import matplotlib.pyplot as plt


def BlandAltmanAnalysis(reference, target, title="Bland-Altman Analysis"):
    """
    Perform Bland-Altman Analysis and return results.

    Parameters:
    - reference: array-like, 1D numpy array or list
        Reference measurements.
    - target: array-like, 1D numpy array or list
        Target measurements.
    - title: str, optional
        Title for the Bland-Altman plot.

    Returns:
    - dict
        A dictionary containing Bland-Altman analysis results:
        - "mean_diff": Mean of the differences
        - "sd_diff": Standard deviation of the differences
        - "upper_limit": Upper limit of agreement
        - "lower_limit": Lower limit of agreement
        - "n": Number of data points
        - "within_limits_percentage": Percentage of data points within the limits
    """
    font_size = 12
    # Convert input data to numpy arrays
    reference = np.array(reference)
    target = np.array(target)

    # Calculate the differences
    diff = target - reference

    # Calculate Bland-Altman statistics
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)  # Use ddof=1 for sample standard deviation
    upper_limit = mean_diff + 1.96 * sd_diff
    lower_limit = mean_diff - 1.96 * sd_diff
    n = len(reference)

    # Calculate the percentage of data points within the limits
    within_limits_percentage = (np.sum((diff >= lower_limit) & (diff <= upper_limit)) / n) * 100

    # Create Bland-Altman plot
    plt.figure(figsize=(6, 4))
    plt.scatter((target + reference) / 2, diff, c='blue', marker='o', s=40)
    plt.axhline(y=mean_diff, color='gray', linestyle='--', label=f'Mean Diff: {mean_diff:.2f}')
    plt.axhline(y=upper_limit, color='red', linestyle='--', label=f'Upper Limit of Agreement: {upper_limit:.2f}')
    plt.axhline(y=lower_limit, color='green', linestyle='--', label=f'Lower Limit of Agreement: {lower_limit:.2f}')
    plt.xlabel('Mean of Reference and Target Measurements', fontsize=font_size)
    plt.ylabel('Difference (Target - Reference)', fontsize=font_size)
    #plt.title(title)
    plt.title(f'BAA (Within Limits %: {within_limits_percentage:.2f})', fontsize=font_size, color='black')
    plt.legend()
    plt.grid(True)

    # Display the percentage of data points within limits on the chart
    # plt.text(plt.xlim()[0] + 0.1, plt.ylim()[1]-0.1, f'Within Limits Percentage: {within_limits_percentage:.2f}%',
    #          fontsize=10, color='black')

    # Show the Bland-Altman plot
    plt.show()

    # Return results as a dictionary
    results = {
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "upper_limit": upper_limit,
        "lower_limit": lower_limit,
        "n": n,
        "within_limits_percentage": within_limits_percentage
    }

    return results
