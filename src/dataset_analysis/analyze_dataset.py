import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def interpret_corelation(corelation: float) -> str:
    abs_corelation: float = abs(corelation)
    if abs_corelation >= 0.8:
        return "Very strong"
    if abs_corelation >= 0.6:
        return "Strong"
    if abs_corelation >= 0.4:
        return "Moderate"
    if abs_corelation >= 0.2:
        return "Weak"
    return "Very weak"


def _get_columns_description(dataset: pd.DataFrame) -> str:
    """
    Returns a description for all columns from dataset in a form of:

        <column 1 name>      <column 1 type>
        <column 2 name>      <column 2 type>
        ...
        <column n name>      <column n type>
        
    Parameters
    - dataset (pd.Dataframe): Describes dataset.

    Returns
    str: Description
    """
    # Plot nulls per column
    ax = dataset.isnull().sum().div(dataset.shape[0]).mul(100).round(1).plot.barh(width=0.8, figsize=(11, 20))
    ax.bar_label(ax.containers[0], label_type='edge')
    plt.title("Percentage of empty values in columns")
    plt.xlabel("% of empty values")
    plt.savefig("datasets/plots/base/null-values.png", bbox_inches="tight")
    plt.clf()

    # Generate textual report
    columns_data: str = "\n".join(
        f"{column:<20} {nulls:>8} ({nulls / dataset[column].shape[0] * 100:>5.2f} %)  {type_ if type_ != "object" else "category"}" 
        for column, type_, nulls in zip(dataset.columns, dataset.dtypes, dataset.isnull().sum())
    )
    return (
        f"Column name          Null entries        Column type\n"
        f"{columns_data}"
    )


def _get_column_details(dataset: pd.DataFrame, column_name: str) -> str:
    if pd.api.types.is_bool_dtype(dataset[column_name]):
        non_linear_corelation: float = dataset.corrwith(dataset["SalePrice"], numeric_only=True, method="spearman")[column_name]
        return (
            f"{column_name}\n"
            f"{'-' * len(column_name)}\n"
            f"Type                  {dataset[column_name].dtype}\n"
            f"Non linear Corelation {non_linear_corelation:12,.2f} ({interpret_corelation(non_linear_corelation)})\n"
        )
    if pd.api.types.is_numeric_dtype(dataset[column_name]):
        non_linear_corelation: float = dataset.corrwith(dataset["SalePrice"], numeric_only=True, method="spearman")[column_name]
        mean: float = dataset[column_name].mean()
        standard_deviation: float = dataset[column_name].std()
        relative_standard_deviation: float = standard_deviation / mean * 100

        return (
            f"{column_name}\n"
            f"{'-' * len(column_name)}\n"
            f"Type                  {dataset[column_name].dtype}\n"
            f"Min                   {dataset[column_name].min():12,.2f}\n"
            f"Mean                  {mean:12,.2f}\n"
            f"Max                   {dataset[column_name].max():12,.2f}\n"
            f"Std. dev              {standard_deviation:12,.2f} ({relative_standard_deviation:.2f} %)\n"
            f"Non linear Corelation {non_linear_corelation:12,.2f} ({interpret_corelation(non_linear_corelation)})\n"
        )
    else:
        values: str = '\n'.join(
            f'{name:<15} {count:>8} ({count / dataset[column_name].size * 100:.2f} %)' 
            for name, count in dataset[column_name].value_counts().to_dict().items()
        )
        return (
            f"{column_name}\n"
            f"{'-' * len(column_name)}\n"
            f"Type        {dataset[column_name].dtype if dataset[column_name].dtype != "object" else "category"}\n"
            f"{values}\n"
        )


def _get_types_overview(dataset: pd.DataFrame) -> str:
    return "\n".join(
        f"{row.replace("object  ", "category")} ({int(row.split()[-1]) / len(dataset.columns) * 100:.2f} %)" 
        for row in str(dataset.dtypes.value_counts()).split("\n")[:-1]
    )


def generate_target_histogram(dataset: pd.DataFrame, target: str = "SalePrice") -> int:
    iqr: float = dataset[target].quantile(0.75) - dataset[target].quantile(0.25)

    iqr_lower_bound_1_5: float = dataset[target].quantile(0.25) - 1.5 * iqr
    if iqr_lower_bound_1_5 < 0:
        iqr_lower_bound_1_5 = 0.0
    iqr_upper_bound_1_5: float = dataset[target].quantile(0.75) + 1.5 * iqr
    iqr_lower_bound_2: float = dataset[target].quantile(0.25) - 2.0 * iqr
    if iqr_lower_bound_2 < 0:
        iqr_lower_bound_2 = 0.0
    iqr_upper_bound_2: float = dataset[target].quantile(0.75) + 2.0 * iqr

    ax = dataset[target].plot.hist(edgecolor="black", figsize=(10, 10))
    ax.bar_label(ax.containers[0], label_type='edge')
    plt.legend()
    plt.title(f"Target histogram")
    plt.savefig(f"datasets/plots/base/target_histogram.png", bbox_inches="tight")

    ax.axvline(
        iqr_lower_bound_1_5, 
        color="red", 
        linestyle="--", 
        label=f"1.5 × IQR Lower bound ${iqr_lower_bound_1_5:,.2f}"
    )
    ax.axvline(
        iqr_upper_bound_1_5, 
        color="green", 
        linestyle="--", 
        label=f"1.5 × IQR Upper bound ${iqr_upper_bound_1_5:,.2f}"
    )
    ax.axvline(
        iqr_lower_bound_2, 
        color="blue", 
        linestyle="--", 
        label=f"2.0 × IQR Lower bound ${iqr_lower_bound_2:,.2f}"
    )
    ax.axvline(
        iqr_upper_bound_2, 
        color="orange", 
        linestyle="--", 
        label=f"2.0 × IQR Upper bound ${iqr_upper_bound_2:,.2f}"
    )

    plt.legend()
    plt.savefig(f"datasets/plots/base/target_histogram_with_outliers.png", bbox_inches="tight")
    plt.clf()
    print("\tGenerated histogram for target")


def generate_correlations_heatmap(dataset: pd.DataFrame) -> None:
    # Correlations heatmap
    _, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(
        dataset.corr(numeric_only=True, method="spearman").round(2),
        xticklabels=True, 
        annot=True, 
        annot_kws={"size": 8},
        yticklabels=True,
        vmax=1.0,
        vmin=-1.0,
        ax=ax
    )
    plt.title("Non-linear correlations for numerical features")
    plt.savefig("datasets/plots/base/correlations.png", bbox_inches="tight")
    plt.clf()


def analyze_dataset(dataset_path: str, output_path: str) -> None:
    TARGET: str = "SalePrice"

    # Generate data for report
    print(f"Analyzing dataset: {dataset_path}")
    dataset: pd.DataFrame = pd.read_csv(dataset_path)
    null_entries: int = dataset.isnull().sum().sum()

    # Types
    types: pd.Series = dataset.dtypes.astype(str).value_counts()
    types_renamed = types.rename(index={"object": "category"})
    types_renamed.plot.pie(autopct='%1.1f%%', ylabel='', title='Data types distribution')
    plt.savefig("datasets/plots/base/types.png", bbox_inches="tight")
    plt.clf()

    # Correlations
    generate_correlations_heatmap(dataset)

    # Target histogram
    generate_target_histogram(dataset)

    report_content = f"""\
DATASET SIZE
------------
Number of records: {dataset.shape[0]}
Number of columns: {dataset.shape[1]}

OVERVIEW
--------
{_get_types_overview(dataset)}
Null entries: {null_entries} ({null_entries / dataset.size * 100:.2f} %)

COLUMNS OVERVIEW
----------------
{_get_columns_description(dataset)}

TARGET
------
Name        {TARGET}
Type        {dataset[TARGET].dtype}
Min         ${dataset[TARGET].min():12,.2f}
Mean        ${dataset[TARGET].mean():12,.2f}
Max         ${dataset[TARGET].max():12,.2f}
Std. dev    ${dataset[TARGET].std():12,.2f}

{"\n".join(_get_column_details(dataset, column_name) for column_name in dataset.columns if column_name != TARGET)}
"""
    with open(output_path, "w") as report:
        report.write(report_content)
    print(f"\tAnalysis results saved to: {output_path}")


if __name__ == '__main__':
    analyze_dataset("datasets/ames-train.csv", "datasets/base-dataset-report.txt")
