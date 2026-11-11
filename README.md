# Myocardial Infarction Complications Prediction Project 4AL3

Project Overview

This project explores the prediction of myocardial infarction complications using a multi-task MLP. It uses the publicly available Myocardial Infarction Complications dataset from the UCI Machine Learning Repository. This can be accessed here: https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications. 

The objective is to predict 11 binary outcomes and one 8-class categorical outcome (LET_IS) using information available within the first 48 hours of hospitalization.

We are currently investigating different class-weighting strategies to address the imbalance in the target variables but moving forwards will try different archetectures and perfrom hyperpareametr tuning. 


## Repository Structure

| File or Directory | Description |
|---|---|
| `data_preprocessing_actual.ipynb` | Main preprocessing notebook. Fetches the UCI dataset, performs feature selection, performs data splits, does imputations, does feature scaling, and prepares the processed datasets. Also contains the majority-class baseline model implementation. |
| `data_preprocessing_with_analysis.ipynb` | Initial data analysis and preprocessing notebook - can go through to explain all preprocessing decisions and step by step process for preprocessing. Contains correlation analysis, missing value analysis etc. |
| `model_implementation.ipynb` | Implements the multi-task MLP using full inverse-frequency class weighting. |
| `second_model_implementation.ipynb` | Implements the multi-task MLP using square-root inverse-frequency weighting (exponent of 0.5). |
| `third_model_implementation.ipynb` | Implements the multi-task MLP using intermediate inverse-frequency weighting (exponent of 0.7). |
| `baseline-model_comparison.ipynb` | Loads the exported evaluation CSV files and compares the majority-class baseline against the three weighting strategies. |
| `baseline.csv` | Evaluation results for the majority-class baseline. |
| `full_weighting.csv` | Evaluation results for the full inverse-frequency weighting model. |
| `square_root_weighting.csv` | Evaluation results for the square-root weighting model. |
| `intermediate_weighting.csv` | Evaluation results for the intermediate weighting model. |
| `preprocessed_data/` | Directory containing the six preprocessed files used by the model notebooks. Can use to look at what preprocessed data looks like. |
 | `Proposal LaTeX/` | Contains the LaTeX source files, bibliography and style files, images of graphs, and compiled project report. CONTAINS THE FINAL PROJECT PROGRESS PDF REPORT. |

## Dependencies

The project uses Python and the following libraries:

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation and CSV handling. |
| `numpy` | Numerical operations and array manipulation. |
| `matplotlib` | Visualizing data and model training results. |
| `ucimlrepo` | Fetching the dataset from the UCI Machine Learning Repository. |
| `iterative-stratification` | Performing multilabel iterative stratification for the train, validation, and test splits. |
| `scikit-learn` | Imputation, feature scaling, class weighting, baseline classification, and evaluation metrics. |
| `tensorflow` | Building, training, and evaluating the multi-task MLP models. |
| `jupyter` | Running the project notebooks. |

### Dependencies Installation

pip install pandas numpy matplotlib ucimlrepo iterative-stratification scikit-learn tensorflow jupyter

## Notes

Please run the data_preprocessing_actual.ipynb file before any of the model files. Please look at the model implementation notebook outputs for graphs and related analysis. 

