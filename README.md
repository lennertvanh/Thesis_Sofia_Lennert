# **Thesis_Sofia_Lennert**

## Introduction

This repository contains the code used for a Master Thesis about "Multi-task tree-ensemble model chains for predicting the progression of Multiple Sclerosis" (academic year 2023-2024). This Thesis is in collaboration with the Master of Statistics and Data Science at KU Leuven and all analysis is conducted by Sofia Mendes and Lennert Vanhaeren, under the supervision and guidance of Prof. Dr. Celine Vens (promotor), and Dr. Felipe Kenji Nakano and Robbe D'hondt (daily supervisors).


## Data

For our research, we are utilizing the Multiple Sclerosis Outcome Assessments Consortium (MSOAC) Placebo Database, provided by the Critical Path Institute. This database compiles placebo arm data from clinical trial datasets, contributed by industry members of MSOAC. It includes 2465 individual patient records from 9 clinical trials and contains 12 CSV files with information on demographics (such as age, gender, country, and race), confirmed and unconfirmed relapses, medical history (including MS diagnosis), functional tests (such as T25FW, NHPT, PASAT, and SDMT), visual acuity, concomitant medications, pregnancy tests, subject’s dominant hand and questionnaires (including EDSS, KFSS, RAND-36, SF-12, and BDI-II). 

Access to the MSOAC Placebo data is granted to qualified researchers upon submission and approval of a request via this link: https://codr.cpath.org/main/acceptTerms.html.


## Installation

If you want to run the notebooks locally, we suggest you to create a separate virtual environment running Python 3.9, and install all of the required dependencies there. Run in Terminal/Command Prompt:

```
git clone https://github.com/lennertvanh/Thesis_Sofia_Lennert.git
cd Thesis_Sofia_Lennert
python3 -m venv thesisenv
```
In Windows: 

```
thesisenv\Scripts\activate
```

To install all of the required `pip` packages to this environment, simply run:

```
pip install -r requirements.txt
```

Now, you're all set to locally run the notebooks (given you have access to the data).


## Repository
A brief explanation of the structure and files in our repository is provided below:
* assets: folder with some images from demographics and feature importance heatmaps
* modeling: folder with all the notebooks of our analysis.
    * chaining.py: file that contains the code to run the model chain, calculate feature importance and output probability outcomes to compute the Brier score
* descriptives_data.ipynb: file with descriptive statistics from the input data, and the variables used as targets
* get_data.py: file with the pipeline routine to generate the unified static dataframe for the modelling from the MSOAC csv files 
* requirements.txt: file containing the Python packages to run the code in this repository

## Authors
* Ana Sofia Mendes - sofiansmendes
* Lennert Vanhaeren - lennertvanh
