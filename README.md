# Predicting Whether a Bill will become Law
---
## Contents:

- [Problem Statement](#Problem-Statement)
- [Software Requirements](#Software-Requirements)
- [Data Sources](#Data-Sources)
- [Data Cleaning](#Data-Cleaning)
- [Data Dictionary](#Data-Dictionary)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
- [Data preprocessing](#Data-Preprocessing)
- [Model Building](#Model-Building)
- [Conclusions and Recommendations](#Conclusions-and-Recommendations)
---
## Problem Statement

The goal of this study is to develop machine learning classification models predicting whether a state bill will pass or not using Natural Language Processing of the bill's title. Performance of this model will be guided by balanced accuracy, and should improve on the baseline accuracy by 10%. 

---
## Software Requirements
- Pandas
- Scikit-learn
- numpy
- matplotlib.pyplot
- seaborn
- plotly

---
## Data Sources
The Data for this project came from severall sources and then was merged together. 

The Legislative composition data for years 2017-2023 comes from the National Conferance of State Legislatures [website](https://www.ncsl.org/about-state-legislatures/state-partisan-composition). The data has information on the party composition of state legislators for each state. It also has the party for each governor. This data needed some preprocessing to convert it into the wanted format of csv files as it is stored in pdf files. 

The Bills data is from the Open States [website](https://openstates.org/data/session-csv/). Open States is an organization that aggregates, standardizes, and cleans legislative data for all 50 states. The data used for this study is from the bulk data they offer of proposed bills in the state's legislature. The data is stored in zip files for each legislative session by state. Open States scrapes their data directly from government websites and seems to be quite reliable. The data on the legislatures party also comes from Open Sates, but from their python [API](https://openstates.github.io/pyopenstates/). A limitation off the API is that there is a limit of requests that can be made per day. Because of this, a few days were needed to collect all of the Legislature data. To use the API you need to create an account and then get an API key from Open States.

The Legislative Bills Progression data comes from the Harvard Dataverse and was published by Dr. Garlick in April, 2023 ([link to data](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8PTHXT)). The dataset contains over a million bills for all states from the years of 2011 to 2019. It includes the 23 step progression of each bill showing where it failed or if it was enacted into law. Dr. Garlick and his team used Open Sates to create this data set.

After merging all of the datasets together, we ended up with two datasets; one with almost 160,000 bills and the other with around 85,000 bills. The smaller dataset is a subset of the larger one but has more information with it. It includes a column of which party sponsored the bill.

---
## Data Cleaning

First, many redundant columns were removed. They were removed because either they had too many missing values, or because they were irrelevant to the study. Secondly, some columns were cleaned up a bit. For example, the subject column had a list of subjects that the bill relates to. It would be more useful if it was in just a normal string format rather than a list so word count vectorizer could be used on it in the modeling phase. Lastly, we will have to get rid of a big chunk of our data because we are only interested in types of Legislation that can infact become laws. The column statute helped with this phase.

Before cleaning, the bigger dataset with one less column had 160,000 bills with 63 columns. After cleaning it has around 110,000 bills with 19 columns. The smaller dataset had around 85,000 bills and ended with 60,000 bills.

---
## Data Dictionary

|Feature|Type|Original Source|Description|
|---|---|---|---|
|**id**|*object*|Open States|Bill ID used on Open States website|
|**classification**|*object*|Open States|Type of legislation. e.g. "bill" or "resolution"|
|**title**|*object*|Open States|Title of bill|
|**subject**|*object*|Open States|Main subjects bill refers to.|
|**abstract**|*object*|Open States|Abstract of bill.|
|**state**|*object*|Open States|State the bill was introduced in.|
|**organization_classification**|*object*|Open States|Which house the bill was introduced in.|
|**year**|*int*|Garlick|Year the bill was introduced.|
|**bill_pre**|*object*|Garlick|Prefix of the bill.|
|**statute**|*int*|Garlick|Whether the bill is a statute or not.|
|**lpcode**|*float*|Garlick|Code that correlates to where the bill failed.|
|**pass_1st_chamber**|*int*|Garlick|Passed the first chamber or not.|
|**pass_2nd_chamber**|*int*|Garlick|Passed the second chamber or not.|
|**law_enacted**|*int*|Garlick|Enacted as law.|
|**senate_dem**|*int*|NCSL|Number of democratic state senators.|
|**senate_rep**|*int*|NCSL|Number of republican state senators.|
|**house_dem**|*int*|NCSL|Number of democrats in the house.|
|**house_rep**|*int*|NCSL|Number of republicans in the house.|
|**gov_party**|*object*|NCSL|Party of the state governor.|
|**senate_party**|*object*|NCSL|Party dominant in the Senate|
|**house_party**|*object*|NCSL|Party dominant in the House|
|**state_party_control**|*object*|NCSL|Republican if House, Senate and Governor are republican. Democrat if it is the opposite. Split if there are a mix of parties.|
|**majority_sponsor_party**|*object*|Open States API|Party that introduced the bill. (only in our smaller dataset)|

Links to original datasets:
   - Open States: [website](https://openstates.org/data/session-csv/), [API](https://openstates.github.io/pyopenstates/)
   - [Garlick](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8PTHXT)
   - [NCLS](https://www.ncsl.org/about-state-legislatures/state-partisan-composition)

---
## Exploratory Data Analysis

Before modeling, we investigated our dataset to see how dfferent features related with heart disease. Below are the most interesting graphs from that process.

<img src = "Code/03_Exploratory_Data_Analysis/Imbalanced.png">
Our target variable, the bill passing or not, is very imbalanced. This will make it more difficult for our models to predict which bills will become law. Oversampling techniques may be able to help the models performance. 

<img src = "Code/03_Exploratory_Data_Analysis/chamber_to_chamber_pass_rate.png">
31% of bills pass the first chamber, 24% pass the second, and 20% end up becoming law.


<img src = "Code/03_Exploratory_Data_Analysis/given_passed_prior_chamber_pass_rate.png">
The hardest chamber for a law to pass is the chamber it was introduced in. Only 31% of the laws pass the first chamber. After passing the first chamber though, laws have an easier time passing the next two steps. 75% of the laws that makee it to the second chamber pass the second chamber and 84% of the laws that make it passed the second chamber get enacted into law.

<img src = "Code/03_Exploratory_Data_Analysis/state_party_control_2017.png">
Party control of each state. Those in red are states with a Republican Governor, House, and Senate. Those in Blue are states with a Democratic Governor, House and Senate. Everything in between those two colors have a mix of democrats and republicans. Notice that most states were republican in 2017.


Nebraska is missing from this map because Nebraska has a univariate congress and thus was left out.

<img src = "Code/03_Exploratory_Data_Analysis/state_party_control_2018.png">
Same as graph above but for 2018. Hard to spot differences from the 2017 graph but it seems like New York and West Virginia became more republican while New Jersey became mored democrat. Very similar to 2017 though.

<img src = "Code/03_Exploratory_Data_Analysis/pass_rate_for_state_control_by_who_introduced.png">
This graph shows the percentage of bills passing for Republican, Democrat, and Split states for republican and democratic bills. As we can see, it seems that republican states are more partisan. 32% of republican bills pass while only 9% of democrat bills pass in republican states. Split states are fairly equal in the rates they pass bills for each party. Democrat states are fairly partisan with 28% of their own bills passing while 21% of republican bills pass.


---
## Model Building

## Conclusions and Recommendations



