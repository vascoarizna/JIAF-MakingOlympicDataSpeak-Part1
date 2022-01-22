# MAKING OLYMPIC DATA SPEAK: A Generic Exploration
<i>Did you know that the youngest Olympian ever was 10 years old when he participated, and even won a medal? (also being the youngest Olympic medalist). Or, did you know that Art Competition or Tug-of-War were Olympic sports?
Take a look into this Report, you won't regret.</i>


In this first Olympic Data Report, we will analyze a dataset that contains the information of all the athletes who competed from Athens 1896 to Rio 2016.

The idea will be to have a rough idea of the data we have, and we will make general questions, in order to try to get some interesting information. In the next releases, we will dig into sports' specific questions and insights.

In this case, the approach will be mainly looking at some data, avoiding showing code as much as possible. If you want to dig into the coding, the full report with all the python codes is available here: GitHub - VascoArizna

In the last report, Analyzing the Employees's Turnover, we paid attention to the three main aspects of Data Science: Descriptive, Predictive, and Prescriptive Analysis. In this case, we will only pay attention to the first one. We will try to ask us questions in order to find 'highsights'. Let's start de Exploration to see if we are able to find any curious data.


```python
#We import the main libraries.


import pandas as pd
import numpy as np
import math

#Libraries for plotting
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import cufflinks as cf
cf.go_offline()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import animation, rc
pd.options.mode.chained_assignment = None  # default='warn'

#This is to show the graphics and videos.
plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\ignac\AppData\Local\ffmpeg\bin\ffmpeg.exe'

#This is for the race bar-chart
import bar_chart_race as bcr

#We set the Warning Parameter
import warnings
warnings.filterwarnings('ignore')
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-2.8.3.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-2.8.3.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




```python
url='https://raw.githubusercontent.com/vascoarizna/JIAF-MakingOlympicDataSpeak-Part1/main/dataset/athlete_events.csv'

#path1='dataset/athlete_events.csv'
olympicData=pd.read_csv(url)
```

# 1.Exploratory Analysis

## 1.1.Peaking on the Data


```python
#Firstly we take a look on how we see the data.
olympicData.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>Team</th>
      <th>NOC</th>
      <th>Games</th>
      <th>Year</th>
      <th>Season</th>
      <th>City</th>
      <th>Sport</th>
      <th>Event</th>
      <th>Medal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>A Dijiang</td>
      <td>M</td>
      <td>24.0</td>
      <td>180.0</td>
      <td>80.0</td>
      <td>China</td>
      <td>CHN</td>
      <td>1992 Summer</td>
      <td>1992</td>
      <td>Summer</td>
      <td>Barcelona</td>
      <td>Basketball</td>
      <td>Basketball Men's Basketball</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>A Lamusi</td>
      <td>M</td>
      <td>23.0</td>
      <td>170.0</td>
      <td>60.0</td>
      <td>China</td>
      <td>CHN</td>
      <td>2012 Summer</td>
      <td>2012</td>
      <td>Summer</td>
      <td>London</td>
      <td>Judo</td>
      <td>Judo Men's Extra-Lightweight</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Gunnar Nielsen Aaby</td>
      <td>M</td>
      <td>24.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Denmark</td>
      <td>DEN</td>
      <td>1920 Summer</td>
      <td>1920</td>
      <td>Summer</td>
      <td>Antwerpen</td>
      <td>Football</td>
      <td>Football Men's Football</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



The data is composed in this structure. By paying attention to the columns, we can see how can we group the data, to make aggregated procedures. On the other hand, by paying attention to the type of value, what kind of operations we will have to do, or what type of Feature Engineering we will have to apply.

For example, we know that the last column, 'Medal', has three possible values:
- NaN (no medal won)
- Bronze (if the athlete won any bronze medal in that specific event in that specific sport in that Game)
- Silver (following the same logic described above)
- Gold (idem)

Also, in this first report, we will pay attention only to Summer Olympic Games. Having this said, we will filter by 'Season' features: 'Summer' will be the value we will choose, to filter the 'Winter' Olympic Games.

## 1.2.Shape of the DataFrame


```python
#Firstly we look for the shape.
olympicData.shape
```




    (271116, 15)



- We have 271116 rows and 15 columns
- This is fantastic. Between all the Games (Summer and Winter), we are talking about 271116 participants from 1896 to 2016.

## 1.3.Info of the DataFrame


```python
olympicData.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 271116 entries, 0 to 271115
    Data columns (total 15 columns):
     #   Column  Non-Null Count   Dtype  
    ---  ------  --------------   -----  
     0   ID      271116 non-null  int64  
     1   Name    271116 non-null  object 
     2   Sex     271116 non-null  object 
     3   Age     261642 non-null  float64
     4   Height  210945 non-null  float64
     5   Weight  208241 non-null  float64
     6   Team    271116 non-null  object 
     7   NOC     271116 non-null  object 
     8   Games   271116 non-null  object 
     9   Year    271116 non-null  int64  
     10  Season  271116 non-null  object 
     11  City    271116 non-null  object 
     12  Sport   271116 non-null  object 
     13  Event   271116 non-null  object 
     14  Medal   39783 non-null   object 
    dtypes: float64(3), int64(2), object(10)
    memory usage: 31.0+ MB
    

We see that the columns have their correspondent data type: ID, Age, height, weight and Year are numerical (integers and floats), and the rest are objects, which makes sense.

Also, we already identified some null values in Age, Height, Weight, and Medal. We will treat these values.

## 1.4.Statistical Description of the DataFrame


```python
olympicData.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>271116.000000</td>
      <td>261642.000000</td>
      <td>210945.000000</td>
      <td>208241.000000</td>
      <td>271116.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>68248.954396</td>
      <td>25.556898</td>
      <td>175.338970</td>
      <td>70.702393</td>
      <td>1978.378480</td>
    </tr>
    <tr>
      <th>std</th>
      <td>39022.286345</td>
      <td>6.393561</td>
      <td>10.518462</td>
      <td>14.348020</td>
      <td>29.877632</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>127.000000</td>
      <td>25.000000</td>
      <td>1896.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>34643.000000</td>
      <td>21.000000</td>
      <td>168.000000</td>
      <td>60.000000</td>
      <td>1960.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>68205.000000</td>
      <td>24.000000</td>
      <td>175.000000</td>
      <td>70.000000</td>
      <td>1988.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>102097.250000</td>
      <td>28.000000</td>
      <td>183.000000</td>
      <td>79.000000</td>
      <td>2002.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>135571.000000</td>
      <td>97.000000</td>
      <td>226.000000</td>
      <td>214.000000</td>
      <td>2016.000000</td>
    </tr>
  </tbody>
</table>
</div>



Ok. This is the first hindsight we find. The minimum registered Age in an Olympic Games is 10 years!. And the maximum is 97!

Regarding Age, we can see that the mean and the median (50%) are almost similar. This talks about a normal-like distribution. On top of that, despite we having a distance of 87 years between the minimum and maximum registered ages, we can appreciate that between the 25th and 75 percentiles (the 50 central percent of the data) we have only 7 years of distance.

Analyzing Height and Weight, from a general point of view, is almost useless, as we have to analyze them inside each sport, and in most cases, inside the respective events. 


# 2.Data Preparation/Cleaning

## 2.1.Missing Values

We are quickly checking for null values. It's understandable that some values from the 'old days' are not available. Nevertheless, if we would like to apply any Machine Learning model to predict any value, we would have to deal with the nulls by, either imputing a value (the mean for the numerical values, and the median for the categorical ones) or dropping the row.

The columns where we find nulls are these ones:


```python
# Check to see if there are any missing values in our data set
olympicData.isnull().any().sort_values(ascending=False).head(5)
```




    Age        True
    Height     True
    Weight     True
    Medal      True
    ID        False
    dtype: bool



The ratio of missing values is important for all the cases, except for Age:


```python
#checking for null values
nullSum=(olympicData).isnull().sum()
```


```python
ratio=nullSum/olympicData.shape[0]
ratio.sort_values(ascending=False).head(5)
```




    Medal     0.853262
    Weight    0.231912
    Height    0.221938
    Age       0.034944
    ID        0.000000
    dtype: float64



Missing Data Summary:

- Age: 3.5% of the data has the age missing
- Height: 22.1% of the data has the Height Missing
- Weigh: 23.1% of the values are missing
- Medal: 85.3% of the values are missing. This value, however, makes sense, as most of the athletes do not win a medal.


As we said, for ML purposes, we should impute values or drop the rows. Each column should be treated in a different way:

For the case of Age, Height, and Weight, the data imputation (mean, in this case) should be of the athlete's event average, rather than from the DataFrame's total average. Why? Because usually, for the same events (for example: in Boxing, the event/category up to 75kg) the athletes tend to have a similar body structure (weight/height).

## 2.3. Label Encoding
In the case of the medals, although there is no reference, we assume that all the NaNs (NULL values) correspond to athletes who never who any medal. That is why we should LabelEncode this category into:

- 0 for NaN
- 1 for Bronze
- 2 for Silver
- 3 for Gold.

This way we would, not only get rid of the NULL values but also will have our dataset already prepare for Fitting it into an ML Model.


```python
olympicData.groupby('Medal').ID.count()
```




    Medal
    Bronze    13295
    Gold      13372
    Silver    13116
    Name: ID, dtype: int64



### 2.2.1. Medal Imputation/Label Encoder


```python
#As we want to have a certain order, we will map manually

olympicData['MedalNumeric']=0
olympicData.loc[olympicData['Medal']=='Bronze','MedalNumeric']=1
olympicData.loc[olympicData['Medal']=='Silver','MedalNumeric']=2
olympicData.loc[olympicData['Medal']=='Gold','MedalNumeric']=3
```

### 2.2.1. Age, Height and Weight Imputation

We will do the imputation following the events


```python
ageMeanEvent=olympicData.groupby('Event').Age.mean()
heightMeanEvent=olympicData.groupby('Event').Height.mean()
weightMeanEvent=olympicData.groupby('Event').Weight.mean()

ageMeanEventDic=ageMeanEvent.to_dict()
heightMeanEventDic=heightMeanEvent.to_dict()
weightMeanEventDic=weightMeanEvent.to_dict()

ageNA=olympicData['Age'].isna()
heightNA=olympicData['Height'].isna()
weightNA=olympicData['Weight'].isna()

olympicData['Height1'] = olympicData.loc[heightNA,'Event'].map(heightMeanEventDic)
olympicData['Weight1'] = olympicData.loc[weightNA,'Event'].map(weightMeanEventDic)
olympicData['Age1'] = olympicData.loc[ageNA,'Event'].map(ageMeanEventDic)


heightNotNA=olympicData['Height1'].isna()
weightNotNA=olympicData['Weight1'].isna()
ageNotNA=olympicData['Age1'].isna()
```


```python
#I assign the non-null values that we held in Height 1 to the Null values we have in Heigh
olympicData.loc[heightNA,'Height']=olympicData.loc[~heightNotNA,'Height1']

#I assign the non-null values that we held in Height 1 to the Null values we have in Heigh
olympicData.loc[weightNA,'Weight']=olympicData.loc[~weightNotNA,'Weight1']

#I assign the non-null values that we held in Height 1 to the Null values we have in Heigh
olympicData.loc[ageNA,'Age']=olympicData.loc[~ageNotNA,'Age1']
```


```python
#checking for null values
print("Sum of NULL values in each column. ")
nullSum=(olympicData).isnull().sum()
nullSum.sort_values(ascending=False).head(5)
```

    Sum of NULL values in each column. 
    




    Age1       261784
    Medal      231333
    Height1    213028
    Weight1    212901
    Weight       4660
    dtype: int64




```python
ratio=nullSum/olympicData.shape[0]
ratio.sort_values(ascending=False).head(6)
```




    Age1       0.965579
    Medal      0.853262
    Height1    0.785745
    Weight1    0.785276
    Weight     0.017188
    Height     0.007683
    dtype: float64



Now we will do the imputation following the sports' average


```python
ageMeanSport=olympicData.groupby('Sport').Age.mean()
heightMeanSport=olympicData.groupby('Sport').Height.mean()
weightMeanSport=olympicData.groupby('Sport').Weight.mean()

ageMeanSportDic=ageMeanSport.to_dict()
heightMeanSportDic=heightMeanSport.to_dict()
weightMeanSportDic=weightMeanSport.to_dict()

ageNA=olympicData['Age'].isna()
heightNA=olympicData['Height'].isna()
weightNA=olympicData['Weight'].isna()

olympicData['Height1'] = olympicData.loc[heightNA,'Sport'].map(heightMeanSportDic)
olympicData['Weight1'] = olympicData.loc[weightNA,'Sport'].map(weightMeanSportDic)
olympicData['Age1'] = olympicData.loc[ageNA,'Sport'].map(ageMeanSportDic)


heightNotNA=olympicData['Height1'].isna()
weightNotNA=olympicData['Weight1'].isna()
ageNotNA=olympicData['Age1'].isna()
```


```python
#I assign the non-null values that we held in Height 1 to the Null values we have in Heigh
olympicData.loc[heightNA,'Height']=olympicData.loc[~heightNotNA,'Height1']

#I assign the non-null values that we held in Height 1 to the Null values we have in Heigh
olympicData.loc[weightNA,'Weight']=olympicData.loc[~weightNotNA,'Weight1']

#I assign the non-null values that we held in Height 1 to the Null values we have in Heigh
olympicData.loc[ageNA,'Age']=olympicData.loc[~ageNotNA,'Age1']
```


```python
#We delete these temporary columns
olympicData=olympicData.drop(columns=['Age1','Weight1','Height1'])
#olympicData=olympicData.drop(columns=['Medal']) #We will keep 'Medal' as categorical, as in this case we won't apply any ML model.
# # We will also keep MedalNumerical.

```


```python
#checking for null values
print("Sum of NULL values in each column. ")
nullSum=(olympicData).isnull().sum()
nullSum.sort_values(ascending=False).head(5)
```

    Sum of NULL values in each column. 
    




    Medal     231333
    Weight       217
    Height        99
    ID             0
    Name           0
    dtype: int64




```python
ratio=nullSum/olympicData.shape[0]
ratio.sort_values(ascending=False).head(5)
```




    Medal     0.853262
    Weight    0.000800
    Height    0.000365
    ID        0.000000
    Name      0.000000
    dtype: float64




```python
# Replace NaNs in columns with the
# mean of values in the same column
olympicData['Age'].fillna(value=olympicData['Age'].mean(), inplace=True)
olympicData['Height'].fillna(value=olympicData['Height'].mean(), inplace=True)
olympicData['Weight'].fillna(value=olympicData['Weight'].mean(), inplace=True)
```


```python
#checking for null values
print("Sum of NULL values in each column. ")
nullSum=(olympicData).isnull().sum()
nullSum.sort_values(ascending=False).head(5)
```

    Sum of NULL values in each column. 
    




    Medal    231333
    ID            0
    Name          0
    Sex           0
    Age           0
    dtype: int64



## 2.4. ISO Codes
The NOC codes are not compatible with ISO 3166-1 alpha-3 standard. And the ISO codes are the ones we must use for interactive plotting purposes. Also, we add the continents into the dataset.


```python
codeMapping = pd.read_excel('dataset/codesMapping.xls', sheet_name = 'mapping')
codeMapping=codeMapping.reset_index()
olympicDataISO =(olympicData.merge(codeMapping, left_on='NOC', right_on='IOC'))
olympicDataISO.drop(columns=['index','IOC'],inplace=True)
```

## 2.5. Continent Codes


```python
#We will also merge with the regions (continents)
```


```python
continent=pd.read_csv('https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv')
```


```python
continent=continent.reset_index()
olympicDataISO =(olympicDataISO.merge(continent, left_on='ISO', right_on='alpha-3'))
olympicDataISO.drop(columns=['index','name','alpha-2','alpha-3','country-code','iso_3166-2','intermediate-region','region-code','sub-region-code','intermediate-region-code'],inplace=True)
olympicDataISO.rename(columns={'region':'Continent'},inplace=True)
```

## 2.6. Season Filtering
In this report we will only analyzing Summer Games


```python
olympicDataISO=olympicDataISO[olympicDataISO.Season=='Summer']
```

---
# 3. QUERIES

# 3.1 Age Distribution of Participants


```python
totalAgeDistribution=olympicDataISO.groupby('Age')[['Medal']].count()
totalAgeDistribution.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Medal</th>
    </tr>
    <tr>
      <th>Age</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10.0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>11.0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>12.0</th>
      <td>6</td>
    </tr>
    <tr>
      <th>13.0</th>
      <td>12</td>
    </tr>
    <tr>
      <th>14.0</th>
      <td>68</td>
    </tr>
  </tbody>
</table>
</div>



Here we have the top 5 youngest Olympian athletes 


```python
totalAgeDistribution.tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Medal</th>
    </tr>
    <tr>
      <th>Age</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>81.0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>84.0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>88.0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>96.0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>97.0</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



In this case, we have the top 5 oldest Olympians.

Taking a look into the head (5 first rows) and the tail (5 last tails) of the Age (ordered numerically in ascending direction), we can see the maximums and the minimums.

It's interesting to see the youngest and oldest Olympians. But, now, what is even more interesting, is that the youngest athlete to participate in any Olympic Games also won a medal.

### 3.1.1 Youngest Olympic


```python
# Youngest Olympian (who is also the youngest medalist)
olympicDataISO[olympicDataISO.Age==totalAgeDistribution.reset_index().Age.min()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>Team</th>
      <th>NOC</th>
      <th>Games</th>
      <th>Year</th>
      <th>Season</th>
      <th>City</th>
      <th>Sport</th>
      <th>Event</th>
      <th>Medal</th>
      <th>MedalNumeric</th>
      <th>Country</th>
      <th>ISO</th>
      <th>Continent</th>
      <th>sub-region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>99108</th>
      <td>71691</td>
      <td>Dimitrios Loundras</td>
      <td>M</td>
      <td>10.0</td>
      <td>159.0</td>
      <td>66.0</td>
      <td>Ethnikos Gymnastikos Syllogos</td>
      <td>GRE</td>
      <td>1896 Summer</td>
      <td>1896</td>
      <td>Summer</td>
      <td>Athina</td>
      <td>Gymnastics</td>
      <td>Gymnastics Men's Parallel Bars, Teams</td>
      <td>Bronze</td>
      <td>1</td>
      <td>Greece</td>
      <td>GRC</td>
      <td>Europe</td>
      <td>Southern Europe</td>
    </tr>
  </tbody>
</table>
</div>



Dimitrios Loundras (6 September 1885 – 15 February 1970) was a Greek gymnast and naval officer who competed at the 1896 Summer Olympics in Athens. He was the last surviving participant of these Games.

Loundras competed in the team beams event. In that competition, Loundras was a member of the Ethnikos Gymnastikos Syllogos team that placed sixty-ninth of the sixty teams in the event, giving him a bronze medal. At 10 years 218 days he remains the youngest medalist and competitor in Olympic history.

<i>(extracted from: https://en.wikipedia.org/wiki/Dimitrios_Loundras)</i>

### 3.1.2 Oldest Olympian


```python
#Oldest Olympian
olympicDataISO[olympicDataISO.Age==totalAgeDistribution.reset_index().Age.max()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>Team</th>
      <th>NOC</th>
      <th>Games</th>
      <th>Year</th>
      <th>Season</th>
      <th>City</th>
      <th>Sport</th>
      <th>Event</th>
      <th>Medal</th>
      <th>MedalNumeric</th>
      <th>Country</th>
      <th>ISO</th>
      <th>Continent</th>
      <th>sub-region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32196</th>
      <td>128719</td>
      <td>John Quincy Adams Ward</td>
      <td>M</td>
      <td>97.0</td>
      <td>173.369803</td>
      <td>76.0</td>
      <td>United States</td>
      <td>USA</td>
      <td>1928 Summer</td>
      <td>1928</td>
      <td>Summer</td>
      <td>Amsterdam</td>
      <td>Art Competitions</td>
      <td>Art Competitions Mixed Sculpturing, Statues</td>
      <td>NaN</td>
      <td>0</td>
      <td>United States</td>
      <td>USA</td>
      <td>Americas</td>
      <td>Northern America</td>
    </tr>
  </tbody>
</table>
</div>



John Quincy Adams Ward (June 29, 1830 – May 1, 1910) was an American sculptor, whose most familiar work is his larger than life-size standing statue of George Washington on the steps of Federal Hall National Memorial in New York City.

His work was part of the sculpture event in the art competition at the 1928 Summer Olympics.

<i>(extracted from: https://en.wikipedia.org/wiki/John_Quincy_Adams_Ward)</i>

This is very curious. John Quincy Adams Ward participated in the 1928 Olympics post-mortem. So, his art participated but not him personally.

This is another interesting fact that we will explode later: the sport where the oldest Olympian participated in was Art Competitions.

### 3.1.3 Age Distribution in the Olympic History


```python
sns.histplot(data=totalAgeDistribution, x="Age", kde=True)
plt.show()
```


    
![png](genericExploration_files/genericExploration_64_0.png)
    


This graphic goes in line with our estimation from the statistics in Point 1.4. Most of the Olympians are between 21 and 29 years old.


```python
totalAgeDistributionGold=olympicDataISO[olympicDataISO.Medal=='Gold'].groupby('Age')[['Medal']].count()
totalAgeDistributionGold
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Medal</th>
    </tr>
    <tr>
      <th>Age</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13.0</th>
      <td>6</td>
    </tr>
    <tr>
      <th>14.0</th>
      <td>24</td>
    </tr>
    <tr>
      <th>15.0</th>
      <td>50</td>
    </tr>
    <tr>
      <th>16.0</th>
      <td>92</td>
    </tr>
    <tr>
      <th>17.0</th>
      <td>155</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>58.0</th>
      <td>3</td>
    </tr>
    <tr>
      <th>59.0</th>
      <td>2</td>
    </tr>
    <tr>
      <th>60.0</th>
      <td>4</td>
    </tr>
    <tr>
      <th>63.0</th>
      <td>4</td>
    </tr>
    <tr>
      <th>64.0</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>102 rows × 1 columns</p>
</div>



Here we can see how old were the oldest Olympians to win a Gold medal, and how young were the youngest to also get the maximum reward.

### 3.1.4 Gold Medalist Age Distribution in the Olympic History


```python
sns.histplot(data=totalAgeDistributionGold, x="Age", kde=True)
plt.show()
```




    <AxesSubplot:xlabel='Age', ylabel='Count'>




    
![png](genericExploration_files/genericExploration_69_1.png)
    


We can see that the Gold Medalist's Age concentration is mainly between 19 and 26 years.

### 3.1.5 Gold Medals for Athletes Over 50 based on Sports


Let's see in which sports Olympians older than 50 years got Gold Medals.


```python
goldMedalistOver50=olympicDataISO[(olympicDataISO.Age>50)&(olympicDataISO.Medal=='Gold')].groupby('Sport')[['Medal']].count()
goldMedalistOver50
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Medal</th>
    </tr>
    <tr>
      <th>Sport</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Archery</th>
      <td>12</td>
    </tr>
    <tr>
      <th>Art Competitions</th>
      <td>8</td>
    </tr>
    <tr>
      <th>Croquet</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Equestrianism</th>
      <td>18</td>
    </tr>
    <tr>
      <th>Roque</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Sailing</th>
      <td>12</td>
    </tr>
    <tr>
      <th>Shooting</th>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12,6))
ax = sns.barplot( x="Sport",
                    y='Medal',
                    #hue="Sex", 
                    data=goldMedalistOver50.reset_index())
plt.xticks(rotation=90)
plt.show()
```


    
![png](genericExploration_files/genericExploration_74_0.png)
    


Equestrian is the sport that most has Over-50-yo-Gold-Medals. Followed by Archery, Sailling and Shooting.

## 3.2 Sports in the Olympic Games statistics
According to the statistics obtained from the dataset, there are 52 different Sports that, at least once, were disputed at the Games. These 52 Sports were composed of 651 different Events.

Some interesting sports we find in the list are

- Aeronautics
- Lacrosse
- Motorboating
- Polo
- Roque
- Art Competitions
- Tug-Of-War

### 3.2.1 How many times each sport make it in the Olympics?


```python
olympicDataISO[olympicDataISO.Season=='Summer'].groupby(['Sport'])[['Games']].nunique().sort_values('Games')
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Games</th>
    </tr>
    <tr>
      <th>Sport</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aeronautics</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Rugby Sevens</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Roque</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Racquets</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Motorboating</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Ice Hockey</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Croquet</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Cricket</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Jeu De Paume</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Basque Pelota</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Alpinism</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Figure Skating</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Lacrosse</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Golf</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Softball</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Rugby</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Polo</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Triathlon</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Trampolining</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Baseball</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Taekwondo</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Tug-Of-War</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Beach Volleyball</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Badminton</th>
      <td>7</td>
    </tr>
    <tr>
      <th>Art Competitions</th>
      <td>7</td>
    </tr>
    <tr>
      <th>Table Tennis</th>
      <td>8</td>
    </tr>
    <tr>
      <th>Rhythmic Gymnastics</th>
      <td>9</td>
    </tr>
    <tr>
      <th>Synchronized Swimming</th>
      <td>9</td>
    </tr>
    <tr>
      <th>Judo</th>
      <td>13</td>
    </tr>
    <tr>
      <th>Handball</th>
      <td>13</td>
    </tr>
    <tr>
      <th>Volleyball</th>
      <td>14</td>
    </tr>
    <tr>
      <th>Archery</th>
      <td>16</td>
    </tr>
    <tr>
      <th>Tennis</th>
      <td>16</td>
    </tr>
    <tr>
      <th>Canoeing</th>
      <td>19</td>
    </tr>
    <tr>
      <th>Basketball</th>
      <td>19</td>
    </tr>
    <tr>
      <th>Hockey</th>
      <td>23</td>
    </tr>
    <tr>
      <th>Modern Pentathlon</th>
      <td>24</td>
    </tr>
    <tr>
      <th>Boxing</th>
      <td>25</td>
    </tr>
    <tr>
      <th>Equestrianism</th>
      <td>25</td>
    </tr>
    <tr>
      <th>Sailing</th>
      <td>26</td>
    </tr>
    <tr>
      <th>Weightlifting</th>
      <td>26</td>
    </tr>
    <tr>
      <th>Shooting</th>
      <td>27</td>
    </tr>
    <tr>
      <th>Football</th>
      <td>27</td>
    </tr>
    <tr>
      <th>Water Polo</th>
      <td>27</td>
    </tr>
    <tr>
      <th>Diving</th>
      <td>27</td>
    </tr>
    <tr>
      <th>Rowing</th>
      <td>28</td>
    </tr>
    <tr>
      <th>Wrestling</th>
      <td>28</td>
    </tr>
    <tr>
      <th>Swimming</th>
      <td>29</td>
    </tr>
    <tr>
      <th>Fencing</th>
      <td>29</td>
    </tr>
    <tr>
      <th>Cycling</th>
      <td>29</td>
    </tr>
    <tr>
      <th>Athletics</th>
      <td>29</td>
    </tr>
    <tr>
      <th>Gymnastics</th>
      <td>29</td>
    </tr>
  </tbody>
</table>
</div>


## 3.3 Gender equality over the Olympic Games

Let's now analyze a critical point where the IOC is paying special attention and it's the Gender Equality in Sports.


```python
gendercount=olympicDataISO.groupby(['Sex','Year'])[['ID']].count().reset_index()
gendercount.rename(columns={'ID':'count'},inplace=True)
gendercount.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Year</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>F</td>
      <td>1900</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F</td>
      <td>1904</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F</td>
      <td>1906</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F</td>
      <td>1908</td>
      <td>47</td>
    </tr>
    <tr>
      <th>4</th>
      <td>F</td>
      <td>1912</td>
      <td>85</td>
    </tr>
  </tbody>
</table>
</div>



As we see, in the first Games, no women participated. They started participating in 1900, but in small numbers.


```python
plt.figure(figsize=(12,6))
ax = sns.countplot(x="Year", hue="Sex", data=olympicDataISO)
plt.xticks(rotation=90)
plt.show()
```


    
![png](genericExploration_files/genericExploration_83_0.png)
    


Through the year, women's participation has increased, reaching Olympians' gender balance at the 2020 Tokyo Olympic Games.


```python
plt.figure(figsize=(12,6))
sns.lineplot(data=gendercount,x="Year", y='count', hue="Sex", markers= ["o","<"])
plt.xticks(rotation=90)
plt.show()
```


    
![png](genericExploration_files/genericExploration_85_0.png)
    


Same graphic as above, but instead of bars, this is a line plot. Here is easier to see the two things:

- The gap between the number of men and women athletes through the history of the Olympic Games.
- The increasing tendency of athletes number through the Games.


### 3.3.1 Medals per Gender per edition of the Games

The idea here will be to analyze the number of medals awarded to both genders through the Games' history. 


```python
medalsPerGenderOverYears=olympicDataISO.groupby(['Year','Sex'])[['MedalNumeric']].sum()
medalsPerGenderOverYears.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>MedalNumeric</th>
    </tr>
    <tr>
      <th>Year</th>
      <th>Sex</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1896</th>
      <th>M</th>
      <td>310</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">1900</th>
      <th>F</th>
      <td>23</td>
    </tr>
    <tr>
      <th>M</th>
      <td>1207</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">1904</th>
      <th>F</th>
      <td>24</td>
    </tr>
    <tr>
      <th>M</th>
      <td>971</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">1906</th>
      <th>F</th>
      <td>12</td>
    </tr>
    <tr>
      <th>M</th>
      <td>913</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">1908</th>
      <th>F</th>
      <td>33</td>
    </tr>
    <tr>
      <th>M</th>
      <td>1610</td>
    </tr>
    <tr>
      <th>1912</th>
      <th>F</th>
      <td>55</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12,6))
ax = sns.lineplot(data=medalsPerGenderOverYears.reset_index(),x="Year", y='MedalNumeric', hue="Sex", markers= ["o","<"])
plt.xticks(rotation=90)
plt.show()
```


    
![png](genericExploration_files/genericExploration_90_0.png)
    


The biggest difference in the medals distribution per gender is found in the 1920 Antwerp Olympic Games (except the 1896 Athen Olympic Games where no women participated).

The gap will be nulled once the gender balance in events is found (without a necessary gender balance in terms of the number of athletes). 

Further analysis at 2020 Tokyo Olympic Games statistics must be done to conclude if the tendency of gender balance in the number of athletes and number of events was reached.

## 3.4 Top 10 Gold Medal Countries

Let's analyze the Top 10 Gold Medal Countries.

* Note: in this dataset, we are analyzing gross medals. This means The Men's Basketball event awards 1 Gold, 1 Silver, and 1 Bronze medal. For computing in the regular NOC's statistics, the NOC who wins the final will be computed with 1 Gold medal. However, in this dataset, we count how many athletes got a Gold medal. Having this said, and following the logic above described: here we will be counting 12 Gold Medals (the number of athletes a NOC Basketball team will have). That is why we call it 'gross count'.


```python
top10 = olympicDataISO[olympicDataISO.Medal=='Gold'].groupby('NOC')[['MedalNumeric']].\
sum().sort_values('MedalNumeric',ascending=False).head(10)
top10
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedalNumeric</th>
    </tr>
    <tr>
      <th>NOC</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>USA</th>
      <td>7416</td>
    </tr>
    <tr>
      <th>GBR</th>
      <td>1908</td>
    </tr>
    <tr>
      <th>GER</th>
      <td>1776</td>
    </tr>
    <tr>
      <th>ITA</th>
      <td>1554</td>
    </tr>
    <tr>
      <th>FRA</th>
      <td>1395</td>
    </tr>
    <tr>
      <th>HUN</th>
      <td>1296</td>
    </tr>
    <tr>
      <th>SWE</th>
      <td>1062</td>
    </tr>
    <tr>
      <th>AUS</th>
      <td>1026</td>
    </tr>
    <tr>
      <th>CHN</th>
      <td>1002</td>
    </tr>
    <tr>
      <th>RUS</th>
      <td>888</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12,6))
ax = sns.barplot( x="NOC",
                    y='MedalNumeric',
                    #hue="Sex", 
                    data=top10.reset_index())
plt.xticks(rotation=90)
plt.show()
```


    
![png](genericExploration_files/genericExploration_95_0.png)
    


The difference shown by the USA against the rest of the NOCs is huge.

## 3.5 Disciplines with the greatest number of Gold Medals


```python
topDisciplines=olympicDataISO[olympicDataISO.Medal=='Gold'].groupby('Event')[['Medal']].count().sort_values('Medal',ascending=False)
topDisciplines=topDisciplines.head(10)
topDisciplines
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Medal</th>
    </tr>
    <tr>
      <th>Event</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Hockey Men's Hockey</th>
      <td>360</td>
    </tr>
    <tr>
      <th>Football Men's Football</th>
      <td>329</td>
    </tr>
    <tr>
      <th>Water Polo Men's Water Polo</th>
      <td>228</td>
    </tr>
    <tr>
      <th>Rowing Men's Coxed Eights</th>
      <td>226</td>
    </tr>
    <tr>
      <th>Gymnastics Men's Team All-Around</th>
      <td>217</td>
    </tr>
    <tr>
      <th>Basketball Men's Basketball</th>
      <td>198</td>
    </tr>
    <tr>
      <th>Hockey Women's Hockey</th>
      <td>158</td>
    </tr>
    <tr>
      <th>Swimming Men's 4 x 200 metres Freestyle Relay</th>
      <td>136</td>
    </tr>
    <tr>
      <th>Fencing Men's epee, Team</th>
      <td>134</td>
    </tr>
    <tr>
      <th>Swimming Women's 4 x 100 metres Freestyle Relay</th>
      <td>120</td>
    </tr>
  </tbody>
</table>
</div>



As we can see, the Top 10 of sports that more medals award, are team sports. These numbers make sense, as we are doing a gross count.

It's interesting to notice that the first women's sport it's in the 7th position on the top list. And, from the Top 10, we only have two women's sports in total. 


```python
plt.figure(figsize=(12,6))
ax = sns.barplot( x="Medal",
                    y='Event',
                    #hue="Sex", 
                    #orient='h',
                    data=topDisciplines.reset_index())
plt.xticks(rotation=90)
plt.show()
```


    
![png](genericExploration_files/genericExploration_100_0.png)
    


For example, for understanding the difference in the number of medals awarded in with Event of Hockey (Men and Women), we need to contextualize how long these sports have been part of the Olympic Program:


```python
theMostEvents=olympicDataISO[(olympicDataISO.Season=='Summer')].groupby(['Event'])[['Games']].nunique().sort_values('Games',ascending=False)
theMostEvent=theMostEvents.reset_index()
quantityOfGames=theMostEvent[theMostEvent.Event.isin(topDisciplines.reset_index().Event)]
quantityOfGames
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Event</th>
      <th>Games</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>Water Polo Men's Water Polo</td>
      <td>27</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Gymnastics Men's Team All-Around</td>
      <td>27</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Football Men's Football</td>
      <td>27</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Rowing Men's Coxed Eights</td>
      <td>27</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Swimming Men's 4 x 200 metres Freestyle Relay</td>
      <td>25</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Fencing Men's epee, Team</td>
      <td>25</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Swimming Women's 4 x 100 metres Freestyle Relay</td>
      <td>24</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Hockey Men's Hockey</td>
      <td>23</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Basketball Men's Basketball</td>
      <td>19</td>
    </tr>
    <tr>
      <th>195</th>
      <td>Hockey Women's Hockey</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>


If we compare Men and Women's Hockey, we see Men's Hockey was part of the Olympic Program 23 times, while the Women's Event was only part 10 times.

Also, when comparing across sports, we would need to create a ratio, and do not compare absolute values, as each sport have their own rules in terms of team cap.


```python
theCount=olympicDataISO[(olympicDataISO.Season=='Summer') & (olympicDataISO.Event.isin(topDisciplines.reset_index().Event))].groupby(['Event','Year','NOC'])[['ID']].count()
theCount.reset_index().groupby('Event')[['ID']].mean()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
    </tr>
    <tr>
      <th>Event</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basketball Men's Basketball</th>
      <td>11.756000</td>
    </tr>
    <tr>
      <th>Fencing Men's epee, Team</th>
      <td>4.772455</td>
    </tr>
    <tr>
      <th>Football Men's Football</th>
      <td>14.851541</td>
    </tr>
    <tr>
      <th>Gymnastics Men's Team All-Around</th>
      <td>7.973978</td>
    </tr>
    <tr>
      <th>Hockey Men's Hockey</th>
      <td>15.524000</td>
    </tr>
    <tr>
      <th>Hockey Women's Hockey</th>
      <td>15.855556</td>
    </tr>
    <tr>
      <th>Rowing Men's Coxed Eights</th>
      <td>9.197425</td>
    </tr>
    <tr>
      <th>Swimming Men's 4 x 200 metres Freestyle Relay</th>
      <td>4.438272</td>
    </tr>
    <tr>
      <th>Swimming Women's 4 x 100 metres Freestyle Relay</th>
      <td>4.382022</td>
    </tr>
    <tr>
      <th>Water Polo Men's Water Polo</th>
      <td>10.603509</td>
    </tr>
  </tbody>
</table>
</div>


Here we have the average of team members (only athletes) each sport has in their respective Olympic history.

The average stands because the sports change their respective IF rules from time to time. One of the rules is the number of athletes available to compete per Event, the number of athletes that can be in the field at the same time, the number of athletes per team, etc.

On top of that, we have NOCs that sometimes do not make the total cap number, taking fewer athletes to the competition.

## 3.6 Height vs Weight of Olympic Gold Medalists

Let's analyze Height vs Weight of Men and Women Gold Medalists through history.


```python
WeightVsHeight = olympicDataISO[olympicDataISO.Medal=='Gold'][['Weight','Height','Sex']]
sns.scatterplot(data=WeightVsHeight, x="Height", y="Weight",hue='Sex',alpha=0.5)
plt.title('Height vs Weight vs Gender')
plt.show()
```


    
![png](genericExploration_files/genericExploration_108_0.png)
    



```python
olympicDataISO[(olympicDataISO.Season=='Summer')].Sex.value_counts()/olympicDataISO[(olympicDataISO.Season=='Summer')].shape[0]
```


    M    0.731465
    F    0.268535
    Name: Sex, dtype: float64


Despite this graphic lack of contextualization, it's interesting to see this scatter plot. One thing we can clearly see is that women then to be smaller and lighter than men. However, we need to clarify something: in this dataset, for Summer Olympic Games, we have the information of 152724 men vs 56068 women. We are talking about the 73.14% vs 26.86%.

Now, coming back to the 'lack of contextualization' idea: this is because we are comparing all the sports, and comparing them it's incorrect if you want to pay attention to the physiognomy. Why? Because combat sports (Boxing, Wrestling, Judo & Taekwondo), and Weightlifting are also weight-regulated sports. There is an internal classification where you compete only in a certain category. To meet the requirements of that category there is a natural body description. Having this said, these sports must be compared, not only against themselves but also only between the athletes inside each Event/Category.

Different it's the sports of Time & Mark, where the only classification criteria are reaching the mark. Nevertheless, when you start paying attention to the statistics, you see a homogeneity across the values of the athletes inside a specific Event, even in a Time & Mark sport.

In conclusion, to analyze in a deeper way body measures, you must compare only across athletes of the same Event.

## 3.7 Variation of Age for Male Athletes over time


Let's analyze the variation of the Men Olympian's Age across Olympic history.


```python
maleAgeVar=olympicDataISO[olympicDataISO.Sex=='M'].groupby('Year')[['Age']].agg([min,np.mean,np.median,max])
maleAgeVar=maleAgeVar.Age
maleAgeVar.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>mean</th>
      <th>median</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1896</th>
      <td>10.0</td>
      <td>25.004807</td>
      <td>24.649695</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>1900</th>
      <td>15.0</td>
      <td>29.043367</td>
      <td>27.788415</td>
      <td>71.0</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>14.0</td>
      <td>26.361537</td>
      <td>24.887500</td>
      <td>71.0</td>
    </tr>
    <tr>
      <th>1906</th>
      <td>13.0</td>
      <td>27.098638</td>
      <td>25.187500</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>1908</th>
      <td>14.0</td>
      <td>27.037845</td>
      <td>25.000000</td>
      <td>61.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12,6))
sns.lineplot(data=maleAgeVar,x=maleAgeVar.index, y='max', markers= ["o","<"],label='max')
sns.lineplot(data=maleAgeVar,x=maleAgeVar.index, y='median', markers= ["o","<"],label='median')
sns.lineplot(data=maleAgeVar,x=maleAgeVar.index, y='mean', markers= ["o","<"],label='mean')
sns.lineplot(data=maleAgeVar,x=maleAgeVar.index, y='min', markers= ["o","<"],label='min')
plt.xticks(rotation=90)
plt.title("Men's Age Variation through the Olympic Games")
plt.ylabel('Variation')
plt.show()
```


    
![png](genericExploration_files/genericExploration_114_0.png)
    


We can see that both the median (the middle value) and the mean (average) are quite constant over time. Also, we can see some great peaks around the 30s. As we read above, in the 1928 Los Angeles Olympic Games, for example, we registered the oldest Olympian, who participated in Art Competition. 

## 3.8 Variation of Age for Female Athletes over time


```python
femaleAgeVar=olympicDataISO[olympicDataISO.Sex=='F'].groupby('Year')[['Age']].agg([min,np.mean,np.median,max])
femaleAgeVar=femaleAgeVar.Age
femaleAgeVar.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>mean</th>
      <th>median</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1900</th>
      <td>13.000000</td>
      <td>30.010635</td>
      <td>29.038462</td>
      <td>46.000000</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>24.000000</td>
      <td>49.909375</td>
      <td>55.000000</td>
      <td>63.000000</td>
    </tr>
    <tr>
      <th>1906</th>
      <td>21.000000</td>
      <td>25.234819</td>
      <td>24.116906</td>
      <td>29.038462</td>
    </tr>
    <tr>
      <th>1908</th>
      <td>19.414545</td>
      <td>34.642863</td>
      <td>36.000000</td>
      <td>54.000000</td>
    </tr>
    <tr>
      <th>1912</th>
      <td>13.000000</td>
      <td>22.411765</td>
      <td>21.000000</td>
      <td>45.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12,6))
sns.lineplot(data=femaleAgeVar,x=femaleAgeVar.index, y='max', markers= ["o","<"],label='max')
sns.lineplot(data=femaleAgeVar,x=femaleAgeVar.index, y='median', markers= ["o","<"],label='median')

sns.lineplot(data=femaleAgeVar,x=femaleAgeVar.index, y='mean', markers= ["o","<"],label='mean')
sns.lineplot(data=femaleAgeVar,x=femaleAgeVar.index, y='min', markers= ["o","<"],label='min')

plt.xticks(rotation=90)
plt.title("Women's Age Variation through the Olympic Games")
plt.ylabel('Variation')
plt.show()
```


    
![png](genericExploration_files/genericExploration_118_0.png)
    


For Women, we can see an increasing tendency in the average and median registered ages since the 1980 Moscow Olympic Games.

## 3.9 Variation of Weight for Male Athletes over time

What about the weight? How this measure has fluctuated over time?

* It's important to mention that, to take any substantial conclusion from this measure, we would have to divide by Events.


```python
maleWeightVar=olympicDataISO[olympicDataISO.Sex=='M'].groupby('Year')[['Weight']].agg([min,np.mean,np.median,max])
maleWeightVar=maleWeightVar.Weight
maleWeightVar.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>mean</th>
      <th>median</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1896</th>
      <td>45.0</td>
      <td>72.364564</td>
      <td>70.963234</td>
      <td>115.269896</td>
    </tr>
    <tr>
      <th>1900</th>
      <td>51.0</td>
      <td>75.681802</td>
      <td>76.173007</td>
      <td>115.269896</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>43.0</td>
      <td>72.032654</td>
      <td>69.970678</td>
      <td>115.269896</td>
    </tr>
    <tr>
      <th>1906</th>
      <td>52.0</td>
      <td>75.583481</td>
      <td>75.778938</td>
      <td>115.269896</td>
    </tr>
    <tr>
      <th>1908</th>
      <td>51.0</td>
      <td>73.867432</td>
      <td>74.000000</td>
      <td>115.269896</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12,6))
sns.lineplot(data=maleWeightVar,x=maleWeightVar.index, y='max', markers= ["o","<"],label='max')

sns.lineplot(data=maleWeightVar,x=maleWeightVar.index, y='median', markers= ["o","<"],label='median')
sns.lineplot(data=maleWeightVar,x=maleWeightVar.index, y='mean', markers= ["o","<"],label='mean')

sns.lineplot(data=maleWeightVar,x=maleWeightVar.index, y='min', markers= ["o","<"],label='min')

plt.xticks(rotation=90)
plt.title("Men's Weight Variation through the Olympic Games")
plt.ylabel('Variation')

plt.show()
```


    
![png](genericExploration_files/genericExploration_123_0.png)
    


We can see that both the median (the middle value) and the mean (average) are showing a slight increase over the last 20 years.

## 3.10 Variation of Weight for Female Athletes over time


```python
womenWeightVar=olympicDataISO[olympicDataISO.Sex=='F'].groupby('Year')[['Weight']].agg([min,np.mean,np.median,max])
womenWeightVar=womenWeightVar.Weight
womenWeightVar.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>mean</th>
      <th>median</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1900</th>
      <td>61.979466</td>
      <td>67.837673</td>
      <td>67.648002</td>
      <td>76.509324</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>69.970678</td>
      <td>69.970678</td>
      <td>69.970678</td>
      <td>69.970678</td>
    </tr>
    <tr>
      <th>1906</th>
      <td>61.979466</td>
      <td>67.169175</td>
      <td>61.979466</td>
      <td>73.396825</td>
    </tr>
    <tr>
      <th>1908</th>
      <td>51.119403</td>
      <td>67.111217</td>
      <td>69.970678</td>
      <td>77.000000</td>
    </tr>
    <tr>
      <th>1912</th>
      <td>61.674173</td>
      <td>64.688523</td>
      <td>61.979466</td>
      <td>73.396825</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12,6))
sns.lineplot(data=womenWeightVar,x=womenWeightVar.index, y='max', markers= ["o","<"],label='max')
sns.lineplot(data=womenWeightVar,x=womenWeightVar.index, y='median', markers= ["o","<"],label='median')

sns.lineplot(data=womenWeightVar,x=womenWeightVar.index, y='mean', markers= ["o","<"],label='mean')
sns.lineplot(data=womenWeightVar,x=womenWeightVar.index, y='min', markers= ["o","<"],label='min')

plt.xticks(rotation=90)
plt.title("Women's Weight Variation through the Olympic Games")
plt.ylabel('Variation')
plt.show()
```


    
![png](genericExploration_files/genericExploration_127_0.png)
    


Same as Age, Women athletes are showing a slight increase in terms of mean and median in weight over the last 40 years.

In a further study, this value would have to be compared with the number of women's events over time, and see if there is any correlation. In a rough analysis, we can say this would make sense.

## 3.11 Variation of Height for Male Athletes over time


```python
maleHeightVar=olympicDataISO[olympicDataISO.Sex=='M'].groupby('Year')[['Height']].agg([min,np.mean,np.median,max])
maleHeightVar=maleHeightVar.Height
maleHeightVar.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>mean</th>
      <th>median</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1896</th>
      <td>154.0</td>
      <td>174.524546</td>
      <td>173.937657</td>
      <td>190.640466</td>
    </tr>
    <tr>
      <th>1900</th>
      <td>153.0</td>
      <td>177.241113</td>
      <td>178.345624</td>
      <td>191.000000</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>155.0</td>
      <td>175.113520</td>
      <td>173.000000</td>
      <td>195.000000</td>
    </tr>
    <tr>
      <th>1906</th>
      <td>165.0</td>
      <td>177.634190</td>
      <td>177.558979</td>
      <td>196.000000</td>
    </tr>
    <tr>
      <th>1908</th>
      <td>157.0</td>
      <td>176.667051</td>
      <td>177.146789</td>
      <td>201.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12,6))
sns.lineplot(data=maleHeightVar,x=maleHeightVar.index, y='max', markers= ["o","<"],label='max')
sns.lineplot(data=maleHeightVar,x=maleHeightVar.index, y='median', markers= ["o","<"],label='median')

sns.lineplot(data=maleHeightVar,x=maleHeightVar.index, y='mean', markers= ["o","<"],label='mean')
sns.lineplot(data=maleHeightVar,x=maleHeightVar.index, y='min', markers= ["o","<"],label='min')

plt.xticks(rotation=90)
plt.title("Men's Height Variation through the Olympic Games")
plt.ylabel('Variation')
plt.show()
```


    
![png](genericExploration_files/genericExploration_131_0.png)
    


We can see that both the median (the middle value) and the mean (average) show an increasing tendency over time.

This could mean that the athletes' population is getting taller.

## 3.12 Variation of Height for Female Athletes over time


```python
womenHeightVar=olympicDataISO[olympicDataISO.Sex=='F'].groupby('Year')[['Height']].agg([min,np.mean,np.median,max])
womenHeightVar=womenHeightVar.Height
womenHeightVar.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>mean</th>
      <th>median</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1900</th>
      <td>166.000000</td>
      <td>173.287562</td>
      <td>172.244399</td>
      <td>180.771429</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>173.086088</td>
      <td>173.086088</td>
      <td>173.086088</td>
      <td>173.086088</td>
    </tr>
    <tr>
      <th>1906</th>
      <td>172.244399</td>
      <td>176.120322</td>
      <td>172.244399</td>
      <td>180.771429</td>
    </tr>
    <tr>
      <th>1908</th>
      <td>161.313351</td>
      <td>172.653574</td>
      <td>173.086088</td>
      <td>181.000000</td>
    </tr>
    <tr>
      <th>1912</th>
      <td>156.833333</td>
      <td>171.575757</td>
      <td>172.244399</td>
      <td>180.771429</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12,6))
sns.lineplot(data=womenHeightVar,x=womenHeightVar.index, y='max', markers= ["o","<"],label='max')
sns.lineplot(data=womenHeightVar,x=womenHeightVar.index, y='median', markers= ["o","<"],label='median')

sns.lineplot(data=womenHeightVar,x=womenHeightVar.index, y='mean', markers= ["o","<"],label='mean')
sns.lineplot(data=womenHeightVar,x=womenHeightVar.index, y='min', markers= ["o","<"],label='min')

plt.xticks(rotation=90)
plt.title("Women's Height Variation through the Olympic Games")
plt.ylabel('Variation')

plt.show()
```


    
![png](genericExploration_files/genericExploration_135_0.png)
    


## 3.13 Weight over year for Male Gymnasts


```python
maleWeightGymVar=olympicDataISO[(olympicDataISO.Sex=='M') & (olympicDataISO.Sport=='Gymnastics')].groupby('Year')[['Weight']].agg([min,np.mean,np.median,max])
maleWeightGymVar=maleWeightGymVar.Weight
maleWeightGymVar.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>mean</th>
      <th>median</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1896</th>
      <td>56.000000</td>
      <td>64.846540</td>
      <td>63.359364</td>
      <td>102.0</td>
    </tr>
    <tr>
      <th>1900</th>
      <td>61.000000</td>
      <td>63.368068</td>
      <td>63.348624</td>
      <td>71.0</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>58.166601</td>
      <td>66.522063</td>
      <td>69.500000</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>1906</th>
      <td>59.000000</td>
      <td>68.786599</td>
      <td>63.348624</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>1908</th>
      <td>54.000000</td>
      <td>63.288332</td>
      <td>63.133305</td>
      <td>86.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12,6))

sns.lineplot(data=maleWeightGymVar,x=maleWeightGymVar.index, y='max', markers= ["o","<"],label='max')
sns.lineplot(data=maleWeightGymVar,x=maleWeightGymVar.index, y='median', markers= ["o","<"],label='median')
sns.lineplot(data=maleWeightGymVar,x=maleWeightGymVar.index, y='mean', markers= ["o","<"],label='mean')
sns.lineplot(data=maleWeightGymVar,x=maleWeightGymVar.index, y='min', markers= ["o","<"],label='min')

plt.xticks(rotation=90)
plt.title("Men's Weight Gymnastics Variation through the Olympic Games")
plt.ylabel('Variation')

plt.show()
```


    
![png](genericExploration_files/genericExploration_138_0.png)
    


We can see that both the median (the middle value) and the mean (average) are quite constant over time, despite the maximum and minimum weights aren't.

## 3.14 Weight over year for Female Gymnasts


```python
femaleWeightGymVar=olympicDataISO[(olympicDataISO.Sex=='F') & (olympicDataISO.Sport=='Gymnastics')].groupby('Year')[['Weight']].agg([min,np.mean,np.median,max])
femaleWeightGymVar=femaleWeightGymVar.Weight
femaleWeightGymVar.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>mean</th>
      <th>median</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1928</th>
      <td>47.322404</td>
      <td>47.322404</td>
      <td>47.322404</td>
      <td>47.322404</td>
    </tr>
    <tr>
      <th>1936</th>
      <td>43.000000</td>
      <td>48.129554</td>
      <td>47.322404</td>
      <td>62.000000</td>
    </tr>
    <tr>
      <th>1948</th>
      <td>47.322404</td>
      <td>47.456815</td>
      <td>47.322404</td>
      <td>57.000000</td>
    </tr>
    <tr>
      <th>1952</th>
      <td>47.322404</td>
      <td>49.974987</td>
      <td>47.867324</td>
      <td>63.000000</td>
    </tr>
    <tr>
      <th>1956</th>
      <td>47.322404</td>
      <td>51.637523</td>
      <td>50.000000</td>
      <td>61.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12,6))

sns.lineplot(data=femaleWeightGymVar,x=femaleWeightGymVar.index, y='max', markers= ["o","<"],label='max')
sns.lineplot(data=femaleWeightGymVar,x=femaleWeightGymVar.index, y='median', markers= ["o","<"],label='median')
sns.lineplot(data=femaleWeightGymVar,x=femaleWeightGymVar.index, y='mean', markers= ["o","<"],label='mean')
sns.lineplot(data=femaleWeightGymVar,x=femaleWeightGymVar.index, y='min', markers= ["o","<"],label='min')

plt.xticks(rotation=90)
plt.title("Women's Weight Gymnastics Variation through the Olympic Games")
plt.ylabel('Variation')

plt.show()
```


    
![png](genericExploration_files/genericExploration_142_0.png)
    


These figures are pretty interesting. We see that there is huge volatility in the weight of women gymnastic athletes over time.

## 3.15 Height over year for Male Lifters

In the case of WeightLifting, makes no sense to analyze the weight in a generic way. This is because it must be contextualized inside each Event/Category, which, by the way, also changes through time, as IF Rules change.

That is why it would be interesting analyzing the Height over the years.


```python
maleheightWLFVar=olympicDataISO[(olympicDataISO.Sex=='M') & (olympicDataISO.Sport=='Weightlifting')].groupby('Year')[['Height']].agg([min,np.mean,np.median,max])
maleheightWLFVar=maleheightWLFVar.Height
maleheightWLFVar.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>mean</th>
      <th>median</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1896</th>
      <td>159.000000</td>
      <td>175.385714</td>
      <td>174.357143</td>
      <td>188.000000</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>170.000000</td>
      <td>172.448980</td>
      <td>172.714286</td>
      <td>176.000000</td>
    </tr>
    <tr>
      <th>1906</th>
      <td>170.000000</td>
      <td>174.012987</td>
      <td>175.000000</td>
      <td>177.000000</td>
    </tr>
    <tr>
      <th>1920</th>
      <td>160.714777</td>
      <td>167.762877</td>
      <td>167.000000</td>
      <td>179.680751</td>
    </tr>
    <tr>
      <th>1924</th>
      <td>160.714777</td>
      <td>168.927445</td>
      <td>169.072848</td>
      <td>179.680751</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12,6))

sns.lineplot(data=maleheightWLFVar,x=maleheightWLFVar.index, y='max', markers= ["o","<"],label='max')
sns.lineplot(data=maleheightWLFVar,x=maleheightWLFVar.index, y='median', markers= ["o","<"],label='median')
sns.lineplot(data=maleheightWLFVar,x=maleheightWLFVar.index, y='mean', markers= ["o","<"],label='mean')
sns.lineplot(data=maleheightWLFVar,x=maleheightWLFVar.index, y='min', markers= ["o","<"],label='min')

plt.xticks(rotation=90)
plt.title("Men's Height Weightlifting Variation through the Olympic Games")
plt.ylabel('Variation')

plt.show()
```


    
![png](genericExploration_files/genericExploration_147_0.png)
    


It's interesting how the maximum height increased (we would need to analyze if this value is also correlated with an increase in the weight). Also, the mean and the median have U shapes.

The wider range shown since the 1980 Moscow Olympic Games, might mean that the number of categories was expanded over time, and/or, the range of the categories was increased.

## 3.16 Height over year for Female Lifters



```python
femaleheightWLFVar=olympicDataISO[(olympicDataISO.Sex=='F') & (olympicDataISO.Sport=='Weightlifting')].groupby('Year')[['Height']].agg([min,np.mean,np.median,max])
femaleheightWLFVar=femaleheightWLFVar.Height
femaleheightWLFVar.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>mean</th>
      <th>median</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000</th>
      <td>145.0</td>
      <td>160.717647</td>
      <td>161.0</td>
      <td>180.0</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>145.0</td>
      <td>160.676827</td>
      <td>160.0</td>
      <td>181.0</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>141.0</td>
      <td>160.206897</td>
      <td>160.0</td>
      <td>181.0</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>142.0</td>
      <td>160.169279</td>
      <td>160.0</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>145.0</td>
      <td>160.611650</td>
      <td>160.0</td>
      <td>178.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12,6))

sns.lineplot(data=femaleheightWLFVar,x=femaleheightWLFVar.index, y='max', markers= ["o","<"],label='max')
sns.lineplot(data=femaleheightWLFVar,x=femaleheightWLFVar.index, y='median', markers= ["o","<"],label='median')
sns.lineplot(data=femaleheightWLFVar,x=femaleheightWLFVar.index, y='mean', markers= ["o","<"],label='mean')
sns.lineplot(data=femaleheightWLFVar,x=femaleheightWLFVar.index, y='min', markers= ["o","<"],label='min')

plt.xticks(rotation=90)
plt.title("Women's Height Weightlifting Variation through the Olympic Games")
plt.ylabel('Variation')

plt.show()
```


    
![png](genericExploration_files/genericExploration_151_0.png)
    


We can see that both the median (the middle value) and the mean (average) are quite constant over time.

## 3.17 World Maps

Let's plot in world maps the medals distribution per color/type.

* There are countries that are not even plotted. This is because they have never won that color/type of medal.


```python
###CODE###
nocMedalsOrder=olympicDataISO.groupby(['NOC','ISO','Medal','MedalNumeric'])[['ID']].count().sort_values(['NOC','MedalNumeric'])
nocMedalsOrder.rename(columns={'ID':'Count'},inplace=True)
medalsDF=nocMedalsOrder.reset_index().pivot(index=['NOC','ISO'], columns='Medal', values='Count').fillna(0).astype(int)
medalsDF=medalsDF.loc[:,['Bronze','Silver','Gold']]
medalsDF['Total']=medalsDF['Bronze']+medalsDF['Silver']+medalsDF['Gold']
medalsDF=medalsDF.reset_index()
medalsDF=medalsDF.sort_values(['Gold','Silver','Bronze','Total'],ascending=False)
```

### 3.17.1 Gold Medals based on Countries


```python
df=medalsDF[medalsDF.Gold>0]
###CODE###
data = [dict(
    type='choropleth',
    locations=df['ISO'],
    z=df['Gold'],
    color_continuous_scale=px.colors.sequential.Plasma,
    text=df.apply(lambda row: f"{row['NOC']}<br>Total Medals: {row['Total']}<br>Gold Medals: {row['Gold']}<br>Silver Medals: {row['Silver']}<br>Bronze Medals: {row['Bronze']}", axis=1),
    hoverinfo="text",
    #autocolorscale=False,
    reversescale=True,
    colorscale = 'Blues',
    marker=dict(
        line=dict(
            color='rgb(180,180,180)',
            width=0.5
        )),
    colorbar=dict(
        autotick=False,
        tickprefix='',
        title='Gold'),
)]

layout = dict(
    colorscale = 'Blues',
    width=900,
    height=600,
    title="GOLD Medallists' NOCs",
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection=dict(
            type='Mercator'
        )
    )
)

fig = dict(data=data, layout=layout)

iplot(fig,validate=False)

```


<div>                            <div id="d83da5e5-7e5b-4378-b42f-aba9da5bc298" class="plotly-graph-div" style="height:600px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("d83da5e5-7e5b-4378-b42f-aba9da5bc298")) {                    Plotly.newPlot(                        "d83da5e5-7e5b-4378-b42f-aba9da5bc298",                        [{"type":"choropleth","locations":["USA","GBR","DEU","ITA","FRA","HUN","SWE","AUS","CHN","RUS","NLD","JPN","NOR","DNK","KOR","CUB","ROU","CAN","DEU","FIN","IND","POL","ESP","BRA","CHE","BEL","ARG","NZL","GRC","HRV","BGR","UKR","PAK","TUR","JAM","KEN","ZAF","URY","MEX","AUT","NGA","ETH","CMR","KAZ","BLR","IRN","ZWE","PRK","SRB","CZE","BHS","SVK","FJI","IDN","IRL","EST","THA","UZB","GEO","AZE","TTO","EGY","SVN","LTU","MAR","COL","DZA","PRT","LUX","TWN","LVA","CHL","TUN","DOM","MNG","ARM","VEN","UGA","PER","SGP","VNM","PRI","HKG","ISR","HTI","CRI","TJK","BHR","CIV","SYR","BDI","ECU","GRD","PAN","MOZ","SUR","ARE","JOR"],"z":[2472,636,592,518,465,432,354,342,334,296,245,230,227,179,171,164,161,158,144,132,131,111,109,109,99,96,91,90,62,54,53,42,42,40,38,34,32,31,30,29,23,22,20,19,18,18,17,16,15,15,14,13,13,11,9,9,9,9,8,7,7,7,6,6,6,5,5,4,4,3,3,3,3,3,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],"color_continuous_scale":["#0d0887","#46039f","#7201a8","#9c179e","#bd3786","#d8576b","#ed7953","#fb9f3a","#fdca26","#f0f921"],"text":["USA<br>Total Medals: 5002<br>Gold Medals: 2472<br>Silver Medals: 1333<br>Bronze Medals: 1197","GBR<br>Total Medals: 1985<br>Gold Medals: 636<br>Silver Medals: 729<br>Bronze Medals: 620","GER<br>Total Medals: 1779<br>Gold Medals: 592<br>Silver Medals: 538<br>Bronze Medals: 649","ITA<br>Total Medals: 1446<br>Gold Medals: 518<br>Silver Medals: 474<br>Bronze Medals: 454","FRA<br>Total Medals: 1627<br>Gold Medals: 465<br>Silver Medals: 575<br>Bronze Medals: 587","HUN<br>Total Medals: 1123<br>Gold Medals: 432<br>Silver Medals: 328<br>Bronze Medals: 363","SWE<br>Total Medals: 1108<br>Gold Medals: 354<br>Silver Medals: 396<br>Bronze Medals: 358","AUS<br>Total Medals: 1304<br>Gold Medals: 342<br>Silver Medals: 452<br>Bronze Medals: 510","CHN<br>Total Medals: 909<br>Gold Medals: 334<br>Silver Medals: 317<br>Bronze Medals: 258","RUS<br>Total Medals: 905<br>Gold Medals: 296<br>Silver Medals: 278<br>Bronze Medals: 331","NED<br>Total Medals: 918<br>Gold Medals: 245<br>Silver Medals: 302<br>Bronze Medals: 371","JPN<br>Total Medals: 850<br>Gold Medals: 230<br>Silver Medals: 287<br>Bronze Medals: 333","NOR<br>Total Medals: 590<br>Gold Medals: 227<br>Silver Medals: 196<br>Bronze Medals: 167","DEN<br>Total Medals: 592<br>Gold Medals: 179<br>Silver Medals: 236<br>Bronze Medals: 177","KOR<br>Total Medals: 552<br>Gold Medals: 171<br>Silver Medals: 206<br>Bronze Medals: 175","CUB<br>Total Medals: 409<br>Gold Medals: 164<br>Silver Medals: 129<br>Bronze Medals: 116","ROU<br>Total Medals: 651<br>Gold Medals: 161<br>Silver Medals: 200<br>Bronze Medals: 290","CAN<br>Total Medals: 741<br>Gold Medals: 158<br>Silver Medals: 239<br>Bronze Medals: 344","FRG<br>Total Medals: 504<br>Gold Medals: 144<br>Silver Medals: 172<br>Bronze Medals: 188","FIN<br>Total Medals: 474<br>Gold Medals: 132<br>Silver Medals: 125<br>Bronze Medals: 217","IND<br>Total Medals: 190<br>Gold Medals: 131<br>Silver Medals: 19<br>Bronze Medals: 40","POL<br>Total Medals: 538<br>Gold Medals: 111<br>Silver Medals: 185<br>Bronze Medals: 242","ESP<br>Total Medals: 487<br>Gold Medals: 109<br>Silver Medals: 243<br>Bronze Medals: 135","BRA<br>Total Medals: 475<br>Gold Medals: 109<br>Silver Medals: 175<br>Bronze Medals: 191","SUI<br>Total Medals: 416<br>Gold Medals: 99<br>Silver Medals: 178<br>Bronze Medals: 139","BEL<br>Total Medals: 455<br>Gold Medals: 96<br>Silver Medals: 193<br>Bronze Medals: 166","ARG<br>Total Medals: 274<br>Gold Medals: 91<br>Silver Medals: 92<br>Bronze Medals: 91","NZL<br>Total Medals: 227<br>Gold Medals: 90<br>Silver Medals: 55<br>Bronze Medals: 82","GRE<br>Total Medals: 255<br>Gold Medals: 62<br>Silver Medals: 109<br>Bronze Medals: 84","CRO<br>Total Medals: 138<br>Gold Medals: 54<br>Silver Medals: 48<br>Bronze Medals: 36","BUL<br>Total Medals: 336<br>Gold Medals: 53<br>Silver Medals: 142<br>Bronze Medals: 141","UKR<br>Total Medals: 188<br>Gold Medals: 42<br>Silver Medals: 51<br>Bronze Medals: 95","PAK<br>Total Medals: 121<br>Gold Medals: 42<br>Silver Medals: 45<br>Bronze Medals: 34","TUR<br>Total Medals: 95<br>Gold Medals: 40<br>Silver Medals: 27<br>Bronze Medals: 28","JAM<br>Total Medals: 157<br>Gold Medals: 38<br>Silver Medals: 75<br>Bronze Medals: 44","KEN<br>Total Medals: 106<br>Gold Medals: 34<br>Silver Medals: 41<br>Bronze Medals: 31","RSA<br>Total Medals: 131<br>Gold Medals: 32<br>Silver Medals: 47<br>Bronze Medals: 52","URU<br>Total Medals: 63<br>Gold Medals: 31<br>Silver Medals: 2<br>Bronze Medals: 30","MEX<br>Total Medals: 110<br>Gold Medals: 30<br>Silver Medals: 26<br>Bronze Medals: 54","AUT<br>Total Medals: 170<br>Gold Medals: 29<br>Silver Medals: 88<br>Bronze Medals: 53","NGR<br>Total Medals: 99<br>Gold Medals: 23<br>Silver Medals: 30<br>Bronze Medals: 46","ETH<br>Total Medals: 53<br>Gold Medals: 22<br>Silver Medals: 9<br>Bronze Medals: 22","CMR<br>Total Medals: 22<br>Gold Medals: 20<br>Silver Medals: 1<br>Bronze Medals: 1","KAZ<br>Total Medals: 70<br>Gold Medals: 19<br>Silver Medals: 22<br>Bronze Medals: 29","BLR<br>Total Medals: 124<br>Gold Medals: 18<br>Silver Medals: 40<br>Bronze Medals: 66","IRI<br>Total Medals: 68<br>Gold Medals: 18<br>Silver Medals: 21<br>Bronze Medals: 29","ZIM<br>Total Medals: 22<br>Gold Medals: 17<br>Silver Medals: 4<br>Bronze Medals: 1","PRK<br>Total Medals: 65<br>Gold Medals: 16<br>Silver Medals: 15<br>Bronze Medals: 34","SRB<br>Total Medals: 85<br>Gold Medals: 15<br>Silver Medals: 29<br>Bronze Medals: 41","CZE<br>Total Medals: 71<br>Gold Medals: 15<br>Silver Medals: 24<br>Bronze Medals: 32","BAH<br>Total Medals: 40<br>Gold Medals: 14<br>Silver Medals: 11<br>Bronze Medals: 15","SVK<br>Total Medals: 42<br>Gold Medals: 13<br>Silver Medals: 17<br>Bronze Medals: 12","FIJ<br>Total Medals: 13<br>Gold Medals: 13<br>Silver Medals: 0<br>Bronze Medals: 0","INA<br>Total Medals: 41<br>Gold Medals: 11<br>Silver Medals: 17<br>Bronze Medals: 13","IRL<br>Total Medals: 35<br>Gold Medals: 9<br>Silver Medals: 13<br>Bronze Medals: 13","EST<br>Total Medals: 43<br>Gold Medals: 9<br>Silver Medals: 10<br>Bronze Medals: 24","THA<br>Total Medals: 30<br>Gold Medals: 9<br>Silver Medals: 8<br>Bronze Medals: 13","UZB<br>Total Medals: 33<br>Gold Medals: 9<br>Silver Medals: 7<br>Bronze Medals: 17","GEO<br>Total Medals: 32<br>Gold Medals: 8<br>Silver Medals: 6<br>Bronze Medals: 18","AZE<br>Total Medals: 44<br>Gold Medals: 7<br>Silver Medals: 12<br>Bronze Medals: 25","TTO<br>Total Medals: 32<br>Gold Medals: 7<br>Silver Medals: 8<br>Bronze Medals: 17","EGY<br>Total Medals: 27<br>Gold Medals: 7<br>Silver Medals: 8<br>Bronze Medals: 12","SLO<br>Total Medals: 30<br>Gold Medals: 6<br>Silver Medals: 9<br>Bronze Medals: 15","LTU<br>Total Medals: 61<br>Gold Medals: 6<br>Silver Medals: 7<br>Bronze Medals: 48","MAR<br>Total Medals: 23<br>Gold Medals: 6<br>Silver Medals: 5<br>Bronze Medals: 12","COL<br>Total Medals: 28<br>Gold Medals: 5<br>Silver Medals: 9<br>Bronze Medals: 14","ALG<br>Total Medals: 17<br>Gold Medals: 5<br>Silver Medals: 4<br>Bronze Medals: 8","POR<br>Total Medals: 41<br>Gold Medals: 4<br>Silver Medals: 11<br>Bronze Medals: 26","LUX<br>Total Medals: 6<br>Gold Medals: 4<br>Silver Medals: 2<br>Bronze Medals: 0","TPE<br>Total Medals: 49<br>Gold Medals: 3<br>Silver Medals: 28<br>Bronze Medals: 18","LAT<br>Total Medals: 20<br>Gold Medals: 3<br>Silver Medals: 11<br>Bronze Medals: 6","CHI<br>Total Medals: 32<br>Gold Medals: 3<br>Silver Medals: 9<br>Bronze Medals: 20","TUN<br>Total Medals: 13<br>Gold Medals: 3<br>Silver Medals: 3<br>Bronze Medals: 7","DOM<br>Total Medals: 7<br>Gold Medals: 3<br>Silver Medals: 2<br>Bronze Medals: 2","MGL<br>Total Medals: 26<br>Gold Medals: 2<br>Silver Medals: 10<br>Bronze Medals: 14","ARM<br>Total Medals: 16<br>Gold Medals: 2<br>Silver Medals: 5<br>Bronze Medals: 9","VEN<br>Total Medals: 15<br>Gold Medals: 2<br>Silver Medals: 3<br>Bronze Medals: 10","UGA<br>Total Medals: 7<br>Gold Medals: 2<br>Silver Medals: 3<br>Bronze Medals: 2","PER<br>Total Medals: 15<br>Gold Medals: 1<br>Silver Medals: 14<br>Bronze Medals: 0","SGP<br>Total Medals: 9<br>Gold Medals: 1<br>Silver Medals: 4<br>Bronze Medals: 4","VIE<br>Total Medals: 4<br>Gold Medals: 1<br>Silver Medals: 3<br>Bronze Medals: 0","PUR<br>Total Medals: 9<br>Gold Medals: 1<br>Silver Medals: 2<br>Bronze Medals: 6","HKG<br>Total Medals: 4<br>Gold Medals: 1<br>Silver Medals: 2<br>Bronze Medals: 1","ISR<br>Total Medals: 9<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 7","HAI<br>Total Medals: 7<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 5","CRC<br>Total Medals: 4<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 2","TJK<br>Total Medals: 4<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 2","BRN<br>Total Medals: 3<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 1","CIV<br>Total Medals: 3<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 1","SYR<br>Total Medals: 3<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 1","BDI<br>Total Medals: 2<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 0","ECU<br>Total Medals: 2<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 0","GRN<br>Total Medals: 2<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 0","PAN<br>Total Medals: 3<br>Gold Medals: 1<br>Silver Medals: 0<br>Bronze Medals: 2","MOZ<br>Total Medals: 2<br>Gold Medals: 1<br>Silver Medals: 0<br>Bronze Medals: 1","SUR<br>Total Medals: 2<br>Gold Medals: 1<br>Silver Medals: 0<br>Bronze Medals: 1","UAE<br>Total Medals: 2<br>Gold Medals: 1<br>Silver Medals: 0<br>Bronze Medals: 1","JOR<br>Total Medals: 1<br>Gold Medals: 1<br>Silver Medals: 0<br>Bronze Medals: 0"],"hoverinfo":"text","reversescale":true,"colorscale":"Blues","marker":{"line":{"color":"rgb(180,180,180)","width":0.5}},"colorbar":{"autotick":false,"tickprefix":"","title":"Gold"}}],                        {"colorscale":"Blues","width":900,"height":600,"title":"GOLD Medallists' NOCs","geo":{"showframe":false,"showcoastlines":false,"projection":{"type":"Mercator"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('d83da5e5-7e5b-4378-b42f-aba9da5bc298');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


### 3.17.2 Silver Medals based on Countries


```python
df=medalsDF[medalsDF.Silver>0]
###CODE###
data = [dict(
    type='choropleth',
    locations=df['ISO'],
    z=df['Silver'],
    color_continuous_scale=px.colors.sequential.Plasma,
    text=df.apply(lambda row: f"{row['NOC']}<br>Total Medals: {row['Total']}<br>Gold Medals: {row['Gold']}<br>Silver Medals: {row['Silver']}<br>Bronze Medals: {row['Bronze']}", axis=1),
    hoverinfo="text",
    #autocolorscale=False,
    reversescale=True,
    colorscale = 'Greens',
    marker=dict(
        line=dict(
            color='rgb(180,180,180)',
            width=0.5
        )),
    colorbar=dict(
        autotick=False,
        tickprefix='',
        title='Silver'),
)]

layout = dict(
    colorscale = 'Blues',
    width=900,
    height=600,
    title="SILVER Medallists' NOCs",
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection=dict(
            type='Mercator'
        )
    )
)

fig = dict(data=data, layout=layout)

iplot(fig,validate=False)

```


<div>                            <div id="c6ec55f7-f7cf-42e9-98ed-69c9932e260e" class="plotly-graph-div" style="height:600px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("c6ec55f7-f7cf-42e9-98ed-69c9932e260e")) {                    Plotly.newPlot(                        "c6ec55f7-f7cf-42e9-98ed-69c9932e260e",                        [{"type":"choropleth","locations":["USA","GBR","DEU","ITA","FRA","HUN","SWE","AUS","CHN","RUS","NLD","JPN","NOR","DNK","KOR","CUB","ROU","CAN","DEU","FIN","IND","POL","ESP","BRA","CHE","BEL","ARG","NZL","GRC","HRV","BGR","UKR","PAK","TUR","JAM","KEN","ZAF","URY","MEX","AUT","NGA","ETH","CMR","KAZ","BLR","IRN","ZWE","PRK","SRB","CZE","BHS","SVK","IDN","IRL","EST","THA","UZB","GEO","AZE","TTO","EGY","SVN","LTU","MAR","COL","DZA","PRT","LUX","TWN","LVA","CHL","TUN","DOM","MNG","ARM","VEN","UGA","PER","SGP","VNM","PRI","HKG","ISR","HTI","CRI","TJK","BHR","CIV","SYR","BDI","ECU","GRD","PRY","ISL","MNE","MYS","NAM","PHL","MDA","LBN","LKA","TZA","GHA","SAU","QAT","KGZ","NER","ZMB","BES","BWA","CYP","GAB","GTM","VIR","SEN","SDN","TON"],"z":[1333,729,538,474,575,328,396,452,317,278,302,287,196,236,206,129,200,239,172,125,19,185,243,175,178,193,92,55,109,48,142,51,45,27,75,41,47,2,26,88,30,9,1,22,40,21,4,15,29,24,11,17,17,13,10,8,7,6,12,8,8,9,7,5,9,4,11,2,28,11,9,3,2,10,5,3,3,14,4,3,2,2,1,1,1,1,1,1,1,1,1,1,17,15,14,11,4,3,3,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],"color_continuous_scale":["#0d0887","#46039f","#7201a8","#9c179e","#bd3786","#d8576b","#ed7953","#fb9f3a","#fdca26","#f0f921"],"text":["USA<br>Total Medals: 5002<br>Gold Medals: 2472<br>Silver Medals: 1333<br>Bronze Medals: 1197","GBR<br>Total Medals: 1985<br>Gold Medals: 636<br>Silver Medals: 729<br>Bronze Medals: 620","GER<br>Total Medals: 1779<br>Gold Medals: 592<br>Silver Medals: 538<br>Bronze Medals: 649","ITA<br>Total Medals: 1446<br>Gold Medals: 518<br>Silver Medals: 474<br>Bronze Medals: 454","FRA<br>Total Medals: 1627<br>Gold Medals: 465<br>Silver Medals: 575<br>Bronze Medals: 587","HUN<br>Total Medals: 1123<br>Gold Medals: 432<br>Silver Medals: 328<br>Bronze Medals: 363","SWE<br>Total Medals: 1108<br>Gold Medals: 354<br>Silver Medals: 396<br>Bronze Medals: 358","AUS<br>Total Medals: 1304<br>Gold Medals: 342<br>Silver Medals: 452<br>Bronze Medals: 510","CHN<br>Total Medals: 909<br>Gold Medals: 334<br>Silver Medals: 317<br>Bronze Medals: 258","RUS<br>Total Medals: 905<br>Gold Medals: 296<br>Silver Medals: 278<br>Bronze Medals: 331","NED<br>Total Medals: 918<br>Gold Medals: 245<br>Silver Medals: 302<br>Bronze Medals: 371","JPN<br>Total Medals: 850<br>Gold Medals: 230<br>Silver Medals: 287<br>Bronze Medals: 333","NOR<br>Total Medals: 590<br>Gold Medals: 227<br>Silver Medals: 196<br>Bronze Medals: 167","DEN<br>Total Medals: 592<br>Gold Medals: 179<br>Silver Medals: 236<br>Bronze Medals: 177","KOR<br>Total Medals: 552<br>Gold Medals: 171<br>Silver Medals: 206<br>Bronze Medals: 175","CUB<br>Total Medals: 409<br>Gold Medals: 164<br>Silver Medals: 129<br>Bronze Medals: 116","ROU<br>Total Medals: 651<br>Gold Medals: 161<br>Silver Medals: 200<br>Bronze Medals: 290","CAN<br>Total Medals: 741<br>Gold Medals: 158<br>Silver Medals: 239<br>Bronze Medals: 344","FRG<br>Total Medals: 504<br>Gold Medals: 144<br>Silver Medals: 172<br>Bronze Medals: 188","FIN<br>Total Medals: 474<br>Gold Medals: 132<br>Silver Medals: 125<br>Bronze Medals: 217","IND<br>Total Medals: 190<br>Gold Medals: 131<br>Silver Medals: 19<br>Bronze Medals: 40","POL<br>Total Medals: 538<br>Gold Medals: 111<br>Silver Medals: 185<br>Bronze Medals: 242","ESP<br>Total Medals: 487<br>Gold Medals: 109<br>Silver Medals: 243<br>Bronze Medals: 135","BRA<br>Total Medals: 475<br>Gold Medals: 109<br>Silver Medals: 175<br>Bronze Medals: 191","SUI<br>Total Medals: 416<br>Gold Medals: 99<br>Silver Medals: 178<br>Bronze Medals: 139","BEL<br>Total Medals: 455<br>Gold Medals: 96<br>Silver Medals: 193<br>Bronze Medals: 166","ARG<br>Total Medals: 274<br>Gold Medals: 91<br>Silver Medals: 92<br>Bronze Medals: 91","NZL<br>Total Medals: 227<br>Gold Medals: 90<br>Silver Medals: 55<br>Bronze Medals: 82","GRE<br>Total Medals: 255<br>Gold Medals: 62<br>Silver Medals: 109<br>Bronze Medals: 84","CRO<br>Total Medals: 138<br>Gold Medals: 54<br>Silver Medals: 48<br>Bronze Medals: 36","BUL<br>Total Medals: 336<br>Gold Medals: 53<br>Silver Medals: 142<br>Bronze Medals: 141","UKR<br>Total Medals: 188<br>Gold Medals: 42<br>Silver Medals: 51<br>Bronze Medals: 95","PAK<br>Total Medals: 121<br>Gold Medals: 42<br>Silver Medals: 45<br>Bronze Medals: 34","TUR<br>Total Medals: 95<br>Gold Medals: 40<br>Silver Medals: 27<br>Bronze Medals: 28","JAM<br>Total Medals: 157<br>Gold Medals: 38<br>Silver Medals: 75<br>Bronze Medals: 44","KEN<br>Total Medals: 106<br>Gold Medals: 34<br>Silver Medals: 41<br>Bronze Medals: 31","RSA<br>Total Medals: 131<br>Gold Medals: 32<br>Silver Medals: 47<br>Bronze Medals: 52","URU<br>Total Medals: 63<br>Gold Medals: 31<br>Silver Medals: 2<br>Bronze Medals: 30","MEX<br>Total Medals: 110<br>Gold Medals: 30<br>Silver Medals: 26<br>Bronze Medals: 54","AUT<br>Total Medals: 170<br>Gold Medals: 29<br>Silver Medals: 88<br>Bronze Medals: 53","NGR<br>Total Medals: 99<br>Gold Medals: 23<br>Silver Medals: 30<br>Bronze Medals: 46","ETH<br>Total Medals: 53<br>Gold Medals: 22<br>Silver Medals: 9<br>Bronze Medals: 22","CMR<br>Total Medals: 22<br>Gold Medals: 20<br>Silver Medals: 1<br>Bronze Medals: 1","KAZ<br>Total Medals: 70<br>Gold Medals: 19<br>Silver Medals: 22<br>Bronze Medals: 29","BLR<br>Total Medals: 124<br>Gold Medals: 18<br>Silver Medals: 40<br>Bronze Medals: 66","IRI<br>Total Medals: 68<br>Gold Medals: 18<br>Silver Medals: 21<br>Bronze Medals: 29","ZIM<br>Total Medals: 22<br>Gold Medals: 17<br>Silver Medals: 4<br>Bronze Medals: 1","PRK<br>Total Medals: 65<br>Gold Medals: 16<br>Silver Medals: 15<br>Bronze Medals: 34","SRB<br>Total Medals: 85<br>Gold Medals: 15<br>Silver Medals: 29<br>Bronze Medals: 41","CZE<br>Total Medals: 71<br>Gold Medals: 15<br>Silver Medals: 24<br>Bronze Medals: 32","BAH<br>Total Medals: 40<br>Gold Medals: 14<br>Silver Medals: 11<br>Bronze Medals: 15","SVK<br>Total Medals: 42<br>Gold Medals: 13<br>Silver Medals: 17<br>Bronze Medals: 12","INA<br>Total Medals: 41<br>Gold Medals: 11<br>Silver Medals: 17<br>Bronze Medals: 13","IRL<br>Total Medals: 35<br>Gold Medals: 9<br>Silver Medals: 13<br>Bronze Medals: 13","EST<br>Total Medals: 43<br>Gold Medals: 9<br>Silver Medals: 10<br>Bronze Medals: 24","THA<br>Total Medals: 30<br>Gold Medals: 9<br>Silver Medals: 8<br>Bronze Medals: 13","UZB<br>Total Medals: 33<br>Gold Medals: 9<br>Silver Medals: 7<br>Bronze Medals: 17","GEO<br>Total Medals: 32<br>Gold Medals: 8<br>Silver Medals: 6<br>Bronze Medals: 18","AZE<br>Total Medals: 44<br>Gold Medals: 7<br>Silver Medals: 12<br>Bronze Medals: 25","TTO<br>Total Medals: 32<br>Gold Medals: 7<br>Silver Medals: 8<br>Bronze Medals: 17","EGY<br>Total Medals: 27<br>Gold Medals: 7<br>Silver Medals: 8<br>Bronze Medals: 12","SLO<br>Total Medals: 30<br>Gold Medals: 6<br>Silver Medals: 9<br>Bronze Medals: 15","LTU<br>Total Medals: 61<br>Gold Medals: 6<br>Silver Medals: 7<br>Bronze Medals: 48","MAR<br>Total Medals: 23<br>Gold Medals: 6<br>Silver Medals: 5<br>Bronze Medals: 12","COL<br>Total Medals: 28<br>Gold Medals: 5<br>Silver Medals: 9<br>Bronze Medals: 14","ALG<br>Total Medals: 17<br>Gold Medals: 5<br>Silver Medals: 4<br>Bronze Medals: 8","POR<br>Total Medals: 41<br>Gold Medals: 4<br>Silver Medals: 11<br>Bronze Medals: 26","LUX<br>Total Medals: 6<br>Gold Medals: 4<br>Silver Medals: 2<br>Bronze Medals: 0","TPE<br>Total Medals: 49<br>Gold Medals: 3<br>Silver Medals: 28<br>Bronze Medals: 18","LAT<br>Total Medals: 20<br>Gold Medals: 3<br>Silver Medals: 11<br>Bronze Medals: 6","CHI<br>Total Medals: 32<br>Gold Medals: 3<br>Silver Medals: 9<br>Bronze Medals: 20","TUN<br>Total Medals: 13<br>Gold Medals: 3<br>Silver Medals: 3<br>Bronze Medals: 7","DOM<br>Total Medals: 7<br>Gold Medals: 3<br>Silver Medals: 2<br>Bronze Medals: 2","MGL<br>Total Medals: 26<br>Gold Medals: 2<br>Silver Medals: 10<br>Bronze Medals: 14","ARM<br>Total Medals: 16<br>Gold Medals: 2<br>Silver Medals: 5<br>Bronze Medals: 9","VEN<br>Total Medals: 15<br>Gold Medals: 2<br>Silver Medals: 3<br>Bronze Medals: 10","UGA<br>Total Medals: 7<br>Gold Medals: 2<br>Silver Medals: 3<br>Bronze Medals: 2","PER<br>Total Medals: 15<br>Gold Medals: 1<br>Silver Medals: 14<br>Bronze Medals: 0","SGP<br>Total Medals: 9<br>Gold Medals: 1<br>Silver Medals: 4<br>Bronze Medals: 4","VIE<br>Total Medals: 4<br>Gold Medals: 1<br>Silver Medals: 3<br>Bronze Medals: 0","PUR<br>Total Medals: 9<br>Gold Medals: 1<br>Silver Medals: 2<br>Bronze Medals: 6","HKG<br>Total Medals: 4<br>Gold Medals: 1<br>Silver Medals: 2<br>Bronze Medals: 1","ISR<br>Total Medals: 9<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 7","HAI<br>Total Medals: 7<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 5","CRC<br>Total Medals: 4<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 2","TJK<br>Total Medals: 4<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 2","BRN<br>Total Medals: 3<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 1","CIV<br>Total Medals: 3<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 1","SYR<br>Total Medals: 3<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 1","BDI<br>Total Medals: 2<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 0","ECU<br>Total Medals: 2<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 0","GRN<br>Total Medals: 2<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 0","PAR<br>Total Medals: 17<br>Gold Medals: 0<br>Silver Medals: 17<br>Bronze Medals: 0","ISL<br>Total Medals: 17<br>Gold Medals: 0<br>Silver Medals: 15<br>Bronze Medals: 2","MNE<br>Total Medals: 14<br>Gold Medals: 0<br>Silver Medals: 14<br>Bronze Medals: 0","MAS<br>Total Medals: 16<br>Gold Medals: 0<br>Silver Medals: 11<br>Bronze Medals: 5","NAM<br>Total Medals: 4<br>Gold Medals: 0<br>Silver Medals: 4<br>Bronze Medals: 0","PHI<br>Total Medals: 10<br>Gold Medals: 0<br>Silver Medals: 3<br>Bronze Medals: 7","MDA<br>Total Medals: 8<br>Gold Medals: 0<br>Silver Medals: 3<br>Bronze Medals: 5","LIB<br>Total Medals: 4<br>Gold Medals: 0<br>Silver Medals: 2<br>Bronze Medals: 2","SRI<br>Total Medals: 2<br>Gold Medals: 0<br>Silver Medals: 2<br>Bronze Medals: 0","TAN<br>Total Medals: 2<br>Gold Medals: 0<br>Silver Medals: 2<br>Bronze Medals: 0","GHA<br>Total Medals: 23<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 22","KSA<br>Total Medals: 6<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 5","QAT<br>Total Medals: 5<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 4","KGZ<br>Total Medals: 3<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 2","NIG<br>Total Medals: 2<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 1","ZAM<br>Total Medals: 2<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 1","AHO<br>Total Medals: 1<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 0","BOT<br>Total Medals: 1<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 0","CYP<br>Total Medals: 1<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 0","GAB<br>Total Medals: 1<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 0","GUA<br>Total Medals: 1<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 0","ISV<br>Total Medals: 1<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 0","SEN<br>Total Medals: 1<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 0","SUD<br>Total Medals: 1<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 0","TGA<br>Total Medals: 1<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 0"],"hoverinfo":"text","reversescale":true,"colorscale":"Greens","marker":{"line":{"color":"rgb(180,180,180)","width":0.5}},"colorbar":{"autotick":false,"tickprefix":"","title":"Silver"}}],                        {"colorscale":"Blues","width":900,"height":600,"title":"SILVER Medallists' NOCs","geo":{"showframe":false,"showcoastlines":false,"projection":{"type":"Mercator"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('c6ec55f7-f7cf-42e9-98ed-69c9932e260e');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


### 3.17.3 Bronze Medals based on Countries



```python
df=medalsDF[medalsDF.Bronze>0]
###CODE###
data = [dict(
    type='choropleth',
    locations=df['ISO'],
    z=df['Bronze'],
    #color_continuous_scale=px.colors.sequential.Plasma,
    text=df.apply(lambda row: f"{row['NOC']}<br>Total Medals: {row['Total']}<br>Gold Medals: {row['Gold']}<br>Silver Medals: {row['Silver']}<br>Bronze Medals: {row['Bronze']}", axis=1),
    hoverinfo="text",
    #autocolorscale=False,
    #reversescale=True,
    colorscale = 'Yellows',
    marker=dict(
        line=dict(
            color='rgb(180,180,180)',
            width=0.5
        )),
    colorbar=dict(
        autotick=False,
        tickprefix='',
        title='Bronze'),
)]

layout = dict(
    colorscale = 'Gold',
    width=900,
    height=600,
    title="BRONZE Medallists' NOCs",
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection=dict(
            type='Mercator'
        )
    )
)

fig = dict(data=data, layout=layout)

iplot(fig,validate=False)

```


<div>                            <div id="95004dd1-60cb-4588-9861-179b40e546bd" class="plotly-graph-div" style="height:600px; width:900px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("95004dd1-60cb-4588-9861-179b40e546bd")) {                    Plotly.newPlot(                        "95004dd1-60cb-4588-9861-179b40e546bd",                        [{"type":"choropleth","locations":["USA","GBR","DEU","ITA","FRA","HUN","SWE","AUS","CHN","RUS","NLD","JPN","NOR","DNK","KOR","CUB","ROU","CAN","DEU","FIN","IND","POL","ESP","BRA","CHE","BEL","ARG","NZL","GRC","HRV","BGR","UKR","PAK","TUR","JAM","KEN","ZAF","URY","MEX","AUT","NGA","ETH","CMR","KAZ","BLR","IRN","ZWE","PRK","SRB","CZE","BHS","SVK","IDN","IRL","EST","THA","UZB","GEO","AZE","TTO","EGY","SVN","LTU","MAR","COL","DZA","PRT","TWN","LVA","CHL","TUN","DOM","MNG","ARM","VEN","UGA","SGP","PRI","HKG","ISR","HTI","CRI","TJK","BHR","CIV","SYR","PAN","MOZ","SUR","ARE","ISL","MYS","PHL","MDA","LBN","GHA","SAU","QAT","KGZ","NER","ZMB","AFG","KWT","BRB","BMU","DJI","ERI","GUY","IRQ","MKD","MCO","MUS","TGO"],"z":[1197,620,649,454,587,363,358,510,258,331,371,333,167,177,175,116,290,344,188,217,40,242,135,191,139,166,91,82,84,36,141,95,34,28,44,31,52,30,54,53,46,22,1,29,66,29,1,34,41,32,15,12,13,13,24,13,17,18,25,17,12,15,48,12,14,8,26,18,6,20,7,2,14,9,10,2,4,6,1,7,5,2,2,1,1,1,2,1,1,1,2,5,7,5,2,22,5,4,2,1,1,2,2,1,1,1,1,1,1,1,1,1,1],"text":["USA<br>Total Medals: 5002<br>Gold Medals: 2472<br>Silver Medals: 1333<br>Bronze Medals: 1197","GBR<br>Total Medals: 1985<br>Gold Medals: 636<br>Silver Medals: 729<br>Bronze Medals: 620","GER<br>Total Medals: 1779<br>Gold Medals: 592<br>Silver Medals: 538<br>Bronze Medals: 649","ITA<br>Total Medals: 1446<br>Gold Medals: 518<br>Silver Medals: 474<br>Bronze Medals: 454","FRA<br>Total Medals: 1627<br>Gold Medals: 465<br>Silver Medals: 575<br>Bronze Medals: 587","HUN<br>Total Medals: 1123<br>Gold Medals: 432<br>Silver Medals: 328<br>Bronze Medals: 363","SWE<br>Total Medals: 1108<br>Gold Medals: 354<br>Silver Medals: 396<br>Bronze Medals: 358","AUS<br>Total Medals: 1304<br>Gold Medals: 342<br>Silver Medals: 452<br>Bronze Medals: 510","CHN<br>Total Medals: 909<br>Gold Medals: 334<br>Silver Medals: 317<br>Bronze Medals: 258","RUS<br>Total Medals: 905<br>Gold Medals: 296<br>Silver Medals: 278<br>Bronze Medals: 331","NED<br>Total Medals: 918<br>Gold Medals: 245<br>Silver Medals: 302<br>Bronze Medals: 371","JPN<br>Total Medals: 850<br>Gold Medals: 230<br>Silver Medals: 287<br>Bronze Medals: 333","NOR<br>Total Medals: 590<br>Gold Medals: 227<br>Silver Medals: 196<br>Bronze Medals: 167","DEN<br>Total Medals: 592<br>Gold Medals: 179<br>Silver Medals: 236<br>Bronze Medals: 177","KOR<br>Total Medals: 552<br>Gold Medals: 171<br>Silver Medals: 206<br>Bronze Medals: 175","CUB<br>Total Medals: 409<br>Gold Medals: 164<br>Silver Medals: 129<br>Bronze Medals: 116","ROU<br>Total Medals: 651<br>Gold Medals: 161<br>Silver Medals: 200<br>Bronze Medals: 290","CAN<br>Total Medals: 741<br>Gold Medals: 158<br>Silver Medals: 239<br>Bronze Medals: 344","FRG<br>Total Medals: 504<br>Gold Medals: 144<br>Silver Medals: 172<br>Bronze Medals: 188","FIN<br>Total Medals: 474<br>Gold Medals: 132<br>Silver Medals: 125<br>Bronze Medals: 217","IND<br>Total Medals: 190<br>Gold Medals: 131<br>Silver Medals: 19<br>Bronze Medals: 40","POL<br>Total Medals: 538<br>Gold Medals: 111<br>Silver Medals: 185<br>Bronze Medals: 242","ESP<br>Total Medals: 487<br>Gold Medals: 109<br>Silver Medals: 243<br>Bronze Medals: 135","BRA<br>Total Medals: 475<br>Gold Medals: 109<br>Silver Medals: 175<br>Bronze Medals: 191","SUI<br>Total Medals: 416<br>Gold Medals: 99<br>Silver Medals: 178<br>Bronze Medals: 139","BEL<br>Total Medals: 455<br>Gold Medals: 96<br>Silver Medals: 193<br>Bronze Medals: 166","ARG<br>Total Medals: 274<br>Gold Medals: 91<br>Silver Medals: 92<br>Bronze Medals: 91","NZL<br>Total Medals: 227<br>Gold Medals: 90<br>Silver Medals: 55<br>Bronze Medals: 82","GRE<br>Total Medals: 255<br>Gold Medals: 62<br>Silver Medals: 109<br>Bronze Medals: 84","CRO<br>Total Medals: 138<br>Gold Medals: 54<br>Silver Medals: 48<br>Bronze Medals: 36","BUL<br>Total Medals: 336<br>Gold Medals: 53<br>Silver Medals: 142<br>Bronze Medals: 141","UKR<br>Total Medals: 188<br>Gold Medals: 42<br>Silver Medals: 51<br>Bronze Medals: 95","PAK<br>Total Medals: 121<br>Gold Medals: 42<br>Silver Medals: 45<br>Bronze Medals: 34","TUR<br>Total Medals: 95<br>Gold Medals: 40<br>Silver Medals: 27<br>Bronze Medals: 28","JAM<br>Total Medals: 157<br>Gold Medals: 38<br>Silver Medals: 75<br>Bronze Medals: 44","KEN<br>Total Medals: 106<br>Gold Medals: 34<br>Silver Medals: 41<br>Bronze Medals: 31","RSA<br>Total Medals: 131<br>Gold Medals: 32<br>Silver Medals: 47<br>Bronze Medals: 52","URU<br>Total Medals: 63<br>Gold Medals: 31<br>Silver Medals: 2<br>Bronze Medals: 30","MEX<br>Total Medals: 110<br>Gold Medals: 30<br>Silver Medals: 26<br>Bronze Medals: 54","AUT<br>Total Medals: 170<br>Gold Medals: 29<br>Silver Medals: 88<br>Bronze Medals: 53","NGR<br>Total Medals: 99<br>Gold Medals: 23<br>Silver Medals: 30<br>Bronze Medals: 46","ETH<br>Total Medals: 53<br>Gold Medals: 22<br>Silver Medals: 9<br>Bronze Medals: 22","CMR<br>Total Medals: 22<br>Gold Medals: 20<br>Silver Medals: 1<br>Bronze Medals: 1","KAZ<br>Total Medals: 70<br>Gold Medals: 19<br>Silver Medals: 22<br>Bronze Medals: 29","BLR<br>Total Medals: 124<br>Gold Medals: 18<br>Silver Medals: 40<br>Bronze Medals: 66","IRI<br>Total Medals: 68<br>Gold Medals: 18<br>Silver Medals: 21<br>Bronze Medals: 29","ZIM<br>Total Medals: 22<br>Gold Medals: 17<br>Silver Medals: 4<br>Bronze Medals: 1","PRK<br>Total Medals: 65<br>Gold Medals: 16<br>Silver Medals: 15<br>Bronze Medals: 34","SRB<br>Total Medals: 85<br>Gold Medals: 15<br>Silver Medals: 29<br>Bronze Medals: 41","CZE<br>Total Medals: 71<br>Gold Medals: 15<br>Silver Medals: 24<br>Bronze Medals: 32","BAH<br>Total Medals: 40<br>Gold Medals: 14<br>Silver Medals: 11<br>Bronze Medals: 15","SVK<br>Total Medals: 42<br>Gold Medals: 13<br>Silver Medals: 17<br>Bronze Medals: 12","INA<br>Total Medals: 41<br>Gold Medals: 11<br>Silver Medals: 17<br>Bronze Medals: 13","IRL<br>Total Medals: 35<br>Gold Medals: 9<br>Silver Medals: 13<br>Bronze Medals: 13","EST<br>Total Medals: 43<br>Gold Medals: 9<br>Silver Medals: 10<br>Bronze Medals: 24","THA<br>Total Medals: 30<br>Gold Medals: 9<br>Silver Medals: 8<br>Bronze Medals: 13","UZB<br>Total Medals: 33<br>Gold Medals: 9<br>Silver Medals: 7<br>Bronze Medals: 17","GEO<br>Total Medals: 32<br>Gold Medals: 8<br>Silver Medals: 6<br>Bronze Medals: 18","AZE<br>Total Medals: 44<br>Gold Medals: 7<br>Silver Medals: 12<br>Bronze Medals: 25","TTO<br>Total Medals: 32<br>Gold Medals: 7<br>Silver Medals: 8<br>Bronze Medals: 17","EGY<br>Total Medals: 27<br>Gold Medals: 7<br>Silver Medals: 8<br>Bronze Medals: 12","SLO<br>Total Medals: 30<br>Gold Medals: 6<br>Silver Medals: 9<br>Bronze Medals: 15","LTU<br>Total Medals: 61<br>Gold Medals: 6<br>Silver Medals: 7<br>Bronze Medals: 48","MAR<br>Total Medals: 23<br>Gold Medals: 6<br>Silver Medals: 5<br>Bronze Medals: 12","COL<br>Total Medals: 28<br>Gold Medals: 5<br>Silver Medals: 9<br>Bronze Medals: 14","ALG<br>Total Medals: 17<br>Gold Medals: 5<br>Silver Medals: 4<br>Bronze Medals: 8","POR<br>Total Medals: 41<br>Gold Medals: 4<br>Silver Medals: 11<br>Bronze Medals: 26","TPE<br>Total Medals: 49<br>Gold Medals: 3<br>Silver Medals: 28<br>Bronze Medals: 18","LAT<br>Total Medals: 20<br>Gold Medals: 3<br>Silver Medals: 11<br>Bronze Medals: 6","CHI<br>Total Medals: 32<br>Gold Medals: 3<br>Silver Medals: 9<br>Bronze Medals: 20","TUN<br>Total Medals: 13<br>Gold Medals: 3<br>Silver Medals: 3<br>Bronze Medals: 7","DOM<br>Total Medals: 7<br>Gold Medals: 3<br>Silver Medals: 2<br>Bronze Medals: 2","MGL<br>Total Medals: 26<br>Gold Medals: 2<br>Silver Medals: 10<br>Bronze Medals: 14","ARM<br>Total Medals: 16<br>Gold Medals: 2<br>Silver Medals: 5<br>Bronze Medals: 9","VEN<br>Total Medals: 15<br>Gold Medals: 2<br>Silver Medals: 3<br>Bronze Medals: 10","UGA<br>Total Medals: 7<br>Gold Medals: 2<br>Silver Medals: 3<br>Bronze Medals: 2","SGP<br>Total Medals: 9<br>Gold Medals: 1<br>Silver Medals: 4<br>Bronze Medals: 4","PUR<br>Total Medals: 9<br>Gold Medals: 1<br>Silver Medals: 2<br>Bronze Medals: 6","HKG<br>Total Medals: 4<br>Gold Medals: 1<br>Silver Medals: 2<br>Bronze Medals: 1","ISR<br>Total Medals: 9<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 7","HAI<br>Total Medals: 7<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 5","CRC<br>Total Medals: 4<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 2","TJK<br>Total Medals: 4<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 2","BRN<br>Total Medals: 3<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 1","CIV<br>Total Medals: 3<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 1","SYR<br>Total Medals: 3<br>Gold Medals: 1<br>Silver Medals: 1<br>Bronze Medals: 1","PAN<br>Total Medals: 3<br>Gold Medals: 1<br>Silver Medals: 0<br>Bronze Medals: 2","MOZ<br>Total Medals: 2<br>Gold Medals: 1<br>Silver Medals: 0<br>Bronze Medals: 1","SUR<br>Total Medals: 2<br>Gold Medals: 1<br>Silver Medals: 0<br>Bronze Medals: 1","UAE<br>Total Medals: 2<br>Gold Medals: 1<br>Silver Medals: 0<br>Bronze Medals: 1","ISL<br>Total Medals: 17<br>Gold Medals: 0<br>Silver Medals: 15<br>Bronze Medals: 2","MAS<br>Total Medals: 16<br>Gold Medals: 0<br>Silver Medals: 11<br>Bronze Medals: 5","PHI<br>Total Medals: 10<br>Gold Medals: 0<br>Silver Medals: 3<br>Bronze Medals: 7","MDA<br>Total Medals: 8<br>Gold Medals: 0<br>Silver Medals: 3<br>Bronze Medals: 5","LIB<br>Total Medals: 4<br>Gold Medals: 0<br>Silver Medals: 2<br>Bronze Medals: 2","GHA<br>Total Medals: 23<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 22","KSA<br>Total Medals: 6<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 5","QAT<br>Total Medals: 5<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 4","KGZ<br>Total Medals: 3<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 2","NIG<br>Total Medals: 2<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 1","ZAM<br>Total Medals: 2<br>Gold Medals: 0<br>Silver Medals: 1<br>Bronze Medals: 1","AFG<br>Total Medals: 2<br>Gold Medals: 0<br>Silver Medals: 0<br>Bronze Medals: 2","KUW<br>Total Medals: 2<br>Gold Medals: 0<br>Silver Medals: 0<br>Bronze Medals: 2","BAR<br>Total Medals: 1<br>Gold Medals: 0<br>Silver Medals: 0<br>Bronze Medals: 1","BER<br>Total Medals: 1<br>Gold Medals: 0<br>Silver Medals: 0<br>Bronze Medals: 1","DJI<br>Total Medals: 1<br>Gold Medals: 0<br>Silver Medals: 0<br>Bronze Medals: 1","ERI<br>Total Medals: 1<br>Gold Medals: 0<br>Silver Medals: 0<br>Bronze Medals: 1","GUY<br>Total Medals: 1<br>Gold Medals: 0<br>Silver Medals: 0<br>Bronze Medals: 1","IRQ<br>Total Medals: 1<br>Gold Medals: 0<br>Silver Medals: 0<br>Bronze Medals: 1","MKD<br>Total Medals: 1<br>Gold Medals: 0<br>Silver Medals: 0<br>Bronze Medals: 1","MON<br>Total Medals: 1<br>Gold Medals: 0<br>Silver Medals: 0<br>Bronze Medals: 1","MRI<br>Total Medals: 1<br>Gold Medals: 0<br>Silver Medals: 0<br>Bronze Medals: 1","TOG<br>Total Medals: 1<br>Gold Medals: 0<br>Silver Medals: 0<br>Bronze Medals: 1"],"hoverinfo":"text","colorscale":"Yellows","marker":{"line":{"color":"rgb(180,180,180)","width":0.5}},"colorbar":{"autotick":false,"tickprefix":"","title":"Bronze"}}],                        {"colorscale":"Gold","width":900,"height":600,"title":"BRONZE Medallists' NOCs","geo":{"showframe":false,"showcoastlines":false,"projection":{"type":"Mercator"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('95004dd1-60cb-4588-9861-179b40e546bd');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


# 4. Summary
We found some interesting hindsight. Why hindsight? Because they were there, maybe we just never paid attention.

- Some interesting sports were part of the Olympic program
- The Olympic Games were always held every 4 years, with exception of the World Wars and the COVID pandemic. However, we find an Olympic Games in 1906, while there were also Olympic Games in 1904 and 1908. These were called 'The 1906 Intercalated Games' . These Games were held by the IOC, but later they were unrecognized.
- The oldest Olympian ever, John Quincy Adams Ward, participated in the 1928 Olympics post-mortem. In reality, his art participated but not him personally.
- Most of the Olympians are between 21 and 29 years old.
- We can see that the Gold Medalist's Age concentration is mainly between 19 and 26 years.
- The men athletes' population is getting taller.


Did you like it? What questions would you do to this dataset?

Thanks for reading!
