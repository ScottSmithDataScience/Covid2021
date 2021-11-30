# Importing Data
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import ttest_rel, f_oneway

# For Post Hoc Analysis
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison as Multi

# For Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('tripsnationalupdated.csv', encoding='utf-8')

# filtering and removing unnecessary columns and creating new columns for use
df = df[df['Level']=='National']
df.dropna(axis=1, how='any',inplace=True)
date_format = '%Y/%m/%d'
df['Date'] = [datetime.strptime(date, date_format) for date in df['Date']]
df['Year'] = [date.year for date in  df['Date']]
df['Month']  = [date.month for date in df['Date']]
df['Day'] = [date.day for date in df['Date']]

# creating dates for thanksgiving in the past 3 years
tk_2019_date = datetime.strptime('2019/11/28', date_format)
tk_2020_date = datetime.strptime('2020/11/26', date_format)
tk_2021_date = datetime.strptime('2021/11/25',date_format)

# separating data into years
data_2019 = df[df['Year']==2019]
data_2020 = df[df['Year']==2020]
data_2021 = df[df['Year']==2021]

# filtering to specifically only week before thanksgiving
begin_2019 = tk_2019_date - timedelta(days=19)
end_2019 = tk_2019_date - timedelta(days=5)
data_2019 = data_2019[(data_2019['Date']> begin_2019) & (data_2019['Date']<= end_2019)]

begin_2020 = tk_2020_date - timedelta(days=19)
end_2020 = tk_2020_date - timedelta(days=5)
data_2020 = data_2020[(data_2020['Date']> begin_2020) & (data_2020['Date']<= end_2020)]

begin_2021 = tk_2021_date - timedelta(days=19)
end_2021 = tk_2021_date - timedelta(days=5)
data_2021 = data_2021[(data_2021['Date']> begin_2021) & (data_2021['Date'] <= end_2021)]

# weighing different years
"""
population in 2019 was 331,028,075 (U.S. Census Bureau)
population in 2020 was 332,013,802 (U.S. Census Bureau)
population in 2021 is 332,952,379 (U.S. Census Bureau)
"""
weight_2019 = 1.0
weight_2020 = round(332013802/331028075, 3)
weight_2021 = round(332952379/331028075, 3)


#Creating Arrays for total trips each year
trips2019 = np.array(data_2019['Number of Trips'])
trips2020 = np.array(data_2020['Number of Trips'])
trips2021 = np.array(data_2021['Number of Trips'])

#Creating Arrays for total trips each year group >= 5 & < 50 miles

Med_trips_2019 = []
Med_trips_2020 = []
Med_trips_2021 = []

dframe = [data_2019,data_2020,data_2021]
appendto = [Med_trips_2019,Med_trips_2020,Med_trips_2021]
weight = [weight_2019,weight_2020,weight_2021]
for d,a,w in zip(dframe,appendto,weight):
    for i,j,k,l in zip(d['Number of Trips 5-10'],d['Number of Trips 10-25'],d['Number of Trips 25-50'],d['Number of Trips 50-100']):
        sum_of_Med = (i + j + k + l) / w
        a.append(sum_of_Med)

#Creating Arrays for total trips each year group >= 100

Long_trips_2019 = []
Long_trips_2020 = []
Long_trips_2021 = []


appendto = [Long_trips_2019,Long_trips_2020,Long_trips_2021]
for d,a,w in zip(dframe,appendto,weight):
    for i,j,k in zip(d['Number of Trips 100-250'],d['Number of Trips 250-500'],d['Number of Trips >=500']):
        sum_of_Long = (i + j + k) / w
        a.append(sum_of_Long)

# creating dict to store ttest values
ttest_vals = {}
#Comparing Total Trips
ttest_vals['Total_2019n2020'] = ttest_rel(trips2019,trips2020).pvalue
ttest_vals['Total_2020n2021'] = ttest_rel(trips2020,trips2021).pvalue
ttest_vals['Total_2019n2021'] = ttest_rel(trips2019,trips2021).pvalue

#Comparing Med Trips
ttest_vals['Med_2019n2020'] = ttest_rel(Med_trips_2019,Med_trips_2020).pvalue
ttest_vals['Med_2020n2021'] = ttest_rel(Med_trips_2020,Med_trips_2021).pvalue
ttest_vals['Med_2019n2021'] = ttest_rel(Med_trips_2019,Med_trips_2021).pvalue

#Comparing Long Trips
ttest_vals['Long_2019n2020'] = ttest_rel(Long_trips_2019,Long_trips_2020).pvalue
ttest_vals['Long_2020n2021'] = ttest_rel(Long_trips_2020,Long_trips_2021).pvalue
ttest_vals['Long_2019n2021'] = ttest_rel(Long_trips_2019,Long_trips_2021).pvalue

# creating dataframe for p-values
ttest_vals = pd.DataFrame(pd.Series(ttest_vals,name = 'p-value'))
ttest_vals['Significant?'] = ['Yes' if p <= 0.05 else 'No' for p in ttest_vals['p-value']]

# pvaluelist = [Total_2019n2020,Total_2019n2021,Total_2020n2021,Med_2019n2020,Med_2019n2021,Med_2020n2021,Long_2019n2020,Long_2019n2021,Long_2020n2021]
# for p in pvaluelist:
#     if p <= .05:
#         print(p)
#     else:
#         print('not significant')

anova_vals = {}
# Anova 3 way testing
anova_vals['Anova_result_total'] = f_oneway(trips2019, trips2020,trips2021).pvalue
anova_vals['Anova_result_Med'] = f_oneway(Med_trips_2019, Med_trips_2020,Med_trips_2021).pvalue
anova_vals['Anova_result_Long'] = f_oneway(Long_trips_2019, Long_trips_2020,Long_trips_2021).pvalue

anova_vals = pd.DataFrame(pd.Series(anova_vals, name='p-value'))
anova_vals['Significant?'] =  ['Yes' if p <= 0.05 else 'No' for p in anova_vals['p-value']]
# Since all three Anova_result_xxxxx are less than 0.05 i.e. there are differences. Doing a PostHoc analysis

MultiComp_Total = Multi(df["Number of Trips"], df ['Year'])
PostHoc_Total = MultiComp_Total.tukeyhsd()
print (PostHoc_Total)

MultiComp_Med = Multi(df["Number of Trips 25-50"] + df["Number of Trips 50-100"] + df["Number of Trips 10-25"] + df["Number of Trips 5-10"], df ['Year'])
PostHoc_Med = MultiComp_Med.tukeyhsd()
print (PostHoc_Med)

MultiComp_Long = Multi(df["Number of Trips 100-250"] + df["Number of Trips 250-500"] + df["Number of Trips >=500"] + df["Number of Trips 5-10"], df ['Year'])
PostHoc_Long = MultiComp_Long.tukeyhsd()
print (PostHoc_Long)




#Creating DataFrame of Corresponding Thanksgiving Weeks

df2019 = df[df['Date'] >= '2019-11-10'][df['Date'] <= '2019-11-28']
df2020 = df[df['Date'] >= '2020-11-08'][df['Date'] <= '2020-11-26']
df2021 = df[df['Date'] >= '2021-11-07'][df['Date'] <= '2021-11-20']
df2019['days_away'] = (28 - df2019['Day'])
df2020['days_away'] = (26 - df2020['Day'])
df2021['days_away'] = (25 - df2021['Day'])
df1 = df2019.merge(df2020,how='outer')
#Merged
df2 = df1.merge(df2021,how='outer')




df3 = df2[['Number of Trips','Year','days_away']]


#Creating Linear Regression, x train, y train, & test
x = df3[['Year','days_away']]
y = df3['Number of Trips']
Year = [2021,2021,2021,2021,2021]
days_away = [4,3,2,1,0]
zipped = zip(Year,days_away)
test = pd.DataFrame(zipped,columns = ['Year','days_away'])


#Creating model and testing for 2021/11/20 through 2021/11/25
model = LinearRegression()
linear = model.fit(x,y)
tested = linear.predict(test)

#creating DataFrame to include Predictions
y2021 = [2021,2021,2021,2021,2021]
days_away2021 = [4,3,2,1,0]
numoftrips2021 = []
for item in tested:
    numoftrips2021.append(round(float(item)))
zippy = zip(numoftrips2021,y2021,days_away2021)
df4add = pd.DataFrame(zippy,columns=['Number of Trips','Year','days_away'])
df4 = df3
dffinal = df4.merge(df4add,how='outer')


#Analyzing w/predictions
pivot = dffinal.pivot(index='days_away',columns='Year',values='Number of Trips')
pivot.plot(kind='line')






