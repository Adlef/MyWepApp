import pandas as pd
import plotly.graph_objs as go
import numpy as np
import os

csv_files_path = os.getcwd() + '/data/WorldHappiness/'
csv_list_files = os.listdir(csv_files_path)

# list of dataframes
df_dict = {} 

for csv_file in  csv_list_files:
    year = int(csv_file.split('.csv')[0])
    df_year = pd.read_csv(csv_files_path + csv_file)
    df_year['Year'] = year
    df_dict[str(year)] = df_year
    #print("World Happiness Report - Year: {}, Data size: {}".format(year, df_year.shape[0]))
    #print("Columns names: {}\n".format(list(df_year.columns)))

# PREPARING THE DATA

# Columns kept: 'Country', 'Happiness Rank', 'Happiness Score','Standard Error',
#               'Economy (GDP per Capita)', 'Social support', 'Health (Life Expectancy)', 
#               'Freedom', 'Trust (Government Corruption)', 'Generosity', 'Dystopia Residual', 'Year',
#               'Lower Confidence Interval', 'Upper Confidence Interval'

# 2015
df_dict['2015'] = df_dict['2015'].rename(columns={'Family': 'Social support'})

# 2016
df_dict['2016'] = df_dict['2016'].rename(columns={'Family': 'Social support'})

# 2017
df_dict['2017'] = df_dict['2017'].rename(columns={'Happiness.Rank': 'Happiness Rank'})
df_dict['2017'] = df_dict['2017'].rename(columns={'Happiness.Score': 'Happiness Score'})
df_dict['2017'] = df_dict['2017'].rename(columns={'Whisker.high': 'Upper Confidence Interval'})
df_dict['2017'] = df_dict['2017'].rename(columns={'Whisker.low': 'Lower Confidence Interval'})
df_dict['2017'] = df_dict['2017'].rename(columns={'Economy..GDP.per.Capita.': 'Economy (GDP per Capita)'})
df_dict['2017'] = df_dict['2017'].rename(columns={'Health..Life.Expectancy.': 'Health (Life Expectancy)'})
df_dict['2017'] = df_dict['2017'].rename(columns={'Trust..Government.Corruption.': 'Trust (Government Corruption)'})
df_dict['2017'] = df_dict['2017'].rename(columns={'Dystopia.Residual': 'Dystopia Residual'})
df_dict['2017'] = df_dict['2017'].rename(columns={'Family': 'Social support'})
    
# 2018
df_dict['2018'] = df_dict['2018'].rename(columns={'Overall rank': 'Happiness Rank'})
df_dict['2018'] = df_dict['2018'].rename(columns={'Score': 'Happiness Score'})
df_dict['2018'] = df_dict['2018'].rename(columns={'Country or region': 'Country'})
df_dict['2018'] = df_dict['2018'].rename(columns={'GDP per capita': 'Economy (GDP per Capita)'})
df_dict['2018'] = df_dict['2018'].rename(columns={'Healthy life expectancy': 'Health (Life Expectancy)'})
df_dict['2018'] = df_dict['2018'].rename(columns={'Freedom to make life choices': 'Freedom'})
df_dict['2018'] = df_dict['2018'].rename(columns={'Perceptions of corruption': 'Trust (Government Corruption)'})

# 2019
df_dict['2019'] = df_dict['2019'].rename(columns={'Overall rank': 'Happiness Rank'})
df_dict['2019'] = df_dict['2019'].rename(columns={'Score': 'Happiness Score'})
df_dict['2019'] = df_dict['2019'].rename(columns={'Country or region': 'Country'})
df_dict['2019'] = df_dict['2019'].rename(columns={'GDP per capita': 'Economy (GDP per Capita)'})
df_dict['2019'] = df_dict['2019'].rename(columns={'Healthy life expectancy': 'Health (Life Expectancy)'})
df_dict['2019'] = df_dict['2019'].rename(columns={'Freedom to make life choices': 'Freedom'})
df_dict['2019'] = df_dict['2019'].rename(columns={'Perceptions of corruption': 'Trust (Government Corruption)'})

# MAKING 1 SINGLE DATAFRAME

df = pd.DataFrame()

for year in df_dict:
    df = pd.concat([df, df_dict[year]], ignore_index=True, sort=True)

# DROPPING UNUSED FEATURES

df = df.drop(['Upper Confidence Interval', 'Lower Confidence Interval', 'Standard Error', 'Dystopia Residual'], axis=1)

# DICTIONNARY OF REGIONS AND CORRESPONDING COUNTRIES

region_list = list(df['Region'].unique())
dict_region_countries = {}

for region in region_list:
    countries_from_region_list = list(df[df['Region'] == region]['Country'].unique())
    dict_region_countries[region] = countries_from_region_list
    
#print("List of regions and countries: {}".format(dict_region_countries))

def fillRegionForCountry(country):
    '''
    This function fills the region for the specified country.
    '''
    try:
        for region in dict_region_countries:
            if country in dict_region_countries[region]:
                df.loc[df['Country'] == country, ['Region']] = region
    except:
        print("Error with {}".format(country))

# FILLING THE REGION ATTRIBUTE

for country in list(df['Country'].unique()):
    fillRegionForCountry(country)
    
# FILLING LAST MISSING NAN VALUES (REGION)

df.loc[df['Country'] == "Taiwan Province of China", ['Region']] = 'Eastern Asia'
df.loc[df['Country'] == "Hong Kong S.A.R., China", ['Region']] = 'Eastern Asia'
df.loc[df['Country'] == "Trinidad & Tobago", ['Region']] = 'Latin America and Caribbean'
df.loc[df['Country'] == "Northern Cyprus", ['Region']] = 'Western Europe'
df.loc[df['Country'] == "North Macedonia", ['Region']] = 'Central and Eastern Europe'
df.loc[df['Country'] == "Gambia", ['Region']] = 'Sub-Saharan Africa'

# FILLING NAN VALUE (TRUST)

df.loc[489, 'Trust (Government Corruption)'] = df[df['Country'] == 'United Arab Emirates']['Trust (Government Corruption)'].mean()

def getTopCountriesForCritPerYear(list_of_countries, variable, dict_values):
    '''
    From a list of top countries considered for a feature, this function returns a DataFrame with the following structure:
    - index: the list of countries
    - columns: 2015, 2016, 2017, 2018, 2019, inc_dec_percent
    '''
    list_of_countries.append('Average Country')
    score_by_country = {}
    for country in list_of_countries:
        try:
            variable_score_country = []
            for year in years:
                if country != 'Average Country':
                    variable_score_country.append(dict_values[variable + '_' + str(year)][country])         
                else:
                    variable_score_country.append(df[df['Year'] == year][crit].median())
            score_by_country[country] = variable_score_country
        except:
            # The country has no value for a criteria
            do_nothing = True
    
    # DATAFRAME WITH COUNTRIES AS INDEX
    df_top_countries_in_variable = pd.DataFrame.from_dict(score_by_country, orient='index', columns=['2015', '2016', '2017', '2018', '2019'])
    
    # PERCENTAGE INCREASE / DECREASE OF FEATURE BETWEEN 2015 AND 2019
    df_top_countries_in_variable['inc_dec_percent'] = round(100*(df_top_countries_in_variable['2019'] - df_top_countries_in_variable['2015'])/df_top_countries_in_variable['2015'],2)

    return df_top_countries_in_variable

# countries sorted out by their top score in the different criteria and per year

top_countries = {}

years = [2015, 2016, 2017, 2018, 2019]
criteria = ['Social support', "Trust (Government Corruption)", 'Economy (GDP per Capita)',
            'Freedom', 'Health (Life Expectancy)', 'Happiness Score','Generosity']

for crit in criteria:
    for year in years:
        top_countries[crit + '_' + str(year)] = df[df['Year'] == year].set_index('Country')[crit].sort_values(ascending=False)
        
        

# FOR COUNTRIES FROM 2015 EXCEPT 'AUSTRALIA' (NOT ANYMORE IN TOP 10 IN 2019)
all_countries = list(top_countries['Happiness Score_2015'].index)

# GET TOP 9 COUNTRIES IN SCORE AND THEIR CORRESPONDING DATAFRAMES IN THE DIFFERENT CRITERIA
dict_of_df = {}
dict_of_df['stacked_score'] = 0
for crit in criteria:
    df_all_countries = getTopCountriesForCritPerYear(all_countries,crit, top_countries)
    dict_of_df[crit] = df_all_countries
    dict_of_df['stacked_score'] += df_all_countries

    
# FOR COUNTRIES GETTING RANK IN THE DIFFERENT FEATURES
dict_rank_countries = {}

#FOR COUNTRIES CONSIDERED
for country_happiness in dict_of_df['stacked_score'].index:
    rank_list_country = []
    
    # GET RANK PER FEATURE
    for crit in criteria:
        for index, country in enumerate(top_countries[crit + "_2019"].index):
            if country == country_happiness:
                rank = index + 1
                rank_list_country.append(rank) 
                #print("[{}] Country: {}, rank: {}/{}".format(crit,country_happiness,rank,len(top_countries[crit + "_2019"].index)))
    dict_rank_countries[country_happiness] = rank_list_country
    
countries_considered = list(dict_of_df[crit].index[:9])
countries_considered.append('Average Country')
    
def return_figures():
    """Creates four plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """

    # first chart plots arable land from 1990 to 2015 in top 10 economies 
    # as a line chart
    
    graph_one = []    
    for country in countries_considered:
        graph_one.append(
          go.Scatter(
          x = [2015,2016,2017,2018,2019],
          y = dict_of_df['Happiness Score'].loc[country, ['2015', '2016','2017','2018','2019']].values,
          mode = 'lines',
          name = country
          )
        )

    layout_one = dict(title = 'Happiness Score For The Top 9 Countries From 2015 to 2019',
                xaxis = dict(title = 'Years'),
                yaxis = dict(title = 'Countries'),
                )

# second chart plots ararble land for 2015 as a bar chart    
    graph_two = []
    
    # Figure 1 - horizontal bars displaying stacked scores from all criteria per top countries - 2019
    countries_sortedby_stacked_score = dict_of_df['stacked_score']['2019'].sort_values().index[125:]
    
    colors_bars = ['cornflowerblue', 'brown', 'gold', 'mediumseagreen', 'darkorange', 'turquoise',
            'ivory']
    
    for index, crit in enumerate(criteria):
        graph_two.append(
          go.Bar(
          y = dict_of_df[crit]['2019'].loc[countries_sortedby_stacked_score].index,
          x = dict_of_df[crit]['2019'].loc[countries_sortedby_stacked_score].values, 
          orientation = 'h',
          name = crit,
          text = ["RANK : " + str(dict_rank_countries[country][index]) + " / " + str(len(dict_of_df['stacked_score']['2019'])) for country in countries_sortedby_stacked_score],
          marker=dict(
            color=colors_bars[index])
          )
        )

    layout_two = dict(title = 'Stacked Scores For Top Countries in Happiness - 2019',
                xaxis = dict(title = 'Stacked Scores'),
                yaxis = dict(tickangle=-30),
                barmode='stack',
                width=800,
                height=400
                )


   
    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))

    return figures