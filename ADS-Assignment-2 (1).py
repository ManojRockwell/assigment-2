
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np


def reading_data(filedata):
    """
        Reads a CSV file and extracts the required data.

        Parameters:
            file_path (str): Path to the CSV file.

        Returns:
            tuple: A tuple containing two pandas DataFrames - new_dataframe and countries.
    """

    # read the CSV file and skip the first 4 rows of metadata
    dataframe = pd.read_csv(filedata, skiprows=4)
    countries = dataframe.drop(
        columns=['Country Code', 'Indicator Code', 'Unnamed: 66'], inplace=True)
    countries = dataframe.set_index('Country Name').T
    new_dataframe = dataframe.set_index('Country Name').reset_index()
    return new_dataframe, countries


def choosing_attribute(indicators, describe_data):
    '''
    function for choosing an choosing_attribute
    '''
    describe_data = describe_data[describe_data['Indicator Name'].isin([indicators])]
    return describe_data


def country_selection(countries, describe_data):
    """
    Extracts data for a specific country from a given DataFrame.

    Parameters:
        country (str): The name of the country to extract data for.
        dataframe (pd.DataFrame): The DataFrame containing the data.

    Returns:
        pd.DataFrame: A new DataFrame containing the extracted data for the specified country.
    """
    describe_data = describe_data[describe_data['Country Name'].isin([countries])]
    describe_data = describe_data.set_index("Indicator Name")
    describe_data = describe_data.drop("Country Name", axis=1)
    # Transposing the dataframe
    describe_data = describe_data.T
    return describe_data

# Function to plot a multi-line graph
def multi_line_plot(indicator, countries, df):
    years = [str(year) for year in range(1990, 2020, 2)]
    data = df.loc[df['Country Name'].isin(countries) & df['Indicator Name'].isin([indicator]), years].T
    kurt = data.kurtosis(axis=0, skipna=True)
    print(kurt.head())

    # Specify the colors using the 'spectral' colormap
    colors = ['#FFCC00', '#FF6600', '#CC3399', '#A71E34', '#006699', '#669900']

    # Plot the lines with the specified colors
    for i in range(len(countries)):
        plt.plot(data.index, data.iloc[:, i], linestyle='-', color=colors[i], linewidth=2.5)

    plt.legend(countries, fontsize="7", loc="upper left")
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.xticks(rotation=90)
    plt.show()

def stacked_bar_chart(countries, indicator):
    """
    Plot the specified indicator for the given list of countries as a stacked bar chart.
    
    Select the indicator and data of your choice
    group the data according to the country and the respective years
    """
    selected_data = new_dataframe.loc[(new_dataframe['Country Name'].isin(countries)) &
                                (new_dataframe['Indicator Name'] == indicator) &
                                (new_dataframe[['1970', '1980', '1990', '2000', '2010', '2020']].notnull().all(axis=1)), :]

    # Grouping the selected data
    selected_data_grouped = selected_data.groupby(
        'Country Name')[['1970', '1980', '1990', '2000', '2010', '2020']].agg(list)

    fig, ax = plt.subplots()

    # Initialize variables for plotting
    width = 0.5
    colors = ['#FFCC00', '#FF6600', '#CC3399', '#A71E34', '#006699', '#669900']
    years = ['1970', '1980', '1990', '2000', '2010', '2020']
    x_coords = np.arange(len(countries))
    bottom = np.zeros(len(countries))

    # Plot the stacked bar chart
    for i, year in enumerate(years):
        data = selected_data_grouped[year].apply(lambda x: x[0])
        ax.bar(x_coords, data, width, label=year,
               color=colors[i], alpha=1, edgecolor='black', bottom=bottom)
        bottom += data

    ax.set_xticks(x_coords)
    ax.set_xticklabels(countries, rotation=90)
    ax.set_ylabel(indicator)
    ax.set_title(indicator, fontsize="10")
    ax.legend(fontsize="7", loc="upper left")

    # Calculate the skewness of the selected data
    skew = selected_data_grouped.skew(axis=0, skipna=True)
    skew.head()

    # Display the plot
    plt.show()


def pie_chart(country, indicator, start_year, end_year, data):
    """
    Plot a pie chart for the specified indicator and country from start_year to end_year.

    Parameters:
        country (str): The name of the country.
        indicator (str): The name of the indicator.
        start_year (int): The starting year.
        end_year (int): The ending year.
        data (pd.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
    # Filter data for the specified country, indicator, and years
    filtered_data = data.loc[(data['Country Name'] == country) &
                             (data['Indicator Name'] == indicator), str(start_year):str(end_year)]

    # Sum the values for each year
    year_totals = filtered_data.iloc[0, :].astype(float)
    colors = ['#FFCC00', '#FF6600', '#CC3399', '#A71E34', '#006699', '#669900']

    # Create the pie chart
    fig, ax = plt.subplots()
    ax.pie(year_totals, labels=year_totals.index, autopct='%1.1f%%', colors=colors)
    ax.set_title(f"{indicator} for {country} ({start_year}-{end_year})")

    plt.show()


def bar_chart(indicator, country, start_year, end_year, data):
    """
    Plot a bar chart for the specified indicator and country from start_year to end_year.

    Parameters:
        indicator (str): The name of the indicator.
        country (str): The name of the country.
        start_year (int): The starting year.
        end_year (int): The ending year.
        data (pd.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
    # Filter data for the specified indicator, country, and years
    filtered_data = data.loc[(data['Country Name'] == country) &
                             (data['Indicator Name'] == indicator), str(start_year):str(end_year)]

    # Get the values for the specified years
    values = filtered_data.iloc[0].astype(float)

    # Create the bar chart
    fig, ax = plt.subplots()
    ax.bar(range(len(values)), values, color='#A71E34')
    ax.set_xlabel('Year')
    ax.set_ylabel('Agriculture, forestry, and fishing (% of GDP)', fontsize=10)
    ax.set_title(f"{indicator} for {country} ({start_year}-{end_year})")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(values.index, rotation=90)

    plt.show()
    
    
#function to plot a heat map
def heatmap(country, data, cols, cmap='viridis'):  
    subset = country_selection(country, data)
    columns = subset[cols]
    corr = columns.corr()
    sb.heatmap(corr, annot=True, cmap=cmap)
    plt.title(f"Correlation matrix for {country}")
    plt.show()

"""  
reading the data and transposing it 
storing it in a new dataframe
"""

new_dataframe, countries = reading_data(r"D:\python core\wbdata.csv")

#describe()
print(new_dataframe.describe())
print(countries.describe())

#plotting a line graph
multi_line_plot('Population growth (annual %)', [
               'Germany', 'Japan', 'United Kingdom','China', 'India', 'Italy'], new_dataframe)

multi_line_plot('Total greenhouse gas emissions (% change from 1990)', [
               'Germany', 'Japan', 'United Kingdom','China', 'India', 'Italy'], new_dataframe)

#seleted countries for bar graph
countries = ['Germany', 'Japan', 'United Kingdom',
              'China', 'India', 'Italy']

indicator = 'Urban population (% of total population)'

stacked_bar_chart(countries, indicator)

# Specify the parameters for the pie chart
country = 'China'
indicator = 'Population growth (annual %)'
start_year = 1991
end_year = 2000

# Generate the pie chart
pie_chart(country, indicator, start_year, end_year, new_dataframe)

bar_chart('Agriculture, forestry, and fishing, value added (% of GDP)', 'China', 1980, 2010, new_dataframe)


#plotting a heat map    
heatmap('China', new_dataframe, ['Electricity production from oil sources (% of total)',
                                    'Electricity production from natural gas sources (% of total)',
                                    'Electricity production from hydroelectric sources (% of total)',
                                    'Electricity production from coal sources (% of total)'], cmap='OrRd')

heatmap('China', new_dataframe, ['Urban population (% of total population)',
                                 'Agriculture, forestry, and fishing, value added (% of GDP)',
                                 'Total greenhouse gas emissions (% change from 1990)',
                                 'Agricultural land (% of land area)',
                                 'Population growth (annual %)'], cmap='OrRd')

