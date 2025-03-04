import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


df = pd.read_csv('Air_Quality.csv')
desc = df.describe()

print(df.count())

average_rows = {}

particles = df.loc[df['Name'] == 'Fine particles (PM 2.5)']


for year in range(2010, 2021):
    year_str = f'Annual Average {year}'
    annual_data = particles[particles['Time Period'] == year_str]
    
    if not annual_data.empty:
        # Calculate the mean of 'Data Value'
        average_value = annual_data['Data Value'].mean()
        
        # Get the entire row with the mean value
        annual_average_row = annual_data.iloc[[0]]  # Take the first row and modify it
        annual_average_row['Data Value'] = average_value
        
        average_rows[year] = annual_average_row
        
average_df = pd.concat(average_rows.values(), ignore_index=True)


particle_map = {
    'Annual Average 2010': '2010',
    'Annual Average 2011': '2011',
    'Annual Average 2012': '2012',
    'Annual Average 2013': '2013',
    'Annual Average 2014': '2014',
    'Annual Average 2015': '2015',
    'Annual Average 2016': '2016',
    'Annual Average 2017': '2017',
    'Annual Average 2018': '2018',
    'Annual Average 2019': '2019',
    'Annual Average 2020': '2020'
    }

average_df['Time Period'] = average_df['Time Period'].map(particle_map)



## Asthma

asthma = df.loc[df['Name'] == 'Asthma emergency department visits due to PM2.5']

asthma_average_rows = {}

for year in range(2009, 2019):
    year_str = f'{year}-{year+2}'
    asthma_annual_data = asthma[asthma['Time Period'] == year_str]
    
    if not asthma_annual_data.empty:
        # Calculate the mean of 'Data Value'
        asthma_average_value = asthma_annual_data['Data Value'].mean()
        
        # Get the entire row with the mean value
        asthma_annual_average_row = asthma_annual_data.iloc[[0]]  # Take the first row and modify it
        asthma_annual_average_row['Data Value'] = asthma_average_value
        
        asthma_average_rows[year] = asthma_annual_average_row
        
        
asthma_average_df = pd.concat(asthma_average_rows.values(), ignore_index=True)

asthma_map = {
    '2009-2011': '2010',
    '2012-2014': '2013',
    '2015-2017': '2016',
    '2017-2019': '2018'
    }

asthma_average_df['Time Period'] = asthma_average_df['Time Period'].map(asthma_map)


# Create a DataFrame for plotting
plot_df = pd.DataFrame({
    'Year': average_df['Time Period'],
    'Average PM2.5': average_df['Data Value']
})

asthma_plot_df = pd.DataFrame({
    'Year': asthma_average_df['Time Period'],
    'Average Asthma Visits': asthma_average_df['Data Value']
})

# Plot the data with lines connecting points
plt.figure(figsize=(10, 6))
plt.scatter(plot_df['Year'], plot_df['Average PM2.5'], color='blue', label='PM2.5 Average')
plt.plot(plot_df['Year'], plot_df['Average PM2.5'], linestyle='-', marker='', color='lightblue')

## Asthma
plt.scatter(asthma_plot_df['Year'], asthma_plot_df['Average Asthma Visits'], color='orange', label='Asthma Average')
plt.plot(asthma_plot_df['Year'], asthma_plot_df['Average Asthma Visits'], linestyle='-', marker='', color='orange')

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Value (μg/m³ for PM2.5, per 100,000 humans for Asthma)')
plt.title('Annual Average Fine Particles (PM 2.5) and Asthma Emergency Department Visits due to PM2.5 from 2010-2020')
plt.legend()

plt.show()


## Statistics
common_years = ['2010', '2013', '2016', '2018']

# Filter Data
pm25_common = average_df[average_df['Time Period'].isin(common_years)]
asthma_common = asthma_average_df[asthma_average_df['Time Period'].isin(common_years)]

# Arrange data
pm25_common = pm25_common.sort_values(by='Time Period')
asthma_common = asthma_common.sort_values(by='Time Period')

# Numpy arrays for testing
pm25_values = pm25_common['Data Value'].to_numpy()
asthma_values = asthma_common['Data Value'].to_numpy()

# Pearson's correlation coefficient
corr_coeff, p_value = pearsonr(pm25_values, asthma_values)

print(f"Pearson correlation coefficient: {corr_coeff:.3f}")
print(f"P-value: {p_value:.3f}")

# Tulosten tulkinta
if p_value < 0.05:
    print("Correlation has statistical significance.")
else:
    print("Correlation doesn't have statistical significance.")



