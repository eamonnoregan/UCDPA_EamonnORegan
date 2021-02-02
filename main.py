import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from nameparser import HumanName

# Read in CSV file
ast_data = pd.read_csv('astronauts.csv')
print('ast_data DataFrame shape and column names (from the .csv file) are shown below: ')
print(ast_data.shape)
print(ast_data.columns)
print("\nFirst entries from the dataset:")
print(ast_data.head())

# 'Year' refers to the year that the astronaut was first selected. The below line counts missing values
year_miss = ast_data['Year'].isna().value_counts()
# 27 missing values were found in the above

# Read in the SQL database: An alternative astronaut database was obtained online,
# which contains the year of astronaut selection. The SQL database will help fill missing values from ast_data
engine = create_engine('sqlite:///data.sqlite')
ast_sql = pd.read_sql_query('SELECT * FROM ASTLIST', engine)
engine.dispose()

# Subset of the SQL DB to give only the name, the year of selection and year of birth
# Duplicate names dropped as not required
ast_sql_filter = ast_sql.loc[:, ['name', 'year_of_se', 'year_of_bi']]
ast_sql.columns = ast_sql.columns.str.rstrip(' ')
ast_sql_drop = ast_sql_filter.copy().drop_duplicates(subset="name")
print("\nEntries from the tidied dataframe obtained from the SQL file are show below:\n")
print(ast_sql_drop.head())
print("DataFrame shape:")
print(ast_sql_drop.shape)

# Obtain Missing Years: Break into 'first' and 'last' names to obtain missing years
# Merge data frames to join the required SQL values to the names with unknown year from the CSV dataframe
# Names did not match up, so a name parser combined with a lambda function was required
ast_sql_drop['first'] = ast_sql_drop['name'].apply(lambda x: HumanName(x).first.lower())
ast_sql_drop['last'] = ast_sql_drop['name'].apply(lambda x: HumanName(x).last.lower())
unknown_year = ast_data.copy()[ast_data['Year'].isna()]
unknown_year['first'] = unknown_year.loc[:, 'Name'].apply(lambda x: HumanName(x).first.lower())
unknown_year['last'] = unknown_year.loc[:, 'Name'].apply(lambda x: HumanName(x).last.lower())
un_year_merge = unknown_year.merge(ast_sql_drop, on=['first', 'last'], how='inner').set_index('Name')
print("\nThe first rows of dataset of astronauts with missing years is shown below, "
      "with new column 'year_of_se' to be used to fill in year in ast_data:")
print(un_year_merge.head())

# Dictionary then created, which was then used to fill missing years in the ast_data df
un_year_dict = un_year_merge.loc[:, 'year_of_se'].to_dict()

for i, ast in ast_data.iterrows():
    if np.math.isnan(ast['Year']):
        name = ast['Name']
        if name in un_year_dict:
            date = un_year_dict[name]
            ast_data.at[i, 'Year'] = date
        else:
            ast_data.at[i, 'Year'] = 0

# Found 5 rows where the year was 0 due differences in the name from both CSV and SQL datasets
# These 5 outliers were removed below to allow us to calculate ages further on
ast_sort = ast_data.copy().drop(ast_data[ast_data['Year'] == 0].index).sort_values('Year').set_index('Name')

# Calculation of the astronauts age when first selected in new column 'Age Selected'
# ast_sort used where we want to consider year, ast_data can still be used further on when we do not need this
ast_dob = ast_sort.loc[:, 'Birth Date']
ast_birthyear = pd.to_datetime([date.replace('/', '-') for date in ast_dob])
ast_sort['Birth Year'] = pd.DatetimeIndex(ast_birthyear).year
ast_sort['Year'] = ast_sort.loc[:, 'Year'].astype(int)
ast_sort['Age Selected'] = ast_sort['Year'] - ast_sort['Birth Year']

# Create a DF for astronauts selected in a Group
# Astronauts that did not belong to a group were dropped.
# Empty Military and Graduate columns filled with n/a for use later.
ast_sort['Military Branch'].fillna('n/a', inplace=True)
ast_sort['Graduate Major'].fillna('n/a', inplace=True)
ast_in_grp = ast_sort.copy().dropna(axis=0, subset=['Group'])

# When running the functions for each group (further on), the below outliers were found and corrected
# Outliers were found by printing the group in each iteration of the for loop (removed this print afterwards)
# 'James A. McDivitt' was found to contain the incorrect year and 'William S. McArthur Jr.' in the incorrect Group
ast_in_grp.loc['James A. McDivitt', 'Year'] = 1962
ast_in_grp.loc['William S. McArthur Jr. ', 'Group'] = 13
jmd_age = ast_in_grp.loc['James A. McDivitt', 'Year'] - ast_in_grp.loc['James A. McDivitt', 'Birth Year']
ast_in_grp.loc['James A. McDivitt', 'Age Selected'] = jmd_age

# 3 Functions were created to summarize each group dataframe :
# first function summarizes the total number, % female, % military, and % graduate major
# second function summarizes ages
# third function summarizes flight experience


def grp_nums(df):
    tot = len(df)
    fem_frac = round(len(df[df['Gender'] == "Female"]) / tot, 2)
    mil_frac = round(len(df[df['Military Branch'] != 'n/a']) / tot, 2)
    grad_frac = round(len(df[df['Graduate Major'] != 'n/a']) / tot, 2)
    qty = (tot, fem_frac, mil_frac, grad_frac)
    return qty


def grp_ages(df):
    age_mean = round(df['Age Selected'].mean(), 2)
    age_min = df['Age Selected'].min()
    age_max = df['Age Selected'].max()
    ages = (age_mean, age_min, age_max)
    return ages


def flt_exp(df):
    fhr_mean = round(df['Space Flight (hr)'].mean(), 2)
    spw_mean = round(df['Space Walks (hr)'].mean(), 2)
    exp = (fhr_mean, spw_mean)
    return exp


# List and Rows created for use in the for loop below
lists = []
rows = []

# for loop below subsets the ast_in_grp dataframe, based on the associated group of each astronaut
# this loop passes each dataframe to the functions defined previously, to create a summary dataframe
for i in range(int(ast_in_grp['Group'].min()), int(ast_in_grp['Group'].max()) + 1):
    rows.append('Group_' + str(i))
    df = ast_in_grp[ast_in_grp['Group'] == i]
    lists.append(grp_nums(df) + grp_ages(df) + flt_exp(df))

# Summary dataframe of all groups is created below.
group_df = pd.DataFrame(lists)
group_df.index = rows
group_df.columns = ['No.Sel', '%F', '%Mil', '%G Mj', 'MeanAge', 'MinAge', 'MaxAge', 'MeanFl(hrs)', 'Mean spw(hrs)']
print('\nSummary Dataframe for each group:\n')
print(group_df)

# List all passes full list to the functions described previously
# (excluding astronauts where 'Year' was not available)
list_all = (grp_nums(ast_sort) + grp_ages(ast_sort) + flt_exp(ast_sort))

# Count the astronauts from each state by splitting the string of each row and save to a DF (Part 1/2)
place_spl = ast_data['Birth Place'].str.split(", ", expand=True)
state_list = place_spl[1]
plc_count = dict(state_list.value_counts())
place_df = pd.DataFrame.from_dict(plc_count, orient="index").reset_index()
place_df.columns = ['Code', 'No. Entries']

# Read in State Names from a .csv file to merge with the state counts from above (Part 2/2)
# Using the merge dataframe, it was possible to create a new col (US born and Non-US born),
# based on cols from the CSV file
state_abb = pd.read_csv('stateabbrev.csv')
state_merge = state_abb.merge(place_df, on='Code', how='outer').sort_values('No. Entries', ascending=False)
state_merge.loc[:, 'Abbrev'].fillna('na', inplace=True)
state_merge.loc[:, 'State'].fillna('na', inplace=True)
state_merge.loc[state_merge.Abbrev != 'na', 'US Born'] = 'Yes'
state_merge.loc[state_merge.Abbrev == 'na', 'US Born'] = 'No'
state_merge.loc[state_merge.State == 'na', 'State'] = state_merge['Code']
state_merge.dropna(subset=['No. Entries'], inplace=True)
print('\nFirst rows from the States of Birth count dataframe:\n')
print(state_merge.head())

# A list of all education institutes that an astronaut attended is contained in the 'Alma Mater' column.
# Since each individual attended multiple, each line of the col was split and the total columns concatenated
col = ast_data['Alma Mater'].str.split("; ", expand=True)
full_col = pd.concat([col[0], col[1], col[2], col[3], col[4], col[5]])
col_count = dict(full_col.value_counts())
college_df = pd.DataFrame.from_dict(col_count, orient="index").reset_index()
college_df.columns = ["College/University", "Individual Count"]
print("\nTop 5 most common educational institutions are listed below:\n")
print(college_df.head())
print()
coll_reduced = college_df[college_df["Individual Count"] > 4]

# Astronaut Experience - we can use the full data set here (use ast_sort to compile most experienced)
# Group by whether the astronauts background is a graduate major or is former military
ast_exp = ast_data.copy().set_index(ast_data['Name'])
ast_exp = ast_exp.drop(ast_exp[ast_exp['Space Flights'] < 1].index)
a = ast_exp.sort_values('Space Flights')

print(ast_exp.tail())
ast_exp['Graduate Major'] = ast_exp.loc[:, 'Graduate Major'].astype(str)
ast_exp['Military Rank'] = ast_exp.loc[:, 'Military Branch'].astype(str)

for i, row in ast_exp.iterrows():
    if row['Graduate Major'] == 'nan' and row['Military Rank'] == 'nan':
        ast_exp.at[i, 'Grad/Mil'] = 'None'
    elif row['Graduate Major'] != 'nan' and row['Military Rank'] == 'nan':
        ast_exp.at[i, 'Grad/Mil'] = 'Graduate'
    elif row['Graduate Major'] == 'nan' and row['Military Rank'] != 'nan':
        ast_exp.at[i, 'Grad/Mil'] = 'Military'
    else:
        ast_exp.at[i, 'Grad/Mil'] = 'Both'

# We use ast_ext here as we are excluded selected astronauts that have never been on a mission (i.e. 0 flights)
# The below groups astronauts into the four categories from above.
group_exp_cols = ['Space Flights', 'Space Walks', 'Space Flight (hr)', 'Space Walks (hr)']
group_exp1 = ast_exp.groupby('Grad/Mil')[group_exp_cols[0:2]].agg([np.mean, np.median])
group_exp2 = ast_exp.groupby('Grad/Mil')[group_exp_cols[2:4]].agg([np.mean, np.median])
print("Grouped Space Flights and Space Walks into the Military/Graduate Categories:")
print(group_exp1)
print(group_exp2)


# As of 2013, % of the total were active, % were in management, % Deceased (using full data: ast_data)
status = dict(ast_data['Status'].value_counts())
print("As of 2013, the statuses of astronauts in the database are categorized below:")
print(status)

# % of all astronauts had died on a mission (using ast_data)
died_on_mission = round(ast_data['Death Mission'].notnull().sum() * 100 / len(ast_data['Death Mission']), 2)
print()
print(str(died_on_mission)+"% of astronauts had died while on mission (as of 2013 data), "
                           "which highlights the dangers of the job")

# ------------------- Plotting Data -----------------------
# Using Matplotlib

# Create a Figure and an array of subplots with 2 rows and 2 columns
# The figure below shows the number selected, % Female, % Military and % Graduate
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
fig.suptitle("Plotted Group Data (Part 1 of 2)", size=12)
ax1.plot(group_df.index, group_df['No.Sel'], marker=".", color='g')
ax1.set_title('Number Selected per Group', size=10)
ax1.set_ylim([0, 40])
ax1.set_ylabel('No. Selected')

ax2.plot(group_df.index, group_df['%F'], marker=".", color='r')
ax2.set_title('% Female per Group', size=10)
ax2.set_ylim([0, 0.35])
ax2.set_ylabel('% Female')

ax3.plot(group_df.index, group_df['%Mil'], marker=".", color='y')
ax3.set_title('% Military Trained per Group', size=10)
ax3.set_ylim([0, 1])
ax3.set_ylabel('% Military')

ax4.plot(group_df.index, group_df['%G Mj'], marker=".", color='m')
ax4.set_title('% Graduate Major per Group', size=10)
ax4.set_ylim([0, 1])
ax4.set_ylabel('% Graduate Major')

for ax in fig.axes:
    plt.xlim([0, 19])
    plt.grid(b=None, which='major', axis='y')
    plt.sca(ax)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.5)
    plt.yticks(size=6)
    plt.xticks(rotation=60, size=6)

# Using Matplotlib
# The next figure shows the second part of the group data, astronaut ages and flight experience

fig2, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9, 9))
fig2.suptitle("Plotted Group Data (Part 2 of 2)", size=12)
ax1.set_title('Mean, Min and Max Ages of Astronauts in each Group', size=12)
ax1.plot(group_df.index, group_df['MeanAge'], label='Mean Age', marker=".", color='b')
ax1.plot(group_df.index, group_df['MinAge'], label='Min Age', color='g', linestyle='--')
ax1.plot(group_df.index, group_df['MaxAge'], label='Max Age', color='r', linestyle='--')
ax1.set_ylim([10, 50])
ax1.set_ylabel('Age at  (Years)')
ax1.grid(b=None, which='major', axis='y')
ax1.tick_params(axis="x", labelsize=6, rotation=60)
ax1.tick_params(axis="y", labelsize=6)
ax1.legend()

ax2.plot(group_df.index, group_df['MeanFl(hrs)'], label='Mean Flight Hours', marker=".", color='y')
ax2.set_title('Mean Flight and Spacewalk Hours per Group', size=12)
ax3 = ax2.twinx()
ax3.plot(group_df.index, group_df['Mean spw(hrs)'], label='Mean Spacewalk Hours', marker=".", color='m')
ax2.set_ylabel('Mean Flight Time (hrs)')
ax3.set_ylabel('Mean Space Walk (hrs)')
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax3.get_legend_handles_labels()
ax3.legend(h1+h2, l1+l2, loc=2)
ax2.tick_params(axis="x", labelsize=6, rotation=60)
ax2.tick_params(axis="y", labelsize=6)
ax3.grid(b=None, which='major', axis='y')

for ax in fig2.axes:
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

plt.show()
with sns.color_palette("tab10"):
    fl_plot = sns.jointplot(x='Space Flights', y='Space Flight (hr)', data=ast_exp, hue='Grad/Mil', alpha=0.9)
    fl_plot.fig.suptitle("Space Flight (hrs) vs Space Flights for Astronaut Background", y=1)
    fl_plot.fig.set_size_inches(5, 5)
    sw_plot = sns.jointplot(x='Space Walks', y='Space Walks (hr)', data=ast_exp, hue='Grad/Mil', alpha=0.9)
    sw_plot.fig.suptitle("Space Walk (hrs) vs Space Walks for Astronaut Background", y=1)
    sw_plot.fig.set_size_inches(5, 5)
plt.show()


# Bar Plot of Educational Institutions attended by astronauts

with sns.axes_style("whitegrid"):
    g_coll = sns.catplot(x="Individual Count", y="College/University", data=coll_reduced, kind="bar")
    g_coll.fig.suptitle("Colleges/Universities Attended by Astronauts ", y=1)
    g_coll.set(xlim=(0, 50))
    g_coll.set_yticklabels(size=8)
    g_coll.set_xticklabels(size=8)

plt.show()

# Bar plot of states/countries where astronauts where born
with sns.axes_style("whitegrid"):
    sns.set_context("paper")
    palette_colors = {"Yes": "blue", "No": "red"}
    sns.set_context(rc={'patch.linewidth': 0.0})
    g_state = sns.catplot(x="No. Entries", y="State", data=state_merge, kind="bar",
                       hue='US Born', palette=palette_colors, aspect=1)

    g_state.fig.suptitle("Count of Astronauts Born in Each State", y=1)
    g_state.set_yticklabels(size=6)
    g_state.set_xticklabels(size=8)
    g_state.set(xlim=(0, 30))

plt.show()
