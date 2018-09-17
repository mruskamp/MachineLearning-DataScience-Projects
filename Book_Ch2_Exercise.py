import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Function to save image of histogram
#def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
#    path = "C:\\Users\\ruskampm\\OneDrive - FM Global\\Desktop\\a.png"
#    print("Saving figure", fig_id)
#    if tight_layout:
#        plt.tight_layout()
#    plt.savefig(path, format=fig_extension, dpi=resolution)

# Seed np
np.random.seed(42)

# Set up graph and file saving for image
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

df = pd.read_csv("C:\\Users\\ruskampm\\OneDrive - FM Global\\Desktop\\housing_xcl.csv")

# Check dataframe
pd.set_option('display.max_columns', None)
#print(df.head(15))
#print(df.values)
#print(df["ocean_proximity"].value_counts())

#df.hist(bins=50, figsize=(20,15))
#save_fig("attribute_histogram_plots")
#plt.show()

# Split data set to train and test data and randomly shuffle them after splitting
#shuffled_indices = np.random.permutation(len(df))
#test_set_size = int(len(df) * .7)
#train_indices = shuffled_indices[:test_set_size:]
#test_indices = shuffled_indices[test_set_size:]

# Use built in split function
#train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
#print(len(train_set), "train +", len(test_set), "test")

# Group all >5 median_incomes to have median_income = 5 into new category income_cat
df["income_cat"] = np.ceil(df["median_income"] / 1.5)
df["income_cat"].where(df["income_cat"] < 5, 5.0, inplace=True)
#df['income_cat'].hist()
#plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]


# Check that the proportions of different median_incomes/income_cats are very close
# to equal between the actual data and our train/test data
#print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))
#print(df["income_cat"].value_counts() / len(df))

# Drop the income_cat column - we know the data is shuffled well
#print(df.head())
df.drop('income_cat', axis=1, inplace=True)
strat_train_set.drop('income_cat', axis=1, inplace=True)
strat_test_set.drop('income_cat', axis=1, inplace=True)
#print(df.head())
#for set in (strat_train_set, strat_test_set):
#    set.drop(["income_cat"], axis=1, inplace=True)

# Print a map of california residences
df = strat_train_set.copy()
#df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
#plt.show()

# Check correlation of each other column with median house value
#corr_matrix = df.corr()
#print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Combine attributes to make more meaningful ones
df["rooms_per_household"] = df["total_rooms"]/df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"]/df["total_rooms"]
df["population_per_household"]=df["population"]/df["households"]

# Drop labels for training set (df) and store them in another variable
df = strat_train_set.drop("median_house_value", axis=1)
df_labels = strat_train_set["median_house_value"].copy()

# Create imputer to find the median of all statistics, then replace missing values
imputer = Imputer(strategy="median")
df_nums = df.drop("ocean_proximity", axis=1)
imputer.fit(df_nums)
#print(imputer.statistics_)
X = imputer.transform(df_nums)
df_tr = pd.DataFrame(X, columns=df_nums.columns)
#print(df_tr.info())

# Encode string column to be represented by ints, like an enum
encoder = LabelEncoder()
df_cat = df['ocean_proximity']
df_cat_encoded = encoder.fit_transform(df_cat)
#print(df_cat_encoded)
#print(encoder.classes_)

# One-hot encoding the labels after reshaping from [14448] to [14448, 1]
encoder = OneHotEncoder()
df_cat_onehot = encoder.fit_transform(df_cat_encoded.reshape(-1,1))
#print(df_cat_onehot.toarray())
# LabelBinarizer can perform cateogry to integers to binary/onehot in one step



