import numpy as np
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
import os
import pandas as pd

cwd = os.getcwd()
data_directory = cwd.replace('Pre-Modelling','')+r'InputData/'

MASTER_2018 = pd.read_csv(data_directory+'SEM_Input_2018.csv')
print(MASTER_2018.shape)
one_hot_encoded_data = pd.get_dummies(MASTER_2018, columns = ['assam_soil', 'assam_lith','land use'],
                                     dummy_na=True,drop_first=True)

# Ensuring OHE data represents Null values.
one_hot_encoded_data.loc[one_hot_encoded_data.assam_soil_nan == 1,

                         [  # "assam_soil_clay, loamy clay, sandy clay, silty clay", reference
                             "assam_soil_loamy sand, sand",
                             "assam_soil_rocky, other non-soil categories",
                             "assam_soil_loam, silt loam, silt, sandy loam"]] = np.nan

del one_hot_encoded_data["assam_soil_nan"]

one_hot_encoded_data.loc[one_hot_encoded_data.assam_lith_nan == 1,

                         [  # "assam_lith_metamorphic, mesozoic and paleozoic intusive", reference
                             "assam_lith_neogene sedimentary rock",
                             "assam_lith_paleogene sedementary rock",
                             "assam_lith_paleozoic rock",
                             "assam_lith_undeveloped precambrian rock",
                             "assam_lith_quaternary sediments",
                             "assam_lith_tertiary sedimentary rocks",
                             "assam_lith_water"
                         ]] = np.nan

del one_hot_encoded_data["assam_lith_nan"]

one_hot_encoded_data.loc[one_hot_encoded_data['land use_nan'] == 1,

                         [  # "land use_bare ground",reference
                             "land use_built",
                             "land use_range land",
                             "land use_vegetation",
                             "land use_water"
                         ]] = np.nan
del one_hot_encoded_data["land use_nan"]

X = one_hot_encoded_data.drop(['ID','yr','x','y'],axis=1)
X = X.replace('U/A', np.nan)

imputer = MissForest(random_state=1337) #miss forest
X_imputed = imputer.fit_transform(X, cat_vars=np.array(range(74,88,1)))
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
X_imputed.to_csv(data_directory+'SEM_X_Imputed.csv')
