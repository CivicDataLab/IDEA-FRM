import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from statsmodels.stats.outliers_influence import variance_inflation_factor
from joblib import Parallel, delayed



cwd = os.getcwd()
data_directory = cwd.replace('Pre-Modelling','')+r'ML_InputData/'
results_directory = cwd+r'/Results/'

# DEPENDENT VARIABLE
y = pd.read_csv(data_directory+'y_train.csv')
# INDEPENDENT VARIABLES - AFTER RF IMPUTATION, OHE and NORMALISATION
X_train = pd.read_csv(data_directory+'X_train.csv')

# Separate continuous and categorical variables
X_train_con = X_train[['ndbi', 'gmted_drainage_density_without_1', 'srtm_filled_dem',
       'GCN250_ARCIII_average', 'assam_dist_from_major_rivers_updated_3857',
       'ndvi', 'sum', 'strm_filled_slope_degrees']]

# CORRELATION MATRIX
rcParams['figure.figsize'] = 15,15
heatmap = sns.heatmap(X_train_con.corr().round(2),
        cmap="YlGnBu",
        annot=True).get_figure()
heatmap.savefig(results_directory+"CorrelationMatrix.png")

#VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X_train_con.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X_train_con.values, i)
                   for i in range(len(X_train_con.columns))]
vif_data.to_csv(results_directory+'vif.csv',index=False)

#PCA
from sklearn.decomposition import PCA
pca2 = PCA(n_components=2)
principalComponents = pca2.fit_transform(X_train)
principalDf = pd.DataFrame(data = principalComponents,
                           columns = ['principal component 1', 'principal component 2'])
principalDf.to_csv(results_directory+'PCA_2comp.csv',index=False)

pca3 = PCA(n_components=3)
principalComponents = pca3.fit_transform(X_train)
principalDf = pd.DataFrame(data = principalComponents,
                           columns = ['principal component 1', 'principal component 2','principal component 3'])
principalDf.to_csv(results_directory+'PCA_3comp.csv',index=False)

#t-SNE
from sklearn.manifold import TSNE
def tsne_2comp(perplexity_score):
    print(perplexity_score)
    model = TSNE(n_components = 2, random_state = 0, perplexity=perplexity_score)
    tsne_data = model.fit_transform(X_train)
    tsne_df = pd.DataFrame(tsne_data,columns=['Dim1','Dim2'])
    tsne_df.to_csv(results_directory+'tsne_2comp_'+str(perplexity_score)+'perplexity'+'.csv',index=False)
    return None

def tsne_3comp(perplexity_score):
    print(perplexity_score)
    model = TSNE(n_components = 3, random_state = 0, perplexity=perplexity_score)
    tsne_data = model.fit_transform(X_train)
    tsne_df = pd.DataFrame(tsne_data,columns=['Dim1','Dim2', 'Dim3'])
    tsne_df.to_csv(results_directory+'tsne_3comp_'+str(perplexity_score)+'perplexity'+'.csv',index=False)
    return None

Parallel(n_jobs=4)(delayed(tsne_2comp)(perplexity_score) for perplexity_score in [10,20,30,40,50,60,70,80,90,100])
