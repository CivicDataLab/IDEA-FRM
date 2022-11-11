import semopy
import pandas as pd
import numpy as np

# Read the prepared dataset of measured variables (Xs) after Random Forest Imputation of Null Values, Standardisation and Removal of Anomalies
X_std_imputed = pd.read_csv('D:/Projects/IDEA-FRM/InputData/SEM_X_Imputed_RemovedAnomalies.csv').drop(['x','y'],axis=1)
cat_variables = ['assam_soil_loam, silt loam, silt, sandy loam',
       'assam_soil_loamy sand, sand',
       'assam_soil_rocky, other non-soil categories',
       'assam_lith_neogene sedimentary rock',
       'assam_lith_paleogene sedementary rock', 'assam_lith_paleozoic rock',
       'assam_lith_quaternary sediments',
       'assam_lith_tertiary sedimentary rocks',
       'assam_lith_undeveloped precambrian rock', 'assam_lith_water',
       'land use_built', 'land use_range land', 'land use_vegetation',
       'land use_water']
X_std_imputed_cat = X_std_imputed[cat_variables]
X_std_imputed_con = X_std_imputed.drop(cat_variables,axis=1)

model_spec1 = """
# measurement model
flood_intensity =~ assam_dist_from_major_rivers_updated_3857 + sum + GCN250_ARCIII_average + strm_filled_slope_degrees + ndvi + srtm_filled_dem + gmted_drainage_density_without_1
demography =~ ind_ppp_UNadj + aged + young + sexratio + percaay + deprived + nophone + noSanitation + nodrinkingWater + totLivestock
infra_access =~ ndbi + proximity_hosptial_rd + proximity_embankment_rd + proximity_rail_rd + proximity_local_rd + proximity_arterial_rd
flood_impact =~ damage_POPULATION_AFFECTED + damage_humanliveslost + damage_animalsaffectedtotal + damage_animalsaffectedpoultry + damage_animalsaffectedbig + damage_animalsaffectedsmall + damage_animals_washed_total + damage_animals_washed_poultry + damage_animals_washed_big + damage_animals_washed_small + damage_Houses_damaged_fully + damage_Houses_damaged_partially + damage_croparea_AFFECTED + Embankment + Other + Road + Bridge

# regressions
flood_impact ~ flood_intensity + demography + infra_access

# residual correlations
ind_ppp_UNadj ~~ aged
ind_ppp_UNadj ~~ young
ind_ppp_UNadj ~~ percaay
ind_ppp_UNadj ~~ deprived
ind_ppp_UNadj ~~ nophone
ind_ppp_UNadj ~~ noSanitation
ind_ppp_UNadj ~~ totLivestock
ind_ppp_UNadj ~~ nodrinkingWater
aged ~~ young
aged ~~ percaay
aged ~~ deprived
aged ~~ nophone
aged ~~ noSanitation
aged ~~ totLivestock
young ~~ percaay
young ~~ deprived
young ~~ nophone
young ~~ noSanitation
young ~~ totLivestock
percaay ~~ deprived
percaay ~~ nophone
percaay ~~ noSanitation
percaay ~~ totLivestock
deprived ~~ nophone
deprived ~~ noSanitation
deprived ~~ totLivestock
nophone ~~ noSanitation
nophone ~~ totLivestock
noSanitation ~~ nodrinkingWater
proximity_hosptial_rd ~~ proximity_local_rd
proximity_hosptial_rd ~~ proximity_arterial_rd
proximity_hosptial_rd ~~ gmted_drainage_density_without_1
proximity_hosptial_rd ~~ srtm_filled_dem
proximity_local_rd ~~ proximity_arterial_rd
ndbi ~~ ndvi
gmted_drainage_density_without_1 ~~ srtm_filled_dem
gmted_drainage_density_without_1 ~~ GCN250_ARCIII_average
GCN250_ARCIII_average ~~ ndvi
damage_animalsaffectedtotal ~~ damage_animalsaffectedpoultry
damage_animalsaffectedtotal ~~ damage_animalsaffectedbig
damage_animalsaffectedtotal ~~ damage_animalsaffectedsmall
damage_animalsaffectedpoultry ~~ damage_animalsaffectedtotal
damage_animalsaffectedpoultry ~~ damage_animalsaffectedsmall
damage_animalsaffectedbig ~~ damage_animalsaffectedsmall
damage_animals_washed_total ~~ damage_animals_washed_poultry
damage_animals_washed_total ~~ damage_animals_washed_big
damage_animals_washed_total ~~ damage_animals_washed_small
damage_animals_washed_small ~~ damage_animals_washed_big
damage_Houses_damaged_fully ~~ damage_Houses_damaged_partially
"""

# Model specification -1
model = semopy.Model(model_spec1)

# Fit Model
model.fit(X_std_imputed.sample(5000),
         obj='WLS',
         solver='SLSQP')

coeff_df = model.inspect()
coeff_df.to_csv('Results/Model1_Estimates.csv',index=False)

stats = semopy.calc_stats(model)
stats.to_csv('Results/Model1_Stats.csv')