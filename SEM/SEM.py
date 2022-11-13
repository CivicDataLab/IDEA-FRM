import semopy
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed

cwd = os.getcwd()
global data_directory
data_directory = cwd.replace(r'SEM', '') + r'InputData/'

# Read the prepared dataset of measured variables (Xs) after Random Forest Imputation of Null Values, Standardisation and Removal of Anomalies
global X_std_imputed
X_std_imputed = pd.read_csv(data_directory + r'SEM_X_Imputed_RemovedAnomalies3.csv')

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

# Fit Model
def sem_results(optimiser):
    preparedness_model_spec_nozeroinflation = """
    # measurement model
    flood_proneness =~ Inundation + assam_dist_from_major_rivers_updated_3857 + sum + GCN250_ARCIII_average + strm_filled_slope_degrees + ndvi + srtm_filled_dem + gmted_drainage_density_without_1 + assam_lith_enc + assam_soil_enc + landuse_enc
    demography =~ ind_ppp_UNadj + aged + young + sexratio + percaay + deprived + nophone + noSanitation + nodrinkingWater + totLivestock
    infra_access =~ ndbi + proximity_hosptial_rd + proximity_embankment_rd + proximity_rail_rd + proximity_local_rd + proximity_arterial_rd

    flood_impact =~ damage_POPULATION_AFFECTED + damage_humanliveslost + damage_animalsaffectedtotal + damage_animalsaffectedpoultry + damage_animalsaffectedbig + damage_animalsaffectedsmall + damage_animals_washed_total + damage_animals_washed_poultry + damage_animals_washed_big + damage_animals_washed_small + damage_Houses_damaged_fully + damage_Houses_damaged_partially + damage_croparea_AFFECTED + Embankment + Other + Road + Bridge
    Preparedness =~ response_inmatesinReliefCamps + Relief_cam + Relief_dis + Rice + Salt + Oil + Dal +  Count_SDRF + Count_relief + Count_new + Count_Erosion + Count_Road + Count_repair + Count_IM + Sum_SDRF + Sum_relief + Sum_new + Sum_Erosion + Sum_Roads + Sum_repair + Sum_IM
    DEFINE(ordinal) assam_lith_enc assam_soil_enc landuse_enc

    # regressions
    flood_impact ~ flood_proneness + demography + infra_access
    Preparedness ~ flood_proneness + demography + infra_access + flood_impact

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

    damage_animalsaffectedsmall ~~ Rice
    damage_animalsaffectedsmall ~~ Dal
    Rice ~~ Dal
    Rice ~~ Salt
    damage_animalsaffectedpoultry ~~ Rice
    damage_animalsaffectedpoultry ~~ Dal
    damage_animalsaffectedtotal ~~ Rice
    damage_animalsaffectedtotal ~~ Dal
    damage_animalsaffectedbig ~~ Rice
    damage_animalsaffectedbig ~~ Dal
    Relief_cam ~~ response_inmatesinReliefCamps
    response_inmatesinReliefCamps ~~ damage_Houses_damaged_fully
    response_inmatesinReliefCamps ~~ damage_Houses_damaged_partially

    Count_Road ~~ Count_IM
    Count_Road ~~ Count_new
    Count_Road ~~ Count_repair


    Sum_Roads ~~ Sum_new
    Sum_Roads ~~ Sum_repair

    Count_Erosion ~~ Sum_Erosion
    """


    model = semopy.Model(preparedness_model_spec_nozeroinflation)
    model.fit(X_std_imputed,
         obj=optimiser,
         solver='SLSQP')

    coeff_df = model.inspect()
    coeff_df.to_csv(cwd+r'Results/'+'Estimates_NoZIF_'+optimiser+'_SLSQP_'+'.csv',index=False)
    stats = semopy.calc_stats(model)
    stats.to_csv(cwd+r'Results/'+'Stats_NoZIF_'+optimiser+'_SLSQP_'+'.csv',index=False)
    return None

Parallel(n_jobs=3)(delayed(sem_results)(optimiser) for optimiser in ['WLS','MLW','DWLS','FIML','ULS','GLS'])