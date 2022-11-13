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

X_std_imputed['damage_POPULATION_AFFECTED_y'] = X_std_imputed['damage_POPULATION_AFFECTED_y'].astype('category')
X_std_imputed['damage_POPULATION_AFFECTED_y'] = X_std_imputed['damage_POPULATION_AFFECTED_y'].cat.codes

X_std_imputed['damage_humanliveslost_y'] = X_std_imputed['damage_humanliveslost_y'].astype('category')
X_std_imputed['damage_humanliveslost_y'] = X_std_imputed['damage_humanliveslost_y'].cat.codes

X_std_imputed['response_inmatesinReliefCamps_y'] = X_std_imputed['response_inmatesinReliefCamps_y'].astype('category')
X_std_imputed['response_inmatesinReliefCamps_y'] = X_std_imputed['response_inmatesinReliefCamps_y'].cat.codes

X_std_imputed['Relief_cam_y'] = X_std_imputed['Relief_cam_y'].astype('category')
X_std_imputed['Relief_cam_y'] = X_std_imputed['Relief_cam_y'].cat.codes

X_std_imputed['Relief_dis_y'] = X_std_imputed['Relief_dis_y'].astype('category')
X_std_imputed['Relief_dis_y'] = X_std_imputed['Relief_dis_y'].cat.codes

X_std_imputed['Rice_y'] = X_std_imputed['Rice_y'].astype('category')
X_std_imputed['Rice_y'] = X_std_imputed['Rice_y'].cat.codes

X_std_imputed['Salt_y'] = X_std_imputed['Salt_y'].astype('category')
X_std_imputed['Salt_y'] = X_std_imputed['Salt_y'].cat.codes

X_std_imputed['Oil_y'] = X_std_imputed['Oil_y'].astype('category')
X_std_imputed['Oil_y'] = X_std_imputed['Oil_y'].cat.codes

X_std_imputed['Dal_y'] = X_std_imputed['Dal_y'].astype('category')
X_std_imputed['Dal_y'] = X_std_imputed['Dal_y'].cat.codes

X_std_imputed['Count_SDRF_y'] = X_std_imputed['Count_SDRF_y'].astype('category')
X_std_imputed['Count_SDRF_y'] = X_std_imputed['Count_SDRF_y'].cat.codes

X_std_imputed['Count_relief_y'] = X_std_imputed['Count_relief_y'].astype('category')
X_std_imputed['Count_relief_y'] = X_std_imputed['Count_relief_y'].cat.codes

X_std_imputed['Count_new_y'] = X_std_imputed['Count_new_y'].astype('category')
X_std_imputed['Count_new_y'] = X_std_imputed['Count_new_y'].cat.codes

X_std_imputed['Count_Erosion_y'] = X_std_imputed['Count_Erosion_y'].astype('category')
X_std_imputed['Count_Erosion_y'] = X_std_imputed['Count_Erosion_y'].cat.codes

X_std_imputed['Count_Road_y'] = X_std_imputed['Count_Road_y'].astype('category')
X_std_imputed['Count_Road_y'] = X_std_imputed['Count_Road_y'].cat.codes

X_std_imputed['Count_repair_y'] = X_std_imputed['Count_repair_y'].astype('category')
X_std_imputed['Count_repair_y'] = X_std_imputed['Count_repair_y'].cat.codes

X_std_imputed['Count_IM_y'] = X_std_imputed['Count_IM_y'].astype('category')
X_std_imputed['Count_IM_y'] = X_std_imputed['Count_IM_y'].cat.codes

X_std_imputed['Sum_SDRF_y'] = X_std_imputed['Sum_SDRF_y'].astype('category')
X_std_imputed['Sum_SDRF_y'] = X_std_imputed['Sum_SDRF_y'].cat.codes

X_std_imputed['Sum_relief_y'] = X_std_imputed['Count_relief_y'].astype('category')
X_std_imputed['Sum_relief_y'] = X_std_imputed['Sum_relief_y'].cat.codes

X_std_imputed['Sum_new_y'] = X_std_imputed['Sum_new_y'].astype('category')
X_std_imputed['Sum_new_y'] = X_std_imputed['Sum_new_y'].cat.codes

X_std_imputed['Sum_Erosion_y'] = X_std_imputed['Sum_Erosion_y'].astype('category')
X_std_imputed['Sum_Erosion_y'] = X_std_imputed['Sum_Erosion_y'].cat.codes

X_std_imputed['Sum_Roads_y'] = X_std_imputed['Sum_Roads_y'].astype('category')
X_std_imputed['Sum_Roads_y'] = X_std_imputed['Sum_Roads_y'].cat.codes

X_std_imputed['Sum_repair_y'] = X_std_imputed['Sum_repair_y'].astype('category')
X_std_imputed['Sum_repair_y'] = X_std_imputed['Sum_repair_y'].cat.codes

X_std_imputed['Sum_IM_y'] = X_std_imputed['Sum_IM_y'].astype('category')
X_std_imputed['Sum_IM_y'] = X_std_imputed['Sum_IM_y'].cat.codes

X_std_imputed['damage_POPULATION_AFFECTED_y'] = X_std_imputed['damage_POPULATION_AFFECTED_y'].astype('category')
X_std_imputed['damage_POPULATION_AFFECTED_y'] = X_std_imputed['damage_POPULATION_AFFECTED_y'].cat.codes

X_std_imputed['damage_humanliveslost_y'] = X_std_imputed['damage_humanliveslost_y'].astype('category')
X_std_imputed['damage_humanliveslost_y'] = X_std_imputed['damage_humanliveslost_y'].cat.codes

X_std_imputed['damage_animalsaffectedtotal_y'] = X_std_imputed['damage_animalsaffectedtotal_y'].astype('category')
X_std_imputed['damage_animalsaffectedtotal_y'] = X_std_imputed['damage_animalsaffectedtotal_y'].cat.codes

X_std_imputed['damage_animalsaffectedpoultry_y'] = X_std_imputed['damage_animalsaffectedpoultry_y'].astype('category')
X_std_imputed['damage_animalsaffectedpoultry_y'] = X_std_imputed['damage_animalsaffectedpoultry_y'].cat.codes

X_std_imputed['damage_animalsaffectedbig_y'] = X_std_imputed['damage_animalsaffectedbig_y'].astype('category')
X_std_imputed['damage_animalsaffectedbig_y'] = X_std_imputed['damage_animalsaffectedbig_y'].cat.codes

X_std_imputed['damage_animalsaffectedsmall_y'] = X_std_imputed['damage_animalsaffectedsmall_y'].astype('category')
X_std_imputed['damage_animalsaffectedsmall_y'] = X_std_imputed['damage_animalsaffectedsmall_y'].cat.codes

X_std_imputed['damage_animals_washed_total_y'] = X_std_imputed['damage_animals_washed_total_y'].astype('category')
X_std_imputed['Bridge_y'] = X_std_imputed['Bridge_y'].cat.codes

X_std_imputed['damage_animals_washed_poultry_y'] = X_std_imputed['damage_animals_washed_poultry_y'].astype('category')
X_std_imputed['damage_animals_washed_poultry_y'] = X_std_imputed['damage_animals_washed_poultry_y'].cat.codes

X_std_imputed['damage_animals_washed_big_y'] = X_std_imputed['damage_animals_washed_big_y'].astype('category')
X_std_imputed['damage_animals_washed_big_y'] = X_std_imputed['damage_animals_washed_big_y'].cat.codes

X_std_imputed['damage_animals_washed_small_y'] = X_std_imputed['damage_animals_washed_small_y'].astype('category')
X_std_imputed['damage_animals_washed_small_y'] = X_std_imputed['damage_animals_washed_small_y'].cat.codes

X_std_imputed['damage_Houses_damaged_fully_y'] = X_std_imputed['damage_Houses_damaged_fully_y'].astype('category')
X_std_imputed['damage_Houses_damaged_fully_y'] = X_std_imputed['damage_Houses_damaged_fully_y'].cat.codes

X_std_imputed['damage_Houses_damaged_partially_y'] = X_std_imputed['damage_Houses_damaged_partially_y'].astype('category')
X_std_imputed['damage_Houses_damaged_partially_y'] = X_std_imputed['damage_Houses_damaged_partially_y'].cat.codes

X_std_imputed['damage_croparea_AFFECTED_y'] = X_std_imputed['damage_croparea_AFFECTED_y'].astype('category')
X_std_imputed['damage_croparea_AFFECTED_y'] = X_std_imputed['damage_croparea_AFFECTED_y'].cat.codes

X_std_imputed['Embankment_y'] = X_std_imputed['Embankment_y'].astype('category')
X_std_imputed['Embankment_y'] = X_std_imputed['Embankment_y'].cat.codes

X_std_imputed['Other_y'] = X_std_imputed['Other_y'].astype('category')
X_std_imputed['Other_y'] = X_std_imputed['Other_y'].cat.codes

X_std_imputed['Road_y'] = X_std_imputed['Road_y'].astype('category')
X_std_imputed['Road_y'] = X_std_imputed['Road_y'].cat.codes

X_std_imputed['Bridge_y'] = X_std_imputed['Bridge_y'].astype('category')
X_std_imputed['Bridge_y'] = X_std_imputed['Bridge_y'].cat.codes


# Fit Model
def sem_results(optimiser):
    preparedness_model_spec_nozeroinflation = """
    # measurement model
    flood_proneness =~ Inundation + assam_dist_from_major_rivers_updated_3857 + sum + GCN250_ARCIII_average + strm_filled_slope_degrees + ndvi + srtm_filled_dem + gmted_drainage_density_without_1 + assam_lith_enc + assam_soil_enc + landuse_enc
    demography =~ ind_ppp_UNadj + aged + young + sexratio + percaay + deprived + nophone + noSanitation + nodrinkingWater + totLivestock
    infra_access =~ ndbi + proximity_hosptial_rd + proximity_embankment_rd + proximity_rail_rd + proximity_local_rd + proximity_arterial_rd

    flood_impact =~ damage_POPULATION_AFFECTED_y + damage_humanliveslost_y + damage_animalsaffectedtotal_y + damage_animalsaffectedpoultry_y + damage_animalsaffectedbig_y + damage_animalsaffectedsmall_y + damage_animals_washed_total_y + damage_animals_washed_poultry_y + damage_animals_washed_big_y + damage_animals_washed_small_y + damage_Houses_damaged_fully_y + damage_Houses_damaged_partially_y + damage_croparea_AFFECTED_y + Embankment_y + Other_y + Road_y + Bridge_y
    Preparedness =~ response_inmatesinReliefCamps_y + Relief_cam_y + Relief_dis_y + Rice_y + Salt_y + Oil_y + Dal_y +  Count_SDRF_y + Count_relief_y + Count_new_y + Count_Erosion_y + Count_Road_y + Count_repair_y + Count_IM_y + Sum_SDRF_y + Sum_relief_y + Sum_new_y + Sum_Erosion_y + Sum_Roads_y + Sum_repair_y + Sum_IM_y
    DEFINE(ordinal) assam_lith_enc assam_soil_enc landuse_enc response_inmatesinReliefCamps_y Relief_cam_y Relief_dis_y Rice_y Salt_y Oil_y Dal_y  Count_SDRF_y Count_relief_y Count_new_y Count_Erosion_y Count_Road_y Count_repair_y Count_IM_y Sum_SDRF_y Sum_relief_y Sum_new_y Sum_Erosion_y Sum_Roads_y Sum_repair_y Sum_IM_y damage_POPULATION_AFFECTED_y damage_humanliveslost_y damage_animalsaffectedtotal_y damage_animalsaffectedpoultry_y damage_animalsaffectedbig_y damage_animalsaffectedsmall_y damage_animals_washed_total_y damage_animals_washed_poultry_y damage_animals_washed_big_y damage_animals_washed_small_y damage_Houses_damaged_fully_y damage_Houses_damaged_partially_y damage_croparea_AFFECTED_y Embankment_y Other_y Road_y Bridge_y

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
    damage_animalsaffectedtotal_y ~~ damage_animalsaffectedbig_y
    damage_animalsaffectedtotal_y ~~ damage_animalsaffectedsmall_y
    damage_animalsaffectedpoultry_y ~~ damage_animalsaffectedtotal_y
    damage_animalsaffectedpoultry_y ~~ damage_animalsaffectedsmall_y
    damage_animalsaffectedbig_y ~~ damage_animalsaffectedsmall_y
    damage_animals_washed_total_y ~~ damage_animals_washed_poultry_y
    damage_animals_washed_total_y ~~ damage_animals_washed_big_y
    damage_animals_washed_total_y ~~ damage_animals_washed_small_y
    damage_animals_washed_small_y ~~ damage_animals_washed_big_y
    damage_Houses_damaged_fully_y ~~ damage_Houses_damaged_partially_y

    damage_animalsaffectedsmall_y ~~ Rice_y
    damage_animalsaffectedsmall_y ~~ Dal_y
    Rice_y ~~ Dal_y
    Rice_y ~~ Salt_y
    damage_animalsaffectedpoultry_y ~~ Rice_y
    damage_animalsaffectedpoultry_y ~~ Dal_y
    damage_animalsaffectedtotal_y ~~ Rice_y
    damage_animalsaffectedtotal_y ~~ Dal_y
    damage_animalsaffectedbig_y ~~ Rice_y
    damage_animalsaffectedbig_y ~~ Dal_y
    Relief_cam_y ~~ response_inmatesinReliefCamps_y
    response_inmatesinReliefCamps_y ~~ damage_Houses_damaged_fully_y
    response_inmatesinReliefCamps_y ~~ damage_Houses_damaged_partially_y

    Count_Road_y ~~ Count_IM_y
    Count_Road_y ~~ Count_new_y
    Count_Road_y ~~ Count_repair_y

    Sum_Roads_y ~~ Sum_new_y
    Sum_Roads_y ~~ Sum_repair_y

    Count_Erosion_y ~~ Sum_Erosion_y
    """
    model = semopy.Model(preparedness_model_spec_nozeroinflation)
    #model = semopy.ModelEffects(preparedness_model_spec_nozeroinflation)
    model.fit(X_std_imputed,
         obj=optimiser,
         solver='SLSQP',
        #groups=['revenue_ci_enc'] - use it for ModelEffects
              )

    coeff_df = model.inspect()
    coeff_df.to_csv(cwd+r'Results/'+'Estimates_NoZIF_'+optimiser+'_SLSQP_'+'.csv',index=False)
    stats = semopy.calc_stats(model)
    stats.to_csv(cwd+r'Results/'+'Stats_NoZIF_'+optimiser+'_SLSQP_'+'.csv',index=False)
    return None

Parallel(n_jobs=3)(delayed(sem_results)(optimiser) for optimiser in ['WLS','MLW','DWLS','FIML','ULS','GLS'])