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

# Fit Model
def sem_results(optimiser):
    preparedness_model_spec_nozeroinflation = """
    # measurement model
    flood_proneness =~ Inundation + assam_dist_from_major_rivers_updated_3857 + sum + GCN250_ARCIII_average + strm_filled_slope_degrees + ndvi + srtm_filled_dem + gmted_drainage_density_without_1 + assam_lith_enc + assam_soil_enc + landuse_enc
    demography =~ ind_ppp_UNadj + aged + young + sexratio + percaay + deprived + nophone + noSanitation + nodrinkingWater + totLivestock
    infra_access =~ ndbi + proximity_hosptial_rd + proximity_embankment_rd + proximity_rail_rd + proximity_local_rd + proximity_arterial_rd

    flood_impact =~ damage_POPULATION_AFFECTED_x + damage_humanliveslost_x + damage_animalsaffectedtotal_x + damage_animalsaffectedpoultry_x + damage_animalsaffectedbig_x + damage_animalsaffectedsmall_x + damage_animals_washed_total_x + damage_animals_washed_poultry_x + damage_animals_washed_big_x + damage_animals_washed_small_x + damage_Houses_damaged_fully_x + damage_Houses_damaged_partially_x + damage_croparea_AFFECTED_x + Embankment_x + Other_x + Road_x + Bridge_x
    Preparedness =~ response_inmatesinReliefCamps_x + Relief_cam_x + Relief_dis_x + Rice_x + Salt_x + Oil_x + Dal_x +  Count_SDRF_x + Count_relief_x + Count_new_x + Count_Erosion_x + Count_Road_x + Count_repair_x + Count_IM_x + Sum_SDRF_x + Sum_relief_x + Sum_new_x + Sum_Erosion_x + Sum_Roads_x + Sum_repair_x + Sum_IM_x
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
    damage_animalsaffectedtotal_x ~~ damage_animalsaffectedbig_x
    damage_animalsaffectedtotal_x ~~ damage_animalsaffectedsmall_x
    damage_animalsaffectedpoultry_x ~~ damage_animalsaffectedtotal_x
    damage_animalsaffectedpoultry_x ~~ damage_animalsaffectedsmall_x
    damage_animalsaffectedbig_x ~~ damage_animalsaffectedsmall_x
    damage_animals_washed_total_x ~~ damage_animals_washed_poultry_x
    damage_animals_washed_total_x ~~ damage_animals_washed_big_x
    damage_animals_washed_total_x ~~ damage_animals_washed_small_x
    damage_animals_washed_small_x ~~ damage_animals_washed_big_x
    damage_Houses_damaged_fully_x ~~ damage_Houses_damaged_partially_x

    damage_animalsaffectedsmall_x ~~ Rice_x
    damage_animalsaffectedsmall_x ~~ Dal_x
    Rice_x ~~ Dal_x
    Rice_x ~~ Salt_x
    damage_animalsaffectedpoultry_x ~~ Rice_x
    damage_animalsaffectedpoultry_x ~~ Dal_x
    damage_animalsaffectedtotal_x ~~ Rice_x
    damage_animalsaffectedtotal_x ~~ Dal_x
    damage_animalsaffectedbig_x ~~ Rice_x
    damage_animalsaffectedbig_x ~~ Dal_x
    Relief_cam_x ~~ response_inmatesinReliefCamps_x
    response_inmatesinReliefCamps_x ~~ damage_Houses_damaged_fully_x
    response_inmatesinReliefCamps_x ~~ damage_Houses_damaged_partially_x

    Count_Road_x ~~ Count_IM_x
    Count_Road_x ~~ Count_new_x
    Count_Road_x ~~ Count_repair_x

    Sum_Roads_x ~~ Sum_new_x
    Sum_Roads_x ~~ Sum_repair_x

    Count_Erosion_x ~~ Sum_Erosion_x
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