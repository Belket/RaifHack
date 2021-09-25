import statistics as sts
from copy import deepcopy
import numpy as np
import pandas as pd


class NanFighter:

    @staticmethod
    def floor_fighter(df):

        def determine_hight_category(x):
            '''
            1 - floors from 1 to 5
            2 - floors from 6 to 10
            3 - floors from 10 to 30
            4 - floors from 30 to 90
            '''
            try:
                x = float(x)
            except:
                return 1
            else:
                if 1 <= x < 5:
                    return 1
                elif 5 <= x < 10:
                    return 2
                elif 10 <= x < 30:
                    return 3
                else:
                    return 4

        def is_wholesale(x):
            x = str(x)
            if len(x.split(',')) >= 2:
                return 1
            return 0

        df['wholesale'] = df['floor'].apply(is_wholesale)
        df['floor_hight_category'] = df['floor'].apply(determine_hight_category)
        df.drop(['floor'], inplace=True, axis=1)
        return df

    @staticmethod
    def osm_city_nearest_population_fighter(df):

        def determine_osm_city_nearest_population(observation):
            if pd.isnull(observation.osm_city_nearest_population):
                value = aggregation_osm_city_nearest_population.loc[observation.region]
                observation.osm_city_nearest_population = value
            return observation.osm_city_nearest_population

        aggregation_osm_city_nearest_population = df.groupby(by=['region']). \
            aggregate({'osm_city_nearest_population': np.mean}). \
            osm_city_nearest_population

        df['osm_city_nearest_population'] = df[['osm_city_nearest_population', 'region']]. \
            apply(determine_osm_city_nearest_population, axis=1)

        return df

    @staticmethod
    def reform_house_population_500_fighter(df):
        aggregation_reform_house_population = df.groupby(by=['city']).aggregate({'reform_house_population_500': np.mean})
        data_for_reform_house_population_500 = aggregation_reform_house_population.reform_house_population_500
        data_for_reform_house_population_500_mode = sts.mode(data_for_reform_house_population_500)

        def determine_reform_house_population_500(observation):
            if pd.isnull(observation.reform_house_population_500):
                value = data_for_reform_house_population_500.loc[observation.city]
                if pd.isnull(value):
                    value = data_for_reform_house_population_500_mode
                observation.reform_house_population_500 = value
            return observation.reform_house_population_500

        df['reform_house_population_500'] = df[['city', 'reform_house_population_500']]. \
            apply(determine_reform_house_population_500, axis=1)
        return df

    @staticmethod
    def reform_house_population_1000_fighter(df):
        aggregation_reform_house_population = df.groupby(by=['city']).aggregate({'reform_house_population_1000': np.mean})
        data_for_reform_house_population_1000 = aggregation_reform_house_population.reform_house_population_1000
        data_for_reform_house_population_1000_mode = sts.mode(data_for_reform_house_population_1000)

        def determine_reform_house_population_1000(observation):
            if pd.isnull(observation.reform_house_population_1000):
                value = data_for_reform_house_population_1000.loc[observation.city]
                if pd.isnull(value):
                    value = data_for_reform_house_population_1000_mode
                observation.reform_house_population_1000 = value
            return observation.reform_house_population_1000

        df['reform_house_population_1000'] = df[['city', 'reform_house_population_1000']].apply(determine_reform_house_population_1000, axis=1)
        return df

    @staticmethod
    def reform_mean_floor_count_500_fighter(df):
        aggregation_reform_mean_floor_count = df.groupby(by=['city']).aggregate({'reform_mean_floor_count_500': np.mean})
        data_for_reform_mean_floor_count_500 = aggregation_reform_mean_floor_count.reform_mean_floor_count_500
        data_for_reform_mean_floor_count_500_mode = sts.mode(data_for_reform_mean_floor_count_500)

        def determine_mean_floor_count_500(observation):
            if pd.isnull(observation.reform_mean_floor_count_500):
                value = data_for_reform_mean_floor_count_500.loc[observation.city]
                if pd.isnull(value):
                    value = data_for_reform_mean_floor_count_500_mode
                observation.reform_mean_floor_count_500 = value
            return observation.reform_mean_floor_count_500

        df['reform_mean_floor_count_500'] = df[['city', 'reform_mean_floor_count_500']]. \
            apply(determine_mean_floor_count_500, axis=1)
        return df

    @staticmethod
    def reform_mean_floor_count_1000_fighter(df):
        aggregation_reform_mean_floor_count = df.groupby(by=['city']).aggregate({'reform_mean_floor_count_1000': np.mean})
        data_for_reform_mean_floor_count_1000 = aggregation_reform_mean_floor_count.reform_mean_floor_count_1000
        data_for_reform_mean_floor_count_1000_mode = sts.mode(data_for_reform_mean_floor_count_1000)

        def determine_mean_floor_count_1000(observation):
            if pd.isnull(observation.reform_mean_floor_count_1000):
                value = data_for_reform_mean_floor_count_1000.loc[observation.city]
                if pd.isnull(value):
                    value = data_for_reform_mean_floor_count_1000_mode
                observation.reform_mean_floor_count_1000 = value
            return observation.reform_mean_floor_count_1000

        df['reform_mean_floor_count_1000'] = df[['city', 'reform_mean_floor_count_1000']]. \
            apply(determine_mean_floor_count_1000, axis=1)
        return df

    @staticmethod
    def street_fighter(df):
        def fill_street(x):
            if x.count() <= 0:
                return np.nan
            return x.value_counts().index[0]

        df['street'] = df.groupby('city')['street'].transform(fill_street)
        df['street'] = df['street'].fillna(df['street'].value_counts().idxmax())
        return df

    @staticmethod
    def reform_mean_year_building_500_fighter(df):
        aggregation_year_mean_city = df.groupby(by=['city']).aggregate({'reform_mean_year_building_500': np.mean})
        aggregation_year_mean_region = df.groupby(by=['region']).aggregate({'reform_mean_year_building_500': np.mean})
        data_for_year_mean_city_500 = aggregation_year_mean_city.reform_mean_year_building_500
        data_for_year_mean_region_500 = aggregation_year_mean_region.reform_mean_year_building_500
        data_for_year_mean_street_500_mode = sts.mode(data_for_year_mean_region_500)

        def determine_mean_year_500(observation):
            if pd.isnull(observation.reform_mean_year_building_500):
                street, city, region = observation.street, observation.city, observation.region
                value = data_for_year_mean_city_500.loc[city]
                if pd.isnull(value):
                    value = data_for_year_mean_region_500.loc[region]
                    if pd.isnull(value):
                        value = data_for_year_mean_street_500_mode
                observation.reform_mean_year_building_500 = value
            return observation.reform_mean_year_building_500

        df['reform_mean_year_building_500'] = df[['street', 'city', 'region', 'reform_mean_year_building_500']]. \
            apply(determine_mean_year_500, axis=1)
        return df

    @staticmethod
    def reform_mean_year_building_1000_fighter(df):
        aggregation_year_mean_city = df.groupby(by=['city']).aggregate({'reform_mean_year_building_1000': np.mean})
        aggregation_year_mean_region = df.groupby(by=['region']).aggregate({'reform_mean_year_building_1000': np.mean})
        data_for_year_mean_city_1000 = aggregation_year_mean_city.reform_mean_year_building_1000
        data_for_year_mean_region_1000 = aggregation_year_mean_region.reform_mean_year_building_1000
        data_for_year_mean_street_1000_mode = sts.mode(data_for_year_mean_region_1000)

        def determine_mean_year_1000(observation):
            if pd.isnull(observation.reform_mean_year_building_1000):
                street, city, region = observation.street, observation.city, observation.region
                value = data_for_year_mean_city_1000.loc[city]
                if pd.isnull(value):
                    value = data_for_year_mean_region_1000.loc[region]
                    if pd.isnull(value):
                        value = data_for_year_mean_street_1000_mode
                observation.reform_mean_year_building_1000 = value
            return observation.reform_mean_year_building_1000

        df['reform_mean_year_building_1000'] = df[['street', 'city', 'region', 'reform_mean_year_building_1000']]. \
            apply(determine_mean_year_1000, axis=1)
        return df

    @staticmethod
    def fight(df):
        result_df = deepcopy(df)
        for feature in df.columns:
            fighter_name = feature + "_fighter"
            fighter = getattr(NanFighter, fighter_name, None)
            if fighter:
                result_df = fighter(result_df)
        return result_df

