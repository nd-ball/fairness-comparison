import pandas as pd
from fairness.data.objects.Data import Data

class CovidCases(Data):

    def __init__(self):
        Data.__init__(self)

        self.dataset_name = 'covidcases'
        self.class_attr = 'Y'
        self.positive_class_val = 1
        self.sensitive_attrs = [
                                'aboveAverageBlack',
                                'aboveNatlAverage', 
                                'majorityFemale',
                                'aboveAverageOver65'
                                ]
        self.privileged_class_names = ['No', 'Yes', 'No', 'No']
        self.categorical_features = ['Stabr', 
                                    'majorityFemaleBin', 
                                    'aboveAverageOver65Bin', 
                                    'aboveStateAverageBin', 
                                    'aboveNatlAverageBin', 
                                    #'aboveAverageBlackBin',
                                    ]
        self.features_to_keep = [
                                # as of 9-21-2020:
                                # trend data
                                #'X_1', 'X_2', 'X_3', 'X_4', 
                                # only using last 4 days for now
                                'X_5', 'X_6', 'X_7', 'X_8', 

                                # features that stay in all models
                                'Stabr', # state abbrev
                                'Med_HH_Income_Percent_of_State_Total_2018',
                                'percent65over',
                                'percentWomen',
                                'percentBlack',

                                # the below will be added/removed to simulate de-biasing
                                # is the county majority female?
                                'majorityFemaleBin',
                                # is the county above average old?
                                'aboveAverageOver65Bin',
                                # is the county below average on income?
                                'aboveStateAverageBin',
                                'aboveNatlAverageBin',
                                # is the county above average black?
                                #'aboveAverageBlackBin',
                                # protected attributes
                                'aboveAverageBlack',
                                'majorityFemale',
                                'aboveAverageOver65',
                                'aboveNatlAverage',
                                # prediction value
                                'Y',
                                ]
                                # next can add H_MALE and H_FEMALE        
        self.missing_val_indicators = []

    def data_specific_processing(self, dataframe):
        # adding a derived sex attribute based on personal_status
        # convert bools to strings
        booleanDictionary = {True: 'Yes', False: 'No'}

        # uncomment as these are added as protected attributes
        dataframe['aboveAverageBlack'] = dataframe['aboveAverageBlack'].map(booleanDictionary)
        dataframe['majorityFemale'] = dataframe['majorityFemale'].map(booleanDictionary)
        dataframe['aboveAverageOver65'] = dataframe['aboveAverageOver65'].map(booleanDictionary)
        dataframe['aboveNatlAverage'] = dataframe['aboveNatlAverage'].map(booleanDictionary)
        
        return dataframe