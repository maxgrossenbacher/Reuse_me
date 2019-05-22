import pandas as pd
import numpy as np

def add_train_validation_split(df, split_by_grouping=[], validation_name='split', validation_size=0.25, test_size=0.1, random=42):
    """
        Adds a validation column specifies 'Train' or 'Validation', by randomly splitting with

        Args:
            df (pd.DataFrame): original data frame to add in split column
            test_size (float): fraction of the validation set size
            split_by_grouping (str list): column names to group by for the split (all rows in the same group will be in either Train or Validation sets uniquely)
            validation_name (str): column name to put 'Train' and 'Validation' labels
        Returns:
            df (pd.DataFrame): with
    """

    # check to see if validation column already exists, if it does and is a valid split then keep it
    split_randomly=True
    if validation_name and validation_name in df.columns.values:
        if ((len(df[validation_name].values) == 3 and 'Test' in df[validation_name].values) or (len(df[validation_name].values) == 2)) \
            and 'Train' in df[validation_name].values and 'Validation' in df[validation_name].values:
            split_randomly = False
        else:
            #write_log('validation column specified broken '+validation_name,legend)
            #write_log('defaulting to random split',legend)
            df.drop(validation_name, axis=1, inplace=True)

    #fix the test set!!

    # do the split
    if split_randomly:
        #write_log('splitting training and validation set randomly',legend)
        # split by group
        from sklearn.model_selection import train_test_split

        if split_by_grouping:
            df_split = df[split_by_grouping].groupby(split_by_grouping).aggregate('count').reset_index()
            x1, x2, idx1, idx2 = train_test_split(df_split, df_split.index, test_size=validation_size, random_state=random)
            df_split[validation_name] = -1
            df_split.ix[idx1,validation_name] = 'Train'
            df_split.ix[idx2,validation_name] = 'Validation'


            if test_size > 0:
                df_testsplit = df_split[df_split[validation_name] == 'Train'].groupby(split_by_grouping).aggregate('count').reset_index()
                _, _, idx3, idx4 = train_test_split(df_testsplit, df_testsplit.index, test_size=test_size/(1-validation_size), random_state=random)

                # train and test sections
                df_testsplit.ix[idx3,validation_name] = 'Train'
                df_testsplit.ix[idx4,validation_name] = 'Test'

                # delete the training from original split
                df_split = df_split[df_split[validation_name]!='Train']

                # concat it back
                df_split = pd.concat([df_testsplit, df_split])


            df = pd.merge(df,df_split,on=split_by_grouping)
        else:
            x1, x2, idx1, idx2 = train_test_split(df, df.index, test_size=validation_size, random_state=random)
            df[validation_name] = -1
            df.ix[idx1,validation_name] = 'Train'
            df.ix[idx2,validation_name] = 'Validation'

            if test_size > 0:
                df_testsplit = df[df[validation_name] == 'Train']
                _, _, idx3, idx4 = train_test_split(df_testsplit, df_testsplit.index, test_size=test_size/(1-validation_size), random_state=random)

                # train and test sections
                df_testsplit.ix[idx3,validation_name] = 'Train'
                df_testsplit.ix[idx4,validation_name] = 'Test'

                # delete the training from original split
                df = df[df[validation_name]!='Train']

                # concat it back
                df = pd.concat([df, df_testsplit])


    return df