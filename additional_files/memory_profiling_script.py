from memory_profiler import profile
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import time


def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Hour'] = df['Time'].dt.hour

    df = df.drop(['Invoice ID', 'Date', 'Year', 'Time'], axis=1)

    return df


def prepare_apriori_data(df):
    apriori_df = df.copy()
    apriori_df = apriori_df.drop(['gross margin percentage'], axis=1)
    exploded_list = apriori_df['Product line'].str.split(
        ' and ').explode().tolist()
    apriori_df = pd.DataFrame(exploded_list, columns=['Product'])
    apriori_df['Count'] = apriori_df.groupby(
        'Product')['Product'].transform('count')
    apriori_df = apriori_df.drop_duplicates().reset_index(drop=True)
    transactions_list = apriori_df.groupby('Count')['Product'].agg(
        list).reset_index(name='Transactions')['Transactions'].tolist()
    tx = TransactionEncoder()
    encoded_array = tx.fit(transactions_list).transform(transactions_list)
    encoded_df = pd.DataFrame(encoded_array, columns=tx.columns_, dtype=int)
    return encoded_df


@profile
def apriori_algorithm(encoded_df):
    start_mem_apriori = time.time()
    frequent_itemsets_apriori = apriori(
        encoded_df, min_support=0.01, use_colnames=True)
    rules_apriori = association_rules(
        frequent_itemsets_apriori, metric="confidence", min_threshold=0.01)
    end_mem_apriori = time.time()
    apriori1_mem = end_mem_apriori - start_mem_apriori
    print(apriori1_mem)


@profile
def fp_growth_algorithm(encoded_df):
    start_time_fp_mem = time.time()
    frequent_itemsets_fp = fpgrowth(
        encoded_df, min_support=0.01, use_colnames=True)
    rules_fp = association_rules(
        frequent_itemsets_fp, metric='confidence', min_threshold=0.1)
    end_time_fp_mem = time.time()
    fp1_mem = end_time_fp_mem - start_time_fp_mem
    print(fp1_mem)


if __name__ == "__main__":
    df = pd.read_csv(
        r"C:\Users\hp\Desktop\sem5\DM\proj\supermarket_sales - Sheet1.csv")
    df = preprocess_data(df)

    apriori_df = prepare_apriori_data(df)

    apriori_algorithm(apriori_df)
    fp_growth_algorithm(apriori_df)
