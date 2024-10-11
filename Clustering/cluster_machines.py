import pandas as pd
import Clustering

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here

    df = pd.read_csv('df/df_from10-04on.csv')
    
    df_nkmeans = pd.read_csv('Results/Clustering results-windows_clustering_fresh_data_now_normalized_correct_manner/merged_df_3kmeans.csv')
    c = Clustering(df=df,os='windows',nclusters=3,note='windows_clustering_fresh_data_not_normalized_at_all', norm='none')
    c.gen_formatted_dataset(2000)
    c.clustering(njobs=-1)
    c.save_clustering_results()
    c.plot_cluster()
    c.plot_per_group()

    # c.initialise(df_nkmeans,df)
    c.plot_all_machines_rolling_mean()
    c.plot_one_doc_per_cluster()
    c.gen_report()
    c.gen_csv_report()
