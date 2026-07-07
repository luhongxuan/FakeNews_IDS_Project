from pheme_retrospective_analysis import run_full_analysis

if __name__ == "__main__":
    desc_stats, test_results = run_full_analysis(df, label_col="is_rumour")