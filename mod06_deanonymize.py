import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    
    merged = anon_df.merge(aux_df, on=["age", "gender", "zip3"], how="inner")
    candidate_counts = merged.groupby("anon_id")["name"].transform("size")
    unique = merged[candidate_counts == 1]
    matches = (
        unique[["anon_id", "name"]]
        .rename(columns={"name": "matched_name"})
        .reset_index(drop=True)
    )
    return matches
    # raise NotImplementedError


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0
    return matches_df["anon_id"].nunique() / anon_df["anon_id"].nunique()

    # raise NotImplementedError
