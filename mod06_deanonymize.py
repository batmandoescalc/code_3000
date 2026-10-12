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

    quasi_identifiers = [c for c in anon_df.columns if c in aux_df.columns and c not in ("anon_id", "name")]
    if not quasi_identifiers:
        raise ValueError("No shared quasi-identifier columns found.")

    merged = anon_df.merge(aux_df, on=quasi_identifiers, how="inner")

    counts = merged.groupby("anon_id")["name"].transform("size")
    unique = merged[counts == 1]

    return(unique.rename(columns={"name": "matched_name"}).reset_index(drop=True))
        
    #raise NotImplementedError


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0
    return matches_df["anon_id"].nunique() / len(anon_df)
    #raise NotImplementedError
