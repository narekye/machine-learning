def preprocess_features(california_housing_dataframe):
    """Prepares input features from California housing data set.

        Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
        Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
    selected_features = california_housing_dataframe[["latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income"]]
     
    processed_features = selected_features.copy()
    processed_features["rooms_per_person"] = (california_housing_dataframe["total_rooms"] / california_housing_dataframe["population"])

    return processed_features

