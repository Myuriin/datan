# Auto-generated helper functions from notebook

# Let's make a function code that will "clean the data"
def clean(data,selected_col_list=None):
  data.columns = data.columns.str.strip() # to clean all the weird spacings

  if selected_col_list is not None:
        data = data[[col for col in selected_col_list if col in data.columns]] # get all the selected data we want. if there is none, it will just clean the spacing of the data

  display(data.head()) # show some of the data
  display(data.info()) # show info of the data

  return data

# Create histogram for optical_gap_comp and optical_gap_exp

# Create the function first:
def plot_histograms(df, columns, bins=30, figsize=(12, 5), layout='horizontal', titles=None):
    """
    Plot separate histograms for each specified column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list of str): List of column names to plot.
    - bins (int): Number of histogram bins. Default is 30.
    - figsize (tuple): Size of the entire figure. Default is (12, 5).
    - layout (str): 'horizontal' for side-by-side, 'vertical' for stacked.
    - titles (list of str): Optional list of titles for each subplot.
    """
    n = len(columns)

    if layout == 'horizontal':
        fig, axes = plt.subplots(1, n, figsize=figsize)
    else:
        fig, axes = plt.subplots(n, 1, figsize=figsize)

    if n == 1:
        axes = [axes]  # Ensure axes is iterable

    for i, col in enumerate(columns):
        if col in df.columns:
            axes[i].hist(df[col].dropna(), bins=bins, color='skyblue', edgecolor='black')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True)
            if titles and i < len(titles):
                axes[i].set_title(titles[i])
            else:
                axes[i].set_title(f'Histogram of {col}')
        else:
            axes[i].text(0.5, 0.5, f"Column '{col}' not found", ha='center', va='center')
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# Create histogram
plot_histograms(df, ['optical_gap_exp', 'optical_gap_comp'], titles=['Experimental Optical Gap', 'Computed Optical Gap'])

# Create a function to get molecule descriptors
from rdkit.Chem import AddHs

def get_descriptors(df, mol_column='mol_obj'):
    """
    Compute 200 RDKit molecular descriptors using molecules with explicit hydrogens.

    Parameters:
    - df: pandas DataFrame containing either RDKit Mol objects or SMILES strings.
    - mol_column: the column name with RDKit Mol objects (default: 'mol_obj').

    Returns:
    - DataFrame with molecular descriptors appended.
    """
    # Handle missing mol_column by generating it from 'SMILES'
    if mol_column not in df.columns:
        if 'SMILES' not in df.columns:
            raise ValueError("DataFrame must have either a 'mol_obj' column or a 'SMILES' column.")
        df = df.copy()  # Avoid changing original
        df['mol_obj'] = df['SMILES'].apply(Chem.MolFromSmiles)
        mol_column = 'mol_obj'  # Update column name

    # Prepare descriptor calculator
    descriptor_names = [desc[0] for desc in Descriptors.descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    # Function to compute descriptors with explicit Hs
    def calc_descriptors_with_Hs(mol):
        if mol is None:
            return [None] * len(descriptor_names)
        try:
            mol_with_H = AddHs(mol)
            return list(calculator.CalcDescriptors(mol_with_H))

        except:
            return [None] * len(descriptor_names)

    # Apply to molecule column
    descriptor_values = df[mol_column].apply(calc_descriptors_with_Hs)
    descriptor_df = pd.DataFrame(descriptor_values.tolist(), columns=descriptor_names)

    # Combine and return
    result_df = pd.concat([df.reset_index(drop=True), descriptor_df], axis=1)
    return result_df

# check if any column (possibly descriptors only have 1 unique value -> we can delete it because its not needed)
# create a fuction to check which column has unique values
def check_unique_value(df):
  one_value_cols_sum = 0
  for col in df.columns:
    if df[col].nunique() == 1:
      print(f"Column '{col}' has only one unique value.")
      one_value_cols_sum += 1
  if one_value_cols_sum == 0:
    print("No column has only one unique value.")
  print()
  print(f"Total number of columns with only one unique value: {one_value_cols_sum}")

check_unique_value(df_with_descriptors)

# create function to drop the columns that have only one unique value
def drop_unique_value(df):
  one_value_cols_sum = 0
  for col in df.columns:
    if df[col].nunique() == 1:
      # drop the column
      df = df.drop(columns=[col])
      one_value_cols_sum += 1
  if one_value_cols_sum == 0:
    print("No column has only one unique value.")
  print()
  print(f"Total number of columns with only one unique value (dropped columns): {one_value_cols_sum}")

  return df

df_with_descriptors = drop_unique_value(df_with_descriptors)

# Create a ready up function where it will drop all obj type columns as this data type does not work with regression
def ready_up(df):
    """
    Drops all object-type columns from the DataFrame.

    Prints the names of the dropped columns.
    Returns the cleaned DataFrame.
    """
    # Identify object-type columns
    obj_cols = df.select_dtypes(include='object').columns.tolist()

    # Print dropped columns if any
    if obj_cols:
        print("Columns dropped:", ", ".join(obj_cols))
    else:
        print("No object-type columns to drop.")

    # Drop and return cleaned DataFrame
    return df.drop(columns=obj_cols)

# Create a fuction to save the model
def save_model(model, X_train, filename):
    """
    Save a trained model with its feature names into a pickle file.
    
    Args:
        model: trained sklearn or xgboost model
        X_train (pd.DataFrame): training features used to fit the model
        filename (str): output pickle file name
    """
    # Store feature names inside the model object
    model.feature_names = list(X_train.columns)
    
    # Save the model
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    
    print(f"✅ Model saved to {filename} with {len(model.feature_names)} features.")

# Create a function to automize target -> target with features ready
# Step 1: Until getting descriptors
def target_ready_step1(target_df):
    # Data Cleaning
    target_df = clean(target_df)
    
    # One-Hot Encoding
    # it is necessary to encode the categorical features such as 'construction', 'architechture' and 'complement' --> using One-hot encoding
    target_df = pd.get_dummies(target_df, columns=['construction', 'architecture', 'complement'])
    # change only the boolean type data to integer
    target_df = target_df.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)
    
    # Get Descriptors
    target_df = get_descriptors(target_df)
    
    # Ready Up
    target_df = ready_up(target_df)
    
    return target_df

# Step 2: Match features to model training set
def target_ready_step2(target_df, model_df):
    """
    Ensure that target_df has the same feature columns as model_df.
    Works for:
    - Only PCE_exp present
    - Only optical_gap_exp present
    - Both present
    - Neither present (already features only)
    """
    # Identify possible target columns
    possible_targets = ["PCE_exp", "optical_gap_exp"]
    present_targets = [col for col in possible_targets if col in model_df.columns]
    
    # Drop target columns if present
    if present_targets:
        model_features = model_df.drop(columns=present_targets).columns
    else:
        model_features = model_df.columns
    
    # Add missing columns to target_df
    for col in model_features:
        if col not in target_df.columns:
            target_df[col] = 0
    
    # Keep only model features
    target_df = target_df[model_features]
    
    return target_df

# Create a function to automize target -> target with features ready
# Step 1: Until getting descriptors
def target_ready_step1(target_df):
    # Data Cleaning
    target_df = clean(target_df)
    
    # One-Hot Encoding
    # it is necessary to encode the categorical features such as 'construction', 'architechture' and 'complement' --> using One-hot encoding
    target_df = pd.get_dummies(target_df, columns=['construction', 'architecture', 'complement'])
    # change only the boolean type data to integer
    target_df = target_df.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)
    
    # Get Descriptors
    target_df = get_descriptors(target_df)
    
    # Ready Up
    target_df = ready_up(target_df)
    
    return target_df

# Step 2: Match features to model training set
def target_ready_step2(target_df, model_df):
    """
    Ensure that target_df has the same feature columns as model_df.
    Works for:
    - Only PCE_exp present
    - Only optical_gap_exp present
    - Both present
    - Neither present (already features only)
    """
    # Identify possible target columns
    possible_targets = ["PCE_exp", "optical_gap_exp"]
    present_targets = [col for col in possible_targets if col in model_df.columns]
    
    # Drop target columns if present
    if present_targets:
        model_features = model_df.drop(columns=present_targets).columns
    else:
        model_features = model_df.columns
    
    # Add missing columns to target_df
    for col in model_features:
        if col not in target_df.columns:
            target_df[col] = 0
    
    # Keep only model features
    target_df = target_df[model_features]
    
    return target_df

def target_ready_step2_model(target_df, model):
    """
    Align target_df to the features expected by a trained model.
    
    Works with models saved using `save_model` (which stores .feature_names).
    
    - Adds missing columns (filled with 0)
    - Drops extra columns
    - Reorders columns to match model
    
    Args:
        target_df (pd.DataFrame): dataframe with descriptors & features
        model: trained model with .feature_names attribute
    
    Returns:
        pd.DataFrame: aligned features ready for prediction
    """
    # Load the saved model
    with open(model, "rb") as f:
        model = pickle.load(f)
    
    model_features = model.feature_names

    # Add missing columns
    missing_cols = [col for col in model_features if col not in target_df.columns]
    for col in missing_cols:
        target_df[col] = 0

    # Drop extra columns
    extra_cols = [col for col in target_df.columns if col not in model_features]
    if extra_cols:
        print(f"⚠️ Dropping {len(extra_cols)} extra columns: {extra_cols}")
        target_df = target_df.drop(columns=extra_cols)

    # Reorder columns
    target_df = target_df[model_features]

    print(f"✅ Features aligned: {len(model_features)} expected, {len(target_df.columns)} matched.")

    return target_df

# Create a feature matching function
def match_features(new_df, model_df, target_cols=["PCE_exp", "optical_gap_exp"]):
    """
    Check if new_df has exactly the same feature columns as model_df.
    
    - Drops target columns from model_df for comparison
    - Reports missing or extra columns
    - Does NOT auto-fix anything
    - Only reorders if features match perfectly
    
    Returns:
        aligned_df (DataFrame) if features match
    """
    # Drop target columns if they exist in model_df
    model_features = [col for col in model_df.columns if col not in target_cols]

    # Find missing and extra columns
    missing_cols = [col for col in model_features if col not in new_df.columns]
    extra_cols = [col for col in new_df.columns if col not in model_features]

    # Print counts for clarity
    print(f"📊 model_df has {len(model_features)} feature columns")
    print(f"📊 new_df has {new_df.shape[1]} columns")

    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
    if extra_cols:
        print(f"⚠️ Extra columns: {extra_cols}")

    # Stop if mismatch
    if missing_cols or extra_cols:
        raise ValueError("Feature mismatch: columns do not match exactly!")

    # Reorder to match model
    aligned_df = new_df[model_features]
    print("✅ Features match perfectly.")
    
    return aligned_df


# Create a predict function
def predict(data, target="both", pce_model=None, gap_model=None):
    """
    Predict PCE and/or optical gap for new data.

    Args:
        data (pd.DataFrame): Raw input data (must include SMILES + categorical cols).
        target (str): "pce", "gap", or "both"
        pce_model (str or model): filename (.pkl) or loaded model for PCE
        gap_model (str or model): filename (.pkl) or loaded model for optical gap

    Returns:
        pd.DataFrame: Original data + prediction columns
    """
    result_df = data.copy()

    # Step 1: descriptors & preprocessing
    data_ready = target_ready_step1(data)

    # Step 2: Predict depending on target
    if target in ["pce", "both"]:
        if pce_model is None:
            raise ValueError("PCE model not provided!")
        aligned_pce = target_ready_step2_model(data_ready.copy(), pce_model)
        if isinstance(pce_model, str):
            with open(pce_model, "rb") as f:
                pce_model = pickle.load(f)
        preds_pce = pce_model.predict(aligned_pce)
        result_df["Predicted_PCE_exp"] = preds_pce

    if target in ["gap", "both"]:
        if gap_model is None:
            raise ValueError("Optical gap model not provided!")
        aligned_gap = target_ready_step2_model(data_ready.copy(), gap_model)
        if isinstance(gap_model, str):
            with open(gap_model, "rb") as f:
                gap_model = pickle.load(f)
        preds_gap = gap_model.predict(aligned_gap)
        result_df["Predicted_optical_gap_exp"] = preds_gap

    return result_df

