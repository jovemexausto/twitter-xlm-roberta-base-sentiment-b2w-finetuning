import pandas as pd
import re
import os

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove double and single quotes
    text = re.sub(r"[\"']", ' ', text)
    # Remove consecutive duplicate characters (3 or more, e.g., "heeeellooo" -> "hello")
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    # Remove non-alphanumeric characters (keeping spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace (replace multiple spaces with single space, strip leading/trailing)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_and_save_sample(df_filtered, polarity_value, output_filename, sample_size=2000):
    if df_filtered.empty:
        print(f"No data available for polarity {polarity_value}")
        return

    # Start with a larger sample to account for potential empty texts after cleaning
    # Use a multiplier to ensure we have enough samples
    initial_sample_size = min(int(sample_size * 1.5), len(df_filtered))
    
    attempts = 0
    max_attempts = 3
    final_sample_df = pd.DataFrame()
    
    while len(final_sample_df) < sample_size and attempts < max_attempts:
        # Calculate how many more samples we need
        needed_samples = sample_size - len(final_sample_df)
        
        # Determine sample size for this attempt
        if attempts == 0:
            current_sample_size = initial_sample_size
        else:
            # Increase sample size for subsequent attempts
            remaining_data = len(df_filtered) - len(final_sample_df)
            current_sample_size = min(int(needed_samples * 1.8), remaining_data)
        
        if current_sample_size <= 0:
            break
            
        # Get sample, excluding already selected indices
        available_df = df_filtered.drop(final_sample_df.index) if not final_sample_df.empty else df_filtered
        
        if available_df.empty:
            break
            
        sample_df = available_df.sample(
            n=min(current_sample_size, len(available_df)), 
            random_state=57 + attempts  # Different seed for each attempt
        )

        # Clean the text columns
        sample_df = sample_df.copy()
        sample_df['review_text_cleaned'] = sample_df['review_text'].apply(clean_text)
        sample_df['review_text_processed_cleaned'] = sample_df['review_text_processed'].apply(clean_text)

        # Filter out rows where both cleaned texts are empty
        sample_df = sample_df[
            (sample_df['review_text_cleaned'] != "") | 
            (sample_df['review_text_processed_cleaned'] != "")
        ]
        
        # Use the cleaned review_text as the main text, fallback to processed if empty
        sample_df['final_text'] = sample_df.apply(
            lambda row: row['review_text_cleaned'] if row['review_text_cleaned'] != "" 
            else row['review_text_processed_cleaned'], axis=1
        )
        
        # Filter out any remaining empty texts
        sample_df = sample_df[sample_df['final_text'] != ""]
        
        # Add to final sample
        final_sample_df = pd.concat([final_sample_df, sample_df], ignore_index=True)
        
        attempts += 1
        
        print(f"Attempt {attempts}: Got {len(sample_df)} valid samples (total: {len(final_sample_df)}/{sample_size})")
    
    # Take only the required number of samples
    if len(final_sample_df) > sample_size:
        final_sample_df = final_sample_df.sample(n=sample_size, random_state=57)
    
    if final_sample_df.empty:
        print(f"No valid samples found for polarity {polarity_value} after cleaning")
        return
    
    # Set polarity and prepare final dataframe
    final_sample_df['polarity'] = polarity_value
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save only the 'polarity' and 'final_text' columns (renamed to 'text')
    output_df = final_sample_df[['polarity', 'final_text']].rename(columns={'final_text': 'text'})
    output_df.to_csv(output_filename, index=False)
    
    print(f"Successfully processed and saved {len(output_df)} samples to {output_filename}")
    if len(output_df) < sample_size:
        print(f"Warning: Only {len(output_df)} samples available (requested: {sample_size})")


def main(input_csv_path='./data/b2w.csv', sample_size=2000):
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at '{input_csv_path}'. Please check the path.")
        return
    except Exception as e:
        print(f"Error reading CSV file '{input_csv_path}': {e}")
        return

    required_columns = ['polarity', 'review_text', 'review_text_processed', 'rating']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"Error: The input CSV is missing one or more required columns: {missing_cols}")
        return

    positive_exclude_keywords = 'porem|caro|ressalva'
    neutral_include_keywords  = 'porem|mas'
    neutral_exclude_keywords  = 'indico|bom|otimo|infelizmente|recomendo|pessimo|pessima|excelente|horrivel'

    print("\nProcessing negative samples...")
    negative_filtered = df[(df['polarity'] == 0)]
    process_and_save_sample(negative_filtered, 0, './data/b2w_negative.csv', sample_size)

    print("\nProcessing neutral samples...")
    neutral_filtered = df[
        (df['rating'] == 3) &
        (df['review_text_processed'].str.contains(neutral_include_keywords, na=False)) &
        (~df['review_text_processed'].str.contains(neutral_exclude_keywords, na=False))
    ]
    process_and_save_sample(neutral_filtered, 1, './data/b2w_neutral.csv', sample_size)

    print("\nProcessing positive samples...")
    positive_filtered = df[
        (df['polarity'] == 1) &
        (~df['review_text_processed'].str.contains(positive_exclude_keywords, na=False))
    ]
    process_and_save_sample(positive_filtered, 2, './data/b2w_positive.csv', sample_size)

    print("\nData processing complete. Check the './data/' directory for output files.")

if __name__ == "__main__":
    main(input_csv_path='./data/b2w.csv', sample_size=2500)