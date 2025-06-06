import pandas as pd
import os
from itertools import zip_longest

def create_sentiment_datasets(
    positive_csv_path='./data/b2w_positive.csv',
    negative_csv_path='./data/b2w_negative.csv',
    neutral_csv_path='./data/b2w_neutral.csv',
    output_dir='./training', # Output directory for train/test/val files
    train_ratio=0.8,
    test_ratio=0.1,
    val_ratio=0.1,
    sample_size=None,  # New parameter: if None, uses all data; if int, samples this many from each CSV
    random_state=42    # New parameter: for reproducible sampling
):
    """
    Loads sentiment data from the generated CSVs, optionally samples from them,
    interleaves it, splits into training, testing, and validation sets, 
    and saves text and labels to .txt files.
    
    Args:
        positive_csv_path: Path to positive sentiment CSV
        negative_csv_path: Path to negative sentiment CSV 
        neutral_csv_path: Path to neutral sentiment CSV
        output_dir: Directory to save train/test/val files
        train_ratio: Proportion of data for training (0.0-1.0)
        test_ratio: Proportion of data for testing (0.0-1.0)
        val_ratio: Proportion of data for validation (0.0-1.0)
        sample_size: Number of samples to take from each CSV file. If None, uses all data.
        random_state: Random seed for reproducible sampling
    """
    if not (train_ratio + test_ratio + val_ratio == 1.0):
        print("Error: Train, test, and validation ratios must sum to 1.0.")
        return

    def load_sentiment_data(file_path, sentiment_name):
        """Helper to load and validate sentiment specific CSVs."""
        if not os.path.exists(file_path):
            print(f"Error: {sentiment_name} CSV file not found at '{file_path}'. Skipping.")
            return [], []
        try:
            df = pd.read_csv(file_path)
            
            # Validate required columns
            if 'text' not in df.columns or 'polarity' not in df.columns:
                missing_cols = [col for col in ['text', 'polarity'] if col not in df.columns]
                print(f"Error: {sentiment_name} CSV missing columns: {missing_cols}. Skipping.")
                return [], []
            
            # Remove any rows with empty text
            df = df[df['text'].notna() & (df['text'].astype(str).str.strip() != '')]
            
            if df.empty:
                print(f"Warning: No valid data in {sentiment_name} CSV after filtering. Skipping.")
                return [], []
            
            # Sample if requested
            if sample_size is not None and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=random_state)
                print(f"Sampled {sample_size} rows from {len(df)} available {sentiment_name} reviews")
            
            texts = df['text'].astype(str).tolist()
            labels = df['polarity'].tolist()
            
            print(f"Loaded {len(texts)} {sentiment_name} reviews from {file_path}")
            return texts, labels
            
        except Exception as e:
            print(f"Error reading {sentiment_name} CSV file '{file_path}': {e}. Skipping.")
            return [], []

    # Load data for each sentiment
    # Note: Ensure the polarity values match what was set in the previous script (0 for neg, 1 for neu, 2 for pos)
    negative_texts, negative_labels = load_sentiment_data(negative_csv_path, "negative")
    neutral_texts, neutral_labels = load_sentiment_data(neutral_csv_path, "neutral")
    positive_texts, positive_labels = load_sentiment_data(positive_csv_path, "positive")

    # Collect non-empty datasets for interleaving
    interleave_items = []
    sentiment_counts = {}
    
    if negative_texts:
        interleave_items.append(list(zip(negative_texts, negative_labels)))
        sentiment_counts['negative'] = len(negative_texts)
    if neutral_texts:
        interleave_items.append(list(zip(neutral_texts, neutral_labels)))
        sentiment_counts['neutral'] = len(neutral_texts)
    if positive_texts:
        interleave_items.append(list(zip(positive_texts, positive_labels)))
        sentiment_counts['positive'] = len(positive_texts)

    if not interleave_items:
        print("No data found from any sentiment CSVs for interleaving. Exiting.")
        return

    print(f"\nSentiment distribution: {sentiment_counts}")

    # Interleave data using zip_longest to handle uneven list lengths
    interleaved_combined = list(zip_longest(*interleave_items))

    final_texts = []
    final_labels = []
    for group in interleaved_combined:
        for item in group:
            if item is not None:
                final_texts.append(item[0])
                final_labels.append(item[1])

    if not final_texts:
        print("No valid texts or labels after interleaving. Exiting.")
        return

    print(f"Total interleaved samples: {len(final_texts)}")

    # Count final label distribution
    label_counts = {}
    for label in final_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"Final label distribution: {label_counts}")

    # Manually split data to preserve the interleaved pattern
    num_samples = len(final_texts)
    train_end = int(num_samples * train_ratio)
    test_end = int(num_samples * (train_ratio + test_ratio))

    X_train = final_texts[:train_end]
    y_train = final_labels[:train_end]

    X_test = final_texts[train_end:test_end]
    y_test = final_labels[train_end:test_end]

    X_val = final_texts[test_end:]
    y_val = final_labels[test_end:]

    print(f"\nDataset splits:")
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Validation set size: {len(X_val)}")

    # Create output directory for final datasets
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCreated output directory: {output_dir}")

    # Save data to text files
    datasets = {
        'train': (X_train, y_train),
        'test': (X_test, y_test),
        'val': (X_val, y_val)
    }

    for name, (texts, labels) in datasets.items():
        if not texts:  # Skip empty datasets
            print(f"Warning: {name} dataset is empty, skipping file creation.")
            continue
            
        text_path = os.path.join(output_dir, f'{name}_text.txt')
        labels_path = os.path.join(output_dir, f'{name}_labels.txt')

        with open(text_path, 'w', encoding='utf-8') as f:
            for text_item in texts:
                # Explicitly convert to string here as a final safeguard
                f.write(str(text_item) + '\n')
        print(f"Saved {len(texts)} texts to {text_path}")

        with open(labels_path, 'w', encoding='utf-8') as f:
            for label_item in labels:
                f.write(str(label_item) + '\n')
        print(f"Saved {len(labels)} labels to {labels_path}")

    print("\nFinal dataset creation complete!")

if __name__ == "__main__":
    create_sentiment_datasets(
        positive_csv_path='./data/b2w_positive.csv',
        negative_csv_path='./data/b2w_negative.csv',
        neutral_csv_path='./data/b2w_neutral.csv',
        output_dir='./training',
        train_ratio=0.8,
        test_ratio=0.1,
        val_ratio=0.1,
        sample_size=2000,
        random_state=57
    )
    
    print("\nOverall data processing pipeline finished!")