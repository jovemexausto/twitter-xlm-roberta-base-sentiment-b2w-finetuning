import pandas as pd
import re

def clean_text(text):
    if pd.isnull(text):
        return text
    text = re.sub(r'(.)\1+', r'\1', text)
    text = re.sub(r"[\"']", '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def main():
    # original_index,review_text,review_text_processed,review_text_tokenized,polarity,rating,kfold_polarity,kfold_rating
    df = pd.read_csv('./data/b2w.csv')
    
    positive = df[
        (df['polarity'] == 1) &
        (~df['review_text_processed'].str.contains('porem|caro|ressalva'))
    ]

    positive_sample = positive.sample(2000)
    positive_sample['review_text'] = positive_sample['review_text'].apply(clean_text)
    positive_sample['review_text_processed'] = positive_sample['review_text_processed'].apply(clean_text)
    positive_sample['polarity'] = 1
    positive_sample[['polarity', 'review_text']].to_csv('./data/b2w_positive.csv', index=False)
    
    negative = df[
        (df['polarity'] == 0)
    ]
    
    negative_sample = negative.sample(2000)
    negative_sample['review_text'] = negative_sample['review_text'].apply(clean_text)
    negative_sample['review_text_processed'] = negative_sample['review_text_processed'].apply(clean_text)
    negative_sample['polarity'] = -1
    negative_sample[['polarity', 'review_text']].to_csv('./data/b2w_negative.csv', index=False)

    neutral = df[
        (df['rating'] == 3) &
        (df['review_text_processed'].str.contains('porem|mas')) &
        (~df['review_text_processed'].str.contains('indico|bom|otimo|infelizmente|recomendo|pessimo|pessima|excelente|horrivel'))

    ]

    neutral_sample = neutral.sample(2000)
    neutral_sample['review_text'] = neutral_sample['review_text'].apply(clean_text)
    neutral_sample['review_text_processed'] = neutral_sample['review_text_processed'].apply(clean_text)
    neutral_sample['polarity'] = 0
    neutral_sample[['polarity', 'review_text']].to_csv('./data/b2w_neutral.csv', index=False)

if __name__ == "__main__":
    main()
