import pandas as pd

from bs4 import BeautifulSoup


# Specify the path to your CSV file
csv_file = '/Users/varshaparthasarathy/Downloads/product.csv'

# Load the CSV file into a dataframe
df = pd.read_csv(csv_file)

# Drop rows where description is empty
df.dropna(subset=['description'], inplace=True)

# Convert HTML to text for description column
df['description'] = df['description'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())

# Remove newline characters from description column
df['description'] = df['description'].str.replace('\n', '')

# Drop columns price_cents and stripe_product_id
df.drop(['price_cents', 'stripe_product_id', 'updated_at', 'active', 'customer_fee', 'partner_fee', 'trial_period_days',
         'product_id', 'partner_id', 'upc_code', 'test_mode', 'shopify_product_id', 'gtin', 'reference_gtin', 'created_at'], axis=1, inplace=True)

df = df.fillna(0)

# Write the dataframe to a new CSV file
output_csv_file = '/Users/varshaparthasarathy/Downloads/cleaned_product.csv'
df.to_csv(output_csv_file, index=False)
print("Dataframe written to CSV file:", output_csv_file)

# Randomly spread hsa_fsa_eligibility
df_random = df.sample(frac=1).reset_index(drop=True)

# Write df_random to csv file
df_random.to_csv('/Users/varshaparthasarathy/Downloads/random_data.csv', index=False)