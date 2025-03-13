import pandas as pd
import urllib.request
import json

# Load the CSV file
file_path = '/Users/thalialightstone/Documents/LIS 598D Search and Discovery /proj2/friendBooks.csv' # Make sure this is the correct path to your CSV file
df = pd.read_csv(file_path)

# Clean ISBN columns to remove =" and "
def clean_isbn(isbn):
    if pd.isna(isbn) or isbn == '=""':
        return None
    return isbn.replace('="', '').replace('"', '')

# Apply cleaning function to ISBN columns
df['ISBN'] = df['ISBN'].apply(clean_isbn)
df['ISBN13'] = df['ISBN13'].apply(clean_isbn)

# Function to fetch genres using the Open Library API
def fetch_genres(isbn):
    if not isbn:
        return "No ISBN"
    base_api_link = "https://openlibrary.org/api/books?bibkeys=ISBN:"
    url = f"{base_api_link}{isbn}&format=json&jscmd=data"
    try:
        with urllib.request.urlopen(url) as f:
            text = f.read()
        decoded_text = text.decode("utf-8")
        obj = json.loads(decoded_text)
        book_data = obj.get(f"ISBN:{isbn}", None)
        if not book_data:
            return "Not Found"
        subjects = book_data.get("subjects", [])
        if subjects:
            return ", ".join([subject["name"] for subject in subjects])
        else:
            return "No categories available"
    except:
        return "Error fetching data"

# Use ISBN13 if available, otherwise use ISBN
df['Used ISBN'] = df.apply(lambda row: row['ISBN13'] if pd.notna(row['ISBN13']) else row['ISBN'], axis=1)

# Fetch genres for each ISBN and add them to a new "Genre" column
df['Genre'] = df['Used ISBN'].apply(fetch_genres)

# Save the updated DataFrame to a new CSV file
updated_file_path = 'friendbooks_with_genres.csv'
df.to_csv(updated_file_path, index=False)

print(f"Updated CSV saved as {updated_file_path}")
