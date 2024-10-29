import requests
from bs4 import BeautifulSoup
import pandas as pd


def scrape_yelp_restaurant(url):
    response = requests.get(url)

    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract restaurant name
    restaurant_name_tag = soup.find('p', {'class': 'businessName__09f24__RKJO3 y-css-sauewc'})
    restaurant_name = restaurant_name_tag.text.strip() if restaurant_name_tag else "Restaurant Name Not Found"

    # Extract total reviews
    total_reviews_tag = soup.find('div', {'class': 'y-css-nb1hjy'})
    total_reviews = total_reviews_tag.text.strip() if total_reviews_tag else "Total Reviews Not Found"

    reviews_section = soup.find_all('div', {'class': 'snippetText__09f24__rMTqT y-css-15zxqka'})

    reviews_data = []

    for review in reviews_section:
        # Extract review text
        review_text_element = review.find('span', attrs={'data-testid': 'adsnippet.description'})
        review_text = review_text_element.text.strip() if review_text_element else "Review Text Not Found"

        # Extract reviewer
        reviewer_element = review.find('span', {'class': 'reviewerName__09f24__FjYSB y-css-1bbn5js'})
        reviewer = reviewer_element.text.strip() if reviewer_element else "Reviewer Not Found"

        reviews_data.append({
            'Reviewer': reviewer,
            'Review Text': review_text,
        })

    # Create a DataFrame to hold the extracted data
    df = pd.DataFrame(reviews_data)

    # Clean and format the DataFrame
    df['Restaurant Name'] = restaurant_name
    df['Total Reviews'] = total_reviews

    # Save the DataFrame into a CSV file
    df.to_csv(f'reviews.csv', index=False)
    print("Scraping completed and data saved to 'reviews.csv'.")


# Input: Yelp Restaurant URL
url = 'https://www.yelp.ca/biz/pai-northern-thai-kitchen-toronto-5?osq=Restaurants'

# Call the scraping function
scrape_yelp_restaurant(url)
