
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_yelp_restaurant(url):
    response = requests.get(url)
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    restaurant_name = soup.find('p', {'class': 'businessName__09f24__RKJO3 y-css-sauewc'}).text.strip()

    # print("adsfafdafdadf", restaurant_name)
    
    
    total_reviews = soup.find('div', {'class': 'y-css-nb1hjy'}).text.strip()
    # print("total reviews", total_reviews)
    reviews_section = soup.find_all('div', {'class': 'snippetText__09f24__rMTqT y-css-15zxqka', })
    # print("reviews section", reviews_section)
    
    reviews_data = []
    
    for review in reviews_section:
        # review_text = review.find('span', {'class': 'y-css-ya63xp'}).text.strip()
        review_text = review.find('span', attrs={'data-testid': 'adsnippet.description'}).text.strip()
        reviewer = review.find('span', {'class': 'reviewerName__09f24__FjYSB y-css-1bbn5js'}).text.strip()
        # print("reviewer", reviewer)
        # rating_class = review.find('div', {'class': 'y-css-1jwbncq'})['aria-label']
        # print('rating class', rating_class)
        # rating = rating_class.split(' ')[0]  # Extract just the rating number
        
        reviews_data.append({
            'Reviewer': reviewer,
            'Review Text': review_text,
        #     'Rating': rating
        })
        
        # print("data", reviews_data)
    
    # Create a DataFrame to hold the extracted data
    df = pd.DataFrame(reviews_data)
    
    # # Clean and format the DataFrame
    df['Restaurant Name'] = restaurant_name
    df['Total Reviews'] = total_reviews
    
    # # Save the DataFrame into a CSV file
    df.to_csv(f'D:\Study Materials\George Brown\Machine Learning I\Exam\{restaurant_name}_reviews.csv', index=False)
    

# Input: Yelp Restaurant URL
url = 'https://www.yelp.ca/biz/pai-northern-thai-kitchen-toronto-5?osq=Restaurants'

# Call the scraping function
scrape_yelp_restaurant(url)

