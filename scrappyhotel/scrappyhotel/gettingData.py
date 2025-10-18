import requests
import pandas as pd
import time

API_KEY = 'AIzaSyA4IYrDw9nPG-amFdhrijAiFU3X7_ku2bI'
MUĞLA_COORDS = '37.2153,28.3636'  # Approximate center of Muğla
RADIUS = 20000  # Radius in meters

# Define keywords for broader search
keywords = ["hotel", "motel", "apart", "pension", "hostel"]

# Initialize data storage
hotel_data = []


# Define a function to retrieve place_ids with pagination
def get_place_ids(query, location, radius):
    place_ids = []
    url = (
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
        f"location={location}&radius={radius}&type=lodging&keyword={query}&key={API_KEY}"
    )

    while url:
        response = requests.get(url).json()

        # Append each place_id from the response
        for result in response.get('results', []):
            place_ids.append(result['place_id'])

        # Check if there’s a next page and update the URL if so
        next_page_token = response.get("next_page_token")
        if next_page_token:
            url = (
                "https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
                f"pagetoken={next_page_token}&key={API_KEY}"
            )
            time.sleep(2)  # Wait before the next request
        else:
            url = None  # End the loop if there are no more pages

    return place_ids


# Define a function to retrieve details and reviews for each hotel by place_id
def get_hotel_details(place_id):
    # Türkçe ve İngilizce için iki ayrı API çağrısı
    url_tr = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&language=tr&key={API_KEY}"
    url_en = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&language=en&key={API_KEY}"

    response_tr = requests.get(url_tr).json()
    response_en = requests.get(url_en).json()

    hotel_info = {
        "name": response_tr.get("result", {}).get("name") or response_en.get("result", {}).get("name"),
        "place_id": place_id,
        "rating": response_tr.get("result", {}).get("rating") or response_en.get("result", {}).get("rating"),
        "total_ratings": response_tr.get("result", {}).get("user_ratings_total") or response_en.get("result", {}).get(
            "user_ratings_total"),
        "address": response_tr.get("result", {}).get("formatted_address") or response_en.get("result", {}).get(
            "formatted_address"),
        "phone": response_tr.get("result", {}).get("formatted_phone_number") or response_en.get("result", {}).get(
            "formatted_phone_number"),
        "reviews": [],
    }

    # Türkçe incelemeleri ekle
    if "reviews" in response_tr.get("result", {}):
        for review in response_tr["result"]["reviews"]:
            hotel_info["reviews"].append({
                "language": "tr",
                "author_name": review.get("author_name"),
                "rating": review.get("rating"),
                "text": review.get("text"),
                "relative_time_description": review.get("relative_time_description")
            })

    # İngilizce incelemeleri ekle
    if "reviews" in response_en.get("result", {}):
        for review in response_en["result"]["reviews"]:
            hotel_info["reviews"].append({
                "language": "en",
                "author_name": review.get("author_name"),
                "rating": review.get("rating"),
                "text": review.get("text"),
                "relative_time_description": review.get("relative_time_description")
            })

    return hotel_info


# Query hotels in Muğla using multiple keywords
for keyword in keywords:
    place_ids = get_place_ids(keyword, MUĞLA_COORDS, RADIUS)

    for place_id in place_ids:
        time.sleep(1)  # Avoid hitting API rate limits
        hotel_details = get_hotel_details(place_id)
        if hotel_details:
            hotel_data.append(hotel_details)

# Convert to a DataFrame and save as CSV
df = pd.DataFrame(hotel_data)
df.to_csv("mugla_hotels_expanded2.csv", index=False)
print("Data saved to mugla_hotels_expanded2.csv")
