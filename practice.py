import requests

class ApiManager:
    def __init__(self, url):
        self.api_url = url

    # 1. עדכן את הפונקציה לקבל פרמטרים (query_params)
    def fetch_data(self, query_params={}):
        try:
            # 2. עדכן את שורת ה-get כך שתשתמש ב-params
            response = requests.get(self.api_url,params=query_params)
            data = response.json()
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return []

# --- Main ---
# --- Main ---
search_url = "https://www.themealdb.com/api/json/v1/1/search.php"

# 3. צור אובייקט מנהל
chef = ApiManager(search_url)

# 4. הגדר את מילון החיפוש (המפתח חייב להיות "s")
my_filters = {"s": "cake"}

# 5. משוך את הנתונים
raw_data = chef.fetch_data(my_filters)

# 6. וודא שחזרו נתונים, וכתוב לולאה
if raw_data and raw_data["meals"]:
    print("saving recipes as files..")
    with open("my_list.txt", "w", encoding="utf-8") as file:
     for recipe in raw_data["meals"]:
        file.write(f"{recipe['strMeal']}\n")

    print("Done! Check your folder for recipes.txt")



else:
        print("No recipes found for that search.")


