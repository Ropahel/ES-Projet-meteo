import pyowm
from datetime import datetime, timedelta
import csv


def get_weather_data_delta(api_key, city,delta):
    # Initialize PyOWM with API key
    owm = pyowm.OWM(api_key)
    mgr = owm.weather_manager()
    
    # Create a list to store weather data
    weather_data = []
    # Set the start date as the current date and time
    current_date = datetime.utcnow()+timedelta(hours=2)
    end_date = current_date + timedelta(hours=delta)

    # Iterate until reaching the end date
    while current_date < end_date:
        # Retrieve weather information for the current date
        observation = mgr.weather_at_place(city)
        current_weather = observation.weather
        
        # Extract relevant information for the current date
        current_data = {
            "pressure": current_weather.pressure['press'],
            "temp": current_weather.temperature('celsius')['temp'],
            "humidity": current_weather.humidity,
            "wind_direction": current_weather.wind()['deg'],
            "wind_speed": current_weather.wind()['speed'],
            "clouds": current_weather.clouds,
            "rain": current_weather.rain.get('1h', 0),
            "visibility": current_weather.visibility_distance
        }

        # Extract relevant information for the past hour
        past_hour_data = {
            "pressure": current_weather.pressure['press'],
            "temp": current_weather.temperature('celsius')['temp'],
            "humidity": current_weather.humidity,
            "wind_direction": current_weather.wind()['deg'],
            "wind_speed": current_weather.wind()['speed'],
            "clouds": current_weather.clouds,
            "rain": current_weather.rain.get('1h', 0),
            "visibility": current_weather.visibility_distance,
            "hour": current_date.hour,
            "minute": current_date.minute,
            "day": current_date.day,
            "month": current_date.month
        }

        # Append the past hour data and current data to the weather data list
        weather_data.append([past_hour_data, current_data])

        # Move to the next hour
        current_date += timedelta(hours=delta)

    return weather_data


def get_weather_data_jeu_5H(api_key, city_name):
    # Initialisation de la liste pour stocker les données
    weather_data = []

    # Initialisation de l'objet OWM
    owm = pyowm.OWM(api_key)
    mgr = owm.weather_manager()

    # Récupération de la date et heure actuelle
    end_date = datetime(2022, 1, 1)
    start_date = end_date - timedelta(days=50)
    
    end_date += timedelta(days=0)
    start_date += timedelta(days=0)
    print(end_date)
    i = 0
    # Boucle à travers chaque heure dans la plage spécifiée
    current_date = start_date
    while current_date <= end_date:
        if i % 100 == 0:
            print("get_weather_data_jeu_5H", i)
        i += 1
        
        weather_data.append(get_weather_data_teste(api_key, city_name, 5, current_date))

        # Avancer d'une heure
        current_date += timedelta(hours=1)

    return weather_data

def get_weather_data_jeu_3H(api_key, city_name):
    # Initialisation de la liste pour stocker les données
    weather_data = []

    # Initialisation de l'objet OWM
    owm = pyowm.OWM(api_key)
    mgr = owm.weather_manager()

    # Récupération de la date et heure actuelle
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    print("Start Date:", start_date)
    print("End Date:", end_date)
    
    # Boucle à travers chaque heure dans la plage spécifiée
    current_date = start_date
    while current_date <= end_date:
        weather_data.append(get_weather_data_teste(api_key, city_name, 3, current_date))
        current_date += timedelta(hours=3)

    return weather_data

def get_weather_data_teste(api_key, city_name, delta, current_time):
    # Initialisation de la liste pour stocker les données
    weather_data = []

    # Initialisation de l'objet OWM
    owm = pyowm.OWM(api_key)
    mgr = owm.weather_manager()

    # Récupération de la date et heure actuelle
    current_date = current_time

    # Récupération des données météorologiques pour l'heure actuelle
    observation = mgr.weather_at_place(city_name)
    current_weather = observation.weather

    # Extraire les données météorologiques nécessaires pour l'heure actuelle
    current_weather_entry = {
        "pressure": current_weather.pressure['press'],
        "temp": current_weather.temperature('celsius')['temp'],
        "humidity": current_weather.humidity,
        "wind_direction": current_weather.wind()['deg'],
        "wind_speed": current_weather.wind()['speed'],
        "clouds": current_weather.clouds,
        "rain": current_weather.rain.get('1h', 0),
        "visibility": current_weather.visibility_distance,
        "hour": current_date.hour,
        "minute": current_date.minute,
        "day": current_date.day,
        "month": current_date.month
    }

    # Calcul de l'heure pour laquelle on veut les prévisions
    next_date = current_date + timedelta(hours=delta)

    try:
        # Récupération des prévisions météorologiques pour l'heure spécifiée
        forecast = mgr.forecast_at_place(city_name, '3h')
        next_hour_forecast = forecast.get_weather_at(next_date)

        # Extraire les données météorologiques nécessaires pour l'heure spécifiée
        next_hour_weather = {
            "pressure": next_hour_forecast.pressure['press'],
            "temp": next_hour_forecast.temperature('celsius')['temp'],
            "humidity": next_hour_forecast.humidity,
            "wind_direction": next_hour_forecast.wind()['deg'],
            "wind_speed": next_hour_forecast.wind()['speed'],
            "clouds": next_hour_forecast.clouds,
            "rain": next_hour_forecast.rain.get('1h', 0),
            "visibility": next_hour_forecast.visibility_distance,
        }

        # Ajouter les prévisions pour l'heure spécifiée à la liste
        weather_data.append([current_weather_entry, next_hour_weather])
    except Exception as e:
        print("Error fetching forecast:", e)

    return weather_data

def weather_data_to_jeu_donnees(weather_data):
    jeu_donnees = []
    L_jeu = []
    L_teste = []
    for jeu in weather_data:
        L_jeu = []
        L_teste = []
        for cle in jeu[0]:
            L_jeu.append(jeu[0][cle])
            for cle in jeu[1]:
                L_teste.append(jeu[1][cle])
    
        jeu_donnees.append([L_jeu,L_teste])
    return jeu_donnees

def weather_data_to_jeu_teste(weather_data):
    jeu_teste = []
    L_jeu = []
    L_teste = []
    for jeu in weather_data:
        L_jeu = []
        L_teste = []
        for cle in jeu[0]:
            L_jeu.append(jeu[0][cle])
            for cle in jeu[1]:
                L_teste.append(jeu[1][cle])
    
        jeu_teste.append([L_jeu,L_teste])
    return jeu_teste

def save_weather_data_to_csv(weather_data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for jeu in weather_data:
            writer.writerow(jeu[0] + jeu[1])  # Concaténer les deux listes de données

def load_weather_data_from_csv(filename):
    weather_data = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Diviser chaque ligne en deux parties égales
            middle_index = len(row) // 2
            jeu_1 = row[:middle_index]
            jeu_2 = row[middle_index:]
            weather_data.append([jeu_1, jeu_2])
    return weather_data

def get_weather_data(api_key, city):
    owm = pyowm.OWM(api_key)
    mgr = owm.weather_manager()

    # Fonction pour obtenir les données météorologiques pour une heure spécifique
    def get_hourly_weather_at_time(city_id, reference_time):
        forecast = mgr.weather_at_time(city_id, reference_time)
        weather = forecast.weather
        return weather.to_dict()

    # Obtenir l'ID de la ville
    city_id = mgr.city_id_registry().ids_for(city)[0]

    # Obtenir les données météorologiques actuelles
    observation = mgr.weather_at_place(city)
    current_data = observation.weather.to_dict()

    # Obtenir les données météorologiques d'il y a une heure
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    one_hour_data = get_hourly_weather_at_time(city_id, one_hour_ago)

    # Traitement des données pour chaque sous-liste
    def process_data(data, include_time=False):
        pressure = data['pressure']['press']
        temp = data['temperature']['temp']
        humidity = data['humidity']
        wind_direction = data['wind']['deg']
        wind_speed = data['wind']['speed']
        clouds = data['clouds']
        rain = data.get('rain', {}).get('1h', 0)
        visibility = data.get('visibility_distance', 0)
        result = [pressure, temp, humidity, wind_direction, wind_speed, clouds, rain, visibility]
        if include_time:
            # Conversion de la date en heure locale
            timestamp = data['reference_time']
            dt_object = datetime.utcfromtimestamp(timestamp)
            hour = dt_object.hour
            minute = dt_object.minute
            day = dt_object.day
            month = dt_object.month
            result.extend([hour, minute, day, month])
        return result

    # Traitement des données pour les deux sous-listes
    current_weather_list = process_data(current_data, include_time=True)
    one_hour_weather_list = process_data(one_hour_data)

    return [one_hour_weather_list, current_weather_list]
    
    
    

api_key = '4a315459a71c408a35c309ffa9c7796d'
city_name = 'Cherbourg'

#
weather_data_jeu_48H = get_weather_data_delta(api_key, city_name, 1)

#
jeu_donnees_jeu_48H = weather_data_to_jeu_donnees(weather_data_jeu_48H)

#
save_weather_data_to_csv(jeu_donnees_jeu_48H, 'CSV_pour_exel.csv')


#weather_data_jeu_24H = get_weather_data_delta(api_key, city_name, 24)

#jeu_donnees_jeu_24H = weather_data_to_jeu_donnees(weather_data_jeu_24H)

#save_weather_data_to_csv(jeu_donnees_jeu_24H, 'jeu_donnees_jeu_pour_ia_24H_16.csv')

print("FIN")

















