from math import exp
from random import randint,shuffle
import csv
import pyowm
from datetime import datetime, timedelta
import csv
import copy

def init_params(nb_couche, nb_neurones_couches, nb_neurones_out):
    L = [[[randint(-5, 5)*0.05619874958473269784, randint(-5, 5)*0.3985674078923] for _ in range(nb_couche)] for _ in range(nb_neurones_couches)]
    L_out = [[randint(1, 50)*0.078692890195838, randint(1, 50)*0.06857426798019581476] for _ in range(nb_neurones_out)]
    return L, L_out

def sigmoid(x):
    if x >= 0:
        result = 1 / (1 + exp(-x))
    else:
        result = exp(x) / (1 + exp(x))
    return result

def sigmoid_deriv(x):
    result = sigmoid(x) * (1 - sigmoid(x))
    return result

def reLU(x):
    return max(0, x)

def reLU_deriv(x):
    return 1 if x > 0 else 0

def affiche(L):
    for i in range(len(L)):
        for j in range(len(L[0])):
            print(L[i][j], end=' ')
        print()

def comparer_a_pres(nombre, cible, tolerance):
    return abs(float(nombre) - float(cible)) <= tolerance

def accuracy(result, attendu):
    accu = 0

    for i in range(len(result)):
        if comparer_a_pres(result[i], attendu[i], 1.2):
            accu += 1 / len(result)
    return accu

def forward_prop(L, L_out, pression, temp, humidite, direc_vent, vitesse_vent, nuage, pluie, visibilitee, heure, minute,
                 jour, mois):
    L_result_out = [[0, 0] for _ in range(len(L_out))]
    L_result = [[0 for _ in range(len(L[0]))] for _ in range(len(L))]
    
    pression = float(pression)
    temp = float(temp)
    humidite = float(humidite)
    direc_vent = float(direc_vent)
    vitesse_vent = float(vitesse_vent)
    nuage = float(nuage)
    pluie = float(pluie)
    visibilitee = float(visibilitee)
    heure = float(heure)
    minute = float(minute)
    jour = float(jour)
    mois = float(mois)

    # Couche 1 à la main
    L_result[0][0] = pression * L[0][0][0] + L[0][0][1]
    L_result[1][0] = temp * L[1][0][0] + L[1][0][1]
    L_result[2][0] = humidite * L[2][0][0] + L[2][0][1]
    L_result[3][0] = direc_vent * L[3][0][0] + L[3][0][1]
    L_result[4][0] = vitesse_vent * L[4][0][0] + L[4][0][1]
    L_result[5][0] = nuage * L[5][0][0] + L[5][0][1]
    L_result[6][0] = pluie * L[6][0][0] + L[6][0][1]
    L_result[7][0] = visibilitee * L[7][0][0] + L[7][0][1]
    L_result[8][0] = heure * L[8][0][0] + L[8][0][1]
    L_result[9][0] = minute * L[9][0][0] + L[9][0][1]
    L_result[10][0] = jour * L[10][0][0] + L[10][0][1]
    L_result[11][0] = mois * L[11][0][0] + L[11][0][1]

    # Pour L_result
    for couches in range(1, len(L[0])):
        S = 0
        for y in range(len(L)):
            S += L_result[y][couches - 1]
        for neurones in range(len(L)):
            poids = L[neurones][couches][0]
            biai = L[neurones][couches][1]
            # Utilisation de la fonction sigmoïde pour les couches cachées
            L_result[neurones][couches] = sigmoid(poids * S + biai)

    # Pour L_result_out
    S = 0
    for y1 in range(len(L)):
        S += L_result[y1][len(L[0])-1]
    for neurones_out in range(len(L_out)):
        poids = L_out[neurones_out][0]
        biai = L_out[neurones_out][1]
        # Utilisation de la fonction ReLU pour la couche de sortie
        L_result_out[neurones_out] = reLU(S * poids + biai)

    return L_result_out, L_result

def fonction_cout(L_result, L_verif):
    L_verif = [float(val) for val in L_verif]
    L_result = [float(val) for val in L_result]
    return sum((result - verif) ** 2 for result, verif in zip(L_result, L_verif))

def descente(L, L_out, L_result, L_verif, learning_rate):
    # Calcul des gradients pour les couches cachées
    L_verif = [float(val) for val in L_verif]
    #Correction des valeurs pour ne pas avoir des gradients nules
    for i in range(len(L_verif)):
        if L_verif[i] == 0:
            L_verif[i] = 0.1
    for j in range(len(L_result)):
        if L_result[i] == 0:
            L_result[i] == 0.1
            
    gradient_poids_cachees = [[0 for _ in range(len(L[0]))] for _ in range(len(L))]
    gradient_biais_cachees = [[0 for _ in range(len(L[0]))] for _ in range(len(L))]
    for neurone_cache in range(len(L[0])):
        for neurone_entree in range(len(L)):
            somme_gradient_poids = 0
            somme_gradient_biais = 0
            for neurone_sortie in range(len(L_out)):
                sigmoide_dirv_result =  sigmoid_deriv(L_result[neurone_sortie])
                #Correction si  sigmoide_dirv_result = 0
                if sigmoide_dirv_result == 0:
                    sigmoide_dirv_result = 0.000001
                delta = 2 * (L_result[neurone_sortie] - L_verif[neurone_sortie]) * sigmoide_dirv_result
                
                somme_gradient_poids += delta * L[neurone_entree][neurone_cache][0]
                somme_gradient_biais += delta
            gradient_poids_cachees[neurone_entree][neurone_cache] = somme_gradient_poids
            gradient_biais_cachees[neurone_entree][neurone_cache] = somme_gradient_biais
            
    # Calcul des gradients pour la couche de sortie
    gradient_poids_sortie = [0] * len(L_out)
    gradient_biais_sortie = [0] * len(L_out)
    for neurone_sortie in range(len(L_out)):
        gradient_poids_sortie[neurone_sortie] = 2 * (L_result[neurone_sortie] - L_verif[neurone_sortie]) * reLU_deriv(L_result[neurone_sortie])  
        gradient_biais_sortie[neurone_sortie] = 2 * (L_result[neurone_sortie] - L_verif[neurone_sortie]) * reLU_deriv(L_result[neurone_sortie])  
    

    # Mise à jour des paramètres pour les couches cachées
    for neurone_entree in range(len(L)):
        for neurone_cache in range(len(L[0])):
            L[neurone_entree][neurone_cache][0] -= learning_rate * gradient_poids_cachees[neurone_entree][neurone_cache]
            L[neurone_entree][neurone_cache][1] -= learning_rate * gradient_biais_cachees[neurone_entree][neurone_cache]

    # Mise à jour des paramètres pour la couche de sortie
    for neurone_sortie in range(len(L_out)):
        L_out[neurone_sortie][0] -= learning_rate * gradient_poids_sortie[neurone_sortie]
        L_out[neurone_sortie][1] -= learning_rate * gradient_biais_sortie[neurone_sortie]

def predict(L, L_out, L_don):
    pression = L_don[0]
    temp = L_don[1]
    humidite = L_don[2]
    direc_vent = L_don[3]
    vitesse_vent = L_don[4]
    nuage = L_don[5]
    pluie = L_don[6]
    visibilitee = L_don[7]
    heure = L_don[8]
    minute = L_don[9]
    jour = L_don[10]
    mois = L_don[11]

    return forward_prop(L, L_out, pression, temp, humidite, direc_vent,
                        vitesse_vent, nuage, pluie, visibilitee, heure, minute, jour, mois)

def apprentissage(L, L_out, learning_rate, epochs, jeu_donnees, jeu_teste):
    print("Apprentissage en cour...")
    for boucle in range(1,epochs+1):
        if boucle % 100 == 0:
            print(boucle,end="/")
        shuffle(jeu_donnees)
        for jeu in jeu_donnees:
            donnee = jeu[0]
            L_verif = jeu[1]

            predi = predict(L, L_out, donnee)
            predi_out = predi[0]
            #L_result = predi[1]

            descente(L, L_out, predi_out, L_verif, learning_rate)
    
    print()

def teste(L,L_out,jeu_teste):
    total_loss = 0
    total_accuracy = 0

    for jeu in jeu_teste:
        donnee = jeu[0]
        L_verif = jeu[1]
        predi = predict(L, L_out, donnee)
        predi_out = predi[0]

        loss = fonction_cout(predi_out, L_verif)
        total_loss += loss

        accu = accuracy(predi_out, L_verif)
        total_accuracy += accu

    avg_loss = total_loss / len(jeu_teste)
    avg_accuracy = total_accuracy / len(jeu_teste)

    print("Average loss =", avg_loss)
    print("Average accuracy =", avg_accuracy)

def aprend(L, L_out, jeu_donnees_teste_loaded, jeu_donnees_jeu_loaded, epochs, learning_rate):
    print("L Avant")
    print()
    affiche(L)
    print()
    print("L_out Avant")
    print()
    affiche(L_out)
    print()
    
    print("Teste Avant")
    teste(L,L_out,jeu_donnees_teste_loaded)
    print()
    print("nb_exmple: ",len(jeu_donnees_jeu_loaded))
    print()
    print("nb_epochs: ", epochs)

    apprentissage(L, L_out, learning_rate, epochs, jeu_donnees_jeu_loaded, jeu_donnees_teste_loaded)

    print("L Après;")
    print()
    print(L)
    print()
    print("L_out Après;")
    print()
    print(L_out)
    print()


    print("Teste après:")
    teste(L,L_out,jeu_donnees_teste_loaded) 
    
def verif_prediction(L,L_out,api_key,city_name,delta):
    weather_data = get_weather_data_delta(api_key, city_name,delta)
    
    donnees_meteo = weather_data_to_jeu_donnees_pour_1_jeu(weather_data)
    
    expected = donnees_meteo[0][1]
    
    teste(L,L_out,donnees_meteo)
    
    prediction =  predict(L, L_out, donnees_meteo[0][0])
    L_ou = prediction[0]
    for i in range(len(L_ou)):
        if i != 7:
            if comparer_a_pres(L_ou[i], expected[i], 2):
                print("Prediction : ",L_ou[i],"/Expected : ",expected[i]," X")
            else:
                print("Prediction : ",L_ou[i],"/Expected : ",expected[i])
        else:
            if comparer_a_pres(L_ou[i], expected[i], 1005):
                print("Prediction : ",L_ou[i],"/Expected : ",expected[i]," X")
            else:
                print("Prediction : ",L_ou[i],"/Expected : ",expected[i])


def apprentissage_8(L, L_out, learning_rate, epochs, jeu_donnees, jeu_teste):
    print("Apprentissage en cour...")
    for boucle in range(1,epochs+1):
        if boucle % 100 == 0:
            print(boucle,end="/")
        for jeu in jeu_donnees:
            donnee = jeu[0]
            L_verif = jeu[1]
            for AI in range(len(L)):
                predi = predict(L[AI], L_out[AI], donnee)
                predi_out = predi[0]
                #L_result = predi[1]
                descente(L[AI], L_out[AI], predi_out, [L_verif[AI]], learning_rate)
    
    print()

def teste_8(L,L_out,jeu_teste):
    total_loss = 0
    total_accuracy = 0
    for jeu in jeu_teste:
        donnee = jeu[0]
        L_verif = jeu[1]
        for AI in range(len(L)):
            predi = predict(L[AI], L_out[AI], donnee)
            predi_out = predi[0]

            loss = fonction_cout(predi_out, [L_verif[AI]])
            total_loss += loss
    
            accu = accuracy_8(predi_out, L_verif,AI)
            total_accuracy += accu
    avg_loss = total_loss / len(jeu_teste)
    avg_accuracy = total_accuracy / len(jeu_teste)

    print("Average loss =", avg_loss)
    print("Average accuracy =", avg_accuracy*100,"%")

def aprend_8(L, L_out, jeu_donnees_teste_loaded, jeu_donnees_jeu_loaded, epochs, learning_rate):
    print("L Avant")
    print()
    #print(L)
    print()
    print("L_out Avant")
    print()
    #print(L_out)
    print()
    
    print("Teste Avant")
    teste_8(L,L_out,jeu_donnees_teste_loaded)
    print()
    print("nb_exmple: ",len(jeu_donnees_jeu_loaded))
    print()
    print("nb_epochs: ", epochs)

    apprentissage_8(L, L_out, learning_rate, epochs, jeu_donnees_jeu_loaded, jeu_donnees_teste_loaded)

    print("L Après;")
    print()
    print(L)
    print()
    print("L_out Après;")
    print()
    print(L_out)
    print()


    print("Teste après:")
    teste_8(L,L_out,jeu_donnees_teste_loaded) 
    
def verif_prediction_8(L,L_out,api_key,city_name,delta):
    weather_data = get_weather_data_delta(api_key, city_name,delta)
    
    donnees_meteo = weather_data_to_jeu_donnees_pour_1_jeu(weather_data)
    print(donnees_meteo[0][0])
    print(datetime.utcnow()-timedelta(hours=delta)+timedelta(hours=2))
    print()
    
    for jeu in donnees_meteo:
        jeu[0][9] == 0.01
        for sous_jeu in jeu:
            sous_jeu[1] = round(float(sous_jeu[1]),1)
            sous_jeu[5] = float(sous_jeu[5])/100
            
            valu = float(sous_jeu[3])
            if valu >= 337.5 or valu <= 22.5:
                sous_jeu[3] = 0
            elif valu <= 76.5:
                sous_jeu[3] = 1
            elif valu <= 112.5:
                sous_jeu[3] = 2
            elif valu <= 157.5:
                sous_jeu[3] = 3
            elif valu <= 205.5:
                sous_jeu = 4
            elif valu <= 247.5:
                sous_jeu[3] = 5
            elif valu <= 292.5:
                sous_jeu[3] = 6
            elif valu <337.5:
                sous_jeu[3] = 7
    
    expected = donnees_meteo[0][1]
    
    teste_8(L,L_out,donnees_meteo)
    L_ou = []
    for AI in range(len(L)):
        prediction =  predict(L[AI], L_out[AI], donnees_meteo[0][0])
        L_ou.append(prediction[0][0])
    
    for i in range(len(L_ou)):
        if i != 7:
            if comparer_a_pres(L_ou[i], expected[i], 2):
                print("Prediction : ",L_ou[i],"/Expected : ",expected[i]," X")
            else:
                print("Prediction : ",L_ou[i],"/Expected : ",expected[i])
        else:
            if comparer_a_pres(L_ou[i], expected[i], 1005):
                print("Prediction : ",L_ou[i],"/Expected : ",expected[i]," X")
            else:
                print("Prediction : ",L_ou[i],"/Expected : ",expected[i])
            
def verif_prediction_8_V(L,L_out,api_key,city_name,delta):
    weather_data = get_weather_data_delta(api_key, city_name,delta)
    
    donnees_meteo = weather_data_to_jeu_donnees_pour_1_jeu(weather_data)
    for jeu in donnees_meteo:
        jeu[0][9] == 0.01
        for sous_jeu in jeu:
            sous_jeu[1] = round(float(sous_jeu[1]),1)
            sous_jeu[5] = float(sous_jeu[5])/100
            
            valu = float(sous_jeu[3])
            if valu >= 337.5 or valu <= 22.5:
                sous_jeu[3] = 0
            elif valu <= 76.5:
                sous_jeu[3] = 1
            elif valu <= 112.5:
                sous_jeu[3] = 2
            elif valu <= 157.5:
                sous_jeu[3] = 3
            elif valu <= 205.5:
                sous_jeu = 4
            elif valu <= 247.5:
                sous_jeu[3] = 5
            elif valu <= 292.5:
                sous_jeu[3] = 6
            elif valu <337.5:
                sous_jeu[3] = 7
    
    donnees_meteo_V = prep_V(donnees_meteo)
    expected = donnees_meteo_V[0][1]
    teste_8(L,L_out,donnees_meteo_V)
    L_ou = []
    for AI in range(len(L)):
        prediction =  predict(L[AI], L_out[AI], donnees_meteo_V[0][0])
        L_ou.append(prediction[0][0])
    
    for i in range(len(L_ou)):
        if comparer_a_pres(L_ou[i], expected[i], 0.3):
            print("Prediction : ",L_ou[i],"/Expected : ",expected[i]," X")
        else:
            print("Prediction : ",L_ou[i],"/Expected : ",expected[i])

def accuracy_8(result, attendu,ind_teste):
    accu = 0
    if ind_teste != 7:
        if comparer_a_pres(result[0], attendu[ind_teste], 0.3):
            accu += 1 / 8
    else:
        if comparer_a_pres(result[0], attendu[ind_teste], 0.3):
            accu += 1 / 8
    return accu

def pourcentage_changement(L_8_1H_3C, L_out_8_1H_3C, L_8_1H_3C_AV, L_out_8_1H_3C_AV ):
    change_poids = 0
    change_biais = 0
    change_T = 0
    nb_neurones = len(L_8_1H_3C) * len(L_8_1H_3C[0]) * len(L_8_1H_3C[0][0]) + 8
    for AI in range(len(L_8_1H_3C)):
        for couches in range(len(L_8_1H_3C[AI])):
            for neurones in range(len(L_8_1H_3C[AI][couches])):
                val_in_P = L_8_1H_3C_AV[AI][couches][0][0]
                val_in_B = L_8_1H_3C_AV[AI][couches][1][1]
                if val_in_P == 0:
                    val_in_P = 0.0000000001
                if val_in_B == 0:
                    val_in_B = 0.0000000001
                change_poids += abs(((L_8_1H_3C[AI][couches][0][0] - val_in_P)/val_in_P)/nb_neurones)
                change_biais += abs(((L_8_1H_3C[AI][couches][1][1] - val_in_B)/val_in_B)/nb_neurones)
    
    for AI in range(len(L_out_8_1H_3C)):
        for couches in range(len(L_out_8_1H_3C[AI])):
            for neurones in range(len(L_out_8_1H_3C[AI][couches])):
                val_in_P = L_out_8_1H_3C_AV[AI][couches][0]
                val_in_B = L_out_8_1H_3C_AV[AI][couches][1]
                if val_in_P == 0:
                    val_in_P = 0.0000000001
                if val_in_B == 0:
                    val_in_B = 0.0000000001
                change_poids += abs(((L_out_8_1H_3C[AI][couches][0] - val_in_P)/val_in_P)/nb_neurones)
                change_biais += abs(((L_out_8_1H_3C[AI][couches][1] - val_in_B)/val_in_B)/nb_neurones)
                
    change_T += change_poids + change_biais
    
    print()
    print("Moy change poids =", (change_poids/nb_neurones))
    print("Moy change biais =", change_biais/nb_neurones)
    print("Moy change T =",(change_T/nb_neurones))

def prep_V(jeu_donnees):
    L_donnees_V = []
    L_V = []
    for jeu in jeu_donnees:
        L_V = []
        for ind in range(len(jeu[1])):
            if float(jeu[0][ind]) > float(jeu[1][ind]):
                L_V.append(0)
            elif float(jeu[0][ind]) < float(jeu[1][ind]):
                L_V.append(1)
            elif float(jeu[0][ind]) == float(jeu[1][ind]):
                L_V.append(0.5)
        L_donnees_V.append([jeu[0],L_V])
    return L_donnees_V
                
                                
def load_weather_data_from_csv(filename):
    L_jeu = []
    L_teste = []
    weather_data = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            L_jeu = [row[i] for i in range(12)]
            L_teste = [row[j] for j in range(12,20)]
            weather_data.append([L_jeu,L_teste])
        for jeu in weather_data:
            for List in jeu:
                for nb in List:
                    nb = float(nb)
    return weather_data

def get_weather_data_teste(api_key, city_name, delta):
    # Initialisation de la liste pour stocker les données
    weather_data = []

    # Initialisation de l'objet OWM
    owm = pyowm.OWM(api_key)
    mgr = owm.weather_manager()

    # Récupération de la date et heure actuelle
    current_date = datetime.utcnow()

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

    # Récupération des prévisions météorologiques pour l'heure spécifiée après l'heure actuelle
    forecast = mgr.forecast_at_place(city_name, '3h')
    next_hour_forecast = forecast.get_weather_at(current_date - timedelta(hours=delta))

    # Extraire les données météorologiques nécessaires pour l'heure spécifiée après l'heure actuelle
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
    # Ajouter les prévisions pour l'heure spécifiée après l'heure actuelle à la liste
    weather_data.append([next_hour_weather,current_weather_entry])

    return weather_data

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

def weather_data_to_jeu_donnees_pour_1_jeu(weather_data):
    
    jeu_donnees = []
    L_jeu = [weather_data[0][0][cle] for cle in weather_data[0][0]]
    L_teste = [weather_data[0][1][cle] for cle in weather_data[0][1]]
    jeu_donnees = [[L_jeu,L_teste]] 
    return jeu_donnees


jeu_donnees_teste_loaded = []
jeu_donnees_jeu_loaded = load_weather_data_from_csv('jeu_donnees_jeu_pour_ia.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_teste_pour_ia.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_2.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_teste_pour_ia_2.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_teste_pour_ia_3.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_4.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_teste_pour_ia_4.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_5.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_teste_pour_ia_5.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_6.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_teste_pour_ia_6.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_7.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_8.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_9.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_10.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_11.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_12.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_13.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_14.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_15.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_16.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_17.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_18.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_19.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_20.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_21.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_22.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_23.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_24.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_25.csv')
jeu_donnees_jeu_loaded += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_26.csv')

jeu_donnees_teste_loaded_3H = []
jeu_donnees_jeu_loaded_3H = load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_1.csv')
jeu_donnees_jeu_loaded_3H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_2.csv')
jeu_donnees_jeu_loaded_3H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_3.csv')
jeu_donnees_jeu_loaded_3H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_4.csv')
jeu_donnees_jeu_loaded_3H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_5.csv')
jeu_donnees_jeu_loaded_3H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_6.csv')
jeu_donnees_jeu_loaded_3H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_7.csv')
jeu_donnees_jeu_loaded_3H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_8.csv')
jeu_donnees_jeu_loaded_3H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_9.csv')
jeu_donnees_jeu_loaded_3H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_10.csv')
jeu_donnees_jeu_loaded_3H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_11.csv')
jeu_donnees_jeu_loaded_3H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_12.csv')
jeu_donnees_jeu_loaded_3H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_13.csv')
jeu_donnees_jeu_loaded_3H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_14.csv')
jeu_donnees_jeu_loaded_3H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_15.csv')
jeu_donnees_jeu_loaded_3H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_16.csv')
jeu_donnees_jeu_loaded_3H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_17.csv')
jeu_donnees_jeu_loaded_3H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_18.csv')
jeu_donnees_jeu_loaded_3H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_19.csv')
jeu_donnees_jeu_loaded_3H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_3H_20.csv')

jeu_donnees_teste_loaded_5H = []
jeu_donnees_jeu_loaded_5H = load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_5H_1.csv')
jeu_donnees_jeu_loaded_5H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_5H_2.csv')
jeu_donnees_jeu_loaded_5H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_5H_3.csv')
jeu_donnees_jeu_loaded_5H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_5H_4.csv')
jeu_donnees_jeu_loaded_5H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_5H_5.csv')
jeu_donnees_jeu_loaded_5H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_5H_6.csv')
jeu_donnees_jeu_loaded_5H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_5H_7.csv')
jeu_donnees_jeu_loaded_5H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_5H_8.csv')
jeu_donnees_jeu_loaded_5H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_5H_9.csv')
jeu_donnees_jeu_loaded_5H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_5H_10.csv')
jeu_donnees_jeu_loaded_5H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_5H_11.csv')
jeu_donnees_jeu_loaded_5H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_5H_12.csv')
jeu_donnees_jeu_loaded_5H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_5H_13.csv')
jeu_donnees_jeu_loaded_5H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_5H_14.csv')
jeu_donnees_jeu_loaded_5H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_5H_15.csv')
jeu_donnees_jeu_loaded_5H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_5H_16.csv')
jeu_donnees_jeu_loaded_5H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_5H_17.csv')
jeu_donnees_jeu_loaded_5H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_5H_18.csv')

jeu_donnees_teste_loaded_24H = []
jeu_donnees_jeu_loaded_24H = load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_24H_1.csv')
jeu_donnees_jeu_loaded_24H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_24H_2.csv')
jeu_donnees_jeu_loaded_24H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_24H_3.csv')
jeu_donnees_jeu_loaded_24H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_24H_4.csv')
jeu_donnees_jeu_loaded_24H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_24H_5.csv')
jeu_donnees_jeu_loaded_24H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_24H_6.csv')
jeu_donnees_jeu_loaded_24H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_24H_7.csv')
jeu_donnees_jeu_loaded_24H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_24H_8.csv')
jeu_donnees_jeu_loaded_24H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_24H_9.csv')
jeu_donnees_jeu_loaded_24H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_24H_10.csv')
jeu_donnees_jeu_loaded_24H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_24H_11.csv')
jeu_donnees_jeu_loaded_24H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_24H_12.csv')
jeu_donnees_jeu_loaded_24H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_24H_13.csv')
jeu_donnees_jeu_loaded_24H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_24H_14.csv')
jeu_donnees_jeu_loaded_24H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_24H_15.csv')
jeu_donnees_jeu_loaded_24H += load_weather_data_from_csv('jeu_donnees_jeu_pour_ia_24H_16.csv')

shuffle(jeu_donnees_jeu_loaded)
shuffle(jeu_donnees_jeu_loaded_3H)
shuffle(jeu_donnees_jeu_loaded_5H)
shuffle(jeu_donnees_jeu_loaded_24H)

#Initialisation/formatage jeu de données/ testes

for jeu in jeu_donnees_jeu_loaded:
    jeu[0][9] = 0.0001
    for sous_jeu in range(len(jeu)):
        jeu[sous_jeu][1] = round(float(jeu[sous_jeu][1]),1)
        jeu[sous_jeu][5] = float(jeu[sous_jeu][5])/100
        
        valu = float(jeu[sous_jeu][3])
        if valu >= 337.5 or valu <= 22.5:
            jeu[sous_jeu][3] = 0
        elif valu <= 76.5:
            jeu[sous_jeu][3] = 1
        elif valu <= 112.5:
            jeu[sous_jeu][3] = 2
        elif valu <= 157.5:
            jeu[sous_jeu][3] = 3
        elif valu <= 205.5:
           jeu[sous_jeu][3] = 4
        elif valu <= 247.5:
            jeu[sous_jeu][3] = 5
        elif valu <= 292.5:
            jeu[sous_jeu][3] = 6
        elif valu <337.5:
            jeu[sous_jeu][3] = 7
        if type(sous_jeu) != list:
            jeu[sous_jeu] = copy.deepcopy(jeu[sous_jeu-2])
        
for jeu in jeu_donnees_jeu_loaded_3H:
    jeu[0][9] = 0.0001
    for sous_jeu in range(len(jeu)):
        jeu[sous_jeu][1] = round(float(jeu[sous_jeu][1]),1)
        jeu[sous_jeu][5] = float(jeu[sous_jeu][5])/100
        
        valu = float(jeu[sous_jeu][3])
        if valu >= 337.5 or valu <= 22.5:
            jeu[sous_jeu][3] = 0
        elif valu <= 76.5:
            jeu[sous_jeu][3] = 1
        elif valu <= 112.5:
            jeu[sous_jeu][3] = 2
        elif valu <= 157.5:
            jeu[sous_jeu][3] = 3
        elif valu <= 205.5:
           jeu[sous_jeu][3] = 4
        elif valu <= 247.5:
            jeu[sous_jeu][3] = 5
        elif valu <= 292.5:
            jeu[sous_jeu][3] = 6
        elif valu <337.5:
            jeu[sous_jeu][3] = 7
        if type(sous_jeu) != list:
            jeu[sous_jeu] = copy.deepcopy(jeu[sous_jeu-2])
    
for jeu in jeu_donnees_jeu_loaded_5H:
    jeu[0][9] = 0.0001
    for sous_jeu in range(len(jeu)):
        jeu[sous_jeu][1] = round(float(jeu[sous_jeu][1]),1)
        jeu[sous_jeu][5] = float(jeu[sous_jeu][5])/100
        
        valu = float(jeu[sous_jeu][3])
        if valu >= 337.5 or valu <= 22.5:
            jeu[sous_jeu][3] = 0
        elif valu <= 76.5:
            jeu[sous_jeu][3] = 1
        elif valu <= 112.5:
            jeu[sous_jeu][3] = 2
        elif valu <= 157.5:
            jeu[sous_jeu][3] = 3
        elif valu <= 205.5:
           jeu[sous_jeu][3] = 4
        elif valu <= 247.5:
            jeu[sous_jeu][3] = 5
        elif valu <= 292.5:
            jeu[sous_jeu][3] = 6
        elif valu <337.5:
            jeu[sous_jeu][3] = 7
        if type(sous_jeu) != list:
            jeu[sous_jeu] = copy.deepcopy(jeu[sous_jeu-2])

for jeu in jeu_donnees_jeu_loaded_24H:
    jeu[0][9] = 0.0001
    for sous_jeu in range(len(jeu)):
        jeu[sous_jeu][1] = round(float(jeu[sous_jeu][1]),1)
        jeu[sous_jeu][5] = float(jeu[sous_jeu][5])/100
        
        valu = float(jeu[sous_jeu][3])
        if valu >= 337.5 or valu <= 22.5:
            jeu[sous_jeu][3] = 0
        elif valu <= 76.5:
            jeu[sous_jeu][3] = 1
        elif valu <= 112.5:
            jeu[sous_jeu][3] = 2
        elif valu <= 157.5:
            jeu[sous_jeu][3] = 3
        elif valu <= 205.5:
           jeu[sous_jeu][3] = 4
        elif valu <= 247.5:
            jeu[sous_jeu][3] = 5
        elif valu <= 292.5:
            jeu[sous_jeu][3] = 6
        elif valu <337.5:
            jeu[sous_jeu][3] = 7
        if type(sous_jeu) != list:
            jeu[sous_jeu] = copy.deepcopy(jeu[sous_jeu-2])

for i in range(100):
    jeu_donnees_teste_loaded.append(jeu_donnees_jeu_loaded[0])
    jeu_donnees_teste_loaded_3H.append(jeu_donnees_jeu_loaded_3H[0])
    jeu_donnees_teste_loaded_5H.append(jeu_donnees_jeu_loaded_5H[0])
    jeu_donnees_teste_loaded_24H.append(jeu_donnees_jeu_loaded_24H[0])
    
    jeu_donnees_jeu_loaded.pop(0)
    jeu_donnees_jeu_loaded_3H.pop(0)
    jeu_donnees_jeu_loaded_5H.pop(0)
    jeu_donnees_jeu_loaded_5H.pop(0)
    jeu_donnees_jeu_loaded_24H.pop(0)

jeu_teste_V_1H = prep_V(jeu_donnees_teste_loaded)
jeu_teste_V_3H = prep_V(jeu_donnees_teste_loaded_3H)
jeu_teste_V_5H = prep_V(jeu_donnees_teste_loaded_5H)
jeu_teste_V_24H = prep_V(jeu_donnees_teste_loaded_24H)
   

city_name = 'Cherbourg'
api_key = '4a315459a71c408a35c309ffa9c7796d'

#On créer 8 IA avec 3 couches et 1 output et chacune d'entre elle va donner une donnée.
    #Pour 1H
L_8_1H_3C = []
L_out_8_1H_3C = []
    #Pour 3H
L_8_3H_3C = []
L_out_8_3H_3C = []
    #1H, variation
L_8_1H_3C_V = []
L_out_8_1H_3C_V = []
    #3h,variation
L_8_3H_3C_V = []
L_out_8_3H_3C_V = []
    #5H
L_8_5H_3C = []
L_out_8_5H_3C = []

    #5H, variation
L_8_5H_3C_V = []
L_out_8_5H_3C_V = []

    #24H
L_8_24H_3C = []
L_out_8_24H_3C = []

    #24H, variation
L_8_24H_3C_V = []
L_out_8_24H_3C_V = []

for i in range(8):
    L_1H_3C, L_out_1H_3C = init_params(3, 12, 1)
    L_3H_3C, L_out_3H_3C = init_params(3, 12, 1)
    L_1H_3C_V, L_out_1H_3C_V = init_params(3, 12, 1)
    
    
    L_8_1H_3C.append(L_1H_3C)
    L_8_3H_3C.append(L_1H_3C)
    L_8_1H_3C_V.append(L_1H_3C)
    L_8_3H_3C_V.append(L_1H_3C)
    L_8_5H_3C.append(L_1H_3C)
    L_8_5H_3C_V.append(L_1H_3C)
    L_8_24H_3C.append(L_1H_3C)
    L_8_24H_3C_V.append(L_1H_3C)
    
    L_out_8_1H_3C.append(L_out_1H_3C)
    L_out_8_3H_3C.append(L_out_1H_3C)
    L_out_8_1H_3C_V.append(L_out_1H_3C)
    L_out_8_3H_3C_V.append(L_out_1H_3C)
    L_out_8_5H_3C.append(L_out_1H_3C)
    L_out_8_5H_3C_V.append(L_out_1H_3C)
    L_out_8_24H_3C.append(L_out_1H_3C)
    L_out_8_24H_3C_V.append(L_out_1H_3C)

# sauvegarde réseaux:  
L_8_1H_3C = [[[[0.04649921265687298, -1.7257937022514118], [0.1859968506274919, -0.5300914785585952], [0.23249606328501515, -0.5300914785585952]], [[-0.04649921265687298, 0.6656107451183045], [-0.23249606328501515, -0.9286588864508954], [-0.13949763797121964, 0.2670433372244069]], [[-0.09299842531374596, 1.064178152994688], [-0.13949763797121964, 1.4627455608869881], [-0.09299842531374596, -0.13152407066771654]], [[-0.09299842531374596, 1.4627455608869881], [0.09299842531374596, -0.13152407066771654], [0.04649921265687298, 1.4627455608869881]], [[-0.04649921265687298, -0.5300914785585952], [0.1859968506274919, 1.8613129687792882], [-0.04649921265687298, -0.9286588864508954]], [[-0.13949763797121964, -1.3272262943591118], [-0.13949763797121964, 0.2670433372244069], [-0.09299842531374596, 1.8613129687792882]], [[-0.1859968506274919, -0.13152407066771654], [-0.23249606328501515, 0.2670433372244069], [-0.23249606328501515, -1.3272262943591118]], [[-0.23249606328501515, -0.9286588864508954], [0.1859968506274919, -0.13152407066771654], [0.1859968506274919, -2.124361110117063]], [[-0.1859968506274919, -0.5300914785585952], [-0.04649921265687298, -1.3272262943591118], [0.04649921265687298, 0.2670433372244069]], [[0.09299842531374596, 1.064178152994688], [-0.23249606328501515, -1.3272262943591118], [-0.04649921265687298, 0.6656107451183045]], [[-0.23249606328501515, 1.064178152994688], [0.09299842531374596, -1.3272262943591118], [0.09299842531374596, 1.4627455608869881]], [[-0.09299842531374596, 1.4627455608869881], [-0.23249606328501515, 1.064178152994688], [0.13949763797121964, -1.3272262943591118]]], [[[0.06409293206487447, 2.124282621900845], [0.0, 0.9285803982267335], [0.32046466032428095, -1.4628240491297113]], [[-0.19227879619459198, -0.2671218254499861], [0.19227879619459198, -1.8613914570220111], [-0.2563717282594979, 1.3271478061163888]], [[0.0, 1.7257152140086887], [-0.12818586412974894, -0.2671218254499861], [0.19227879619459198, -0.2671218254499861]], [[-0.19227879619459198, -0.2671218254499861], [-0.2563717282594979, 0.5300129903345634], [0.32046466032428095, -0.6656892333424664]], [[0.32046466032428095, 1.3271478061163888], [-0.32046466032428095, -1.4628240491297113], [0.2563717282594979, -1.8613914570220111]], [[0.0, -1.8613914570220111], [0.19227879619459198, -0.2671218254499861], [-0.06409293206487447, -1.064256641237411]], [[0.0, 1.7257152140086887], [0.19227879619459198, -1.4628240491297113], [-0.06409293206487447, 0.13144558244072665]], [[-0.12818586412974894, 0.13144558244072665], [-0.19227879619459198, 0.13144558244072665], [0.0, -1.8613914570220111]], [[0.0, 1.7257152140086887], [0.0, 1.3271478061163888], [-0.06409293206487447, 0.5300129903345634]], [[-0.32046466032428095, 2.124282621900845], [-0.19227879619459198, 0.9285803982267335], [-0.06409293206487447, 1.7257152140086887]], [[-0.12818586412974894, 0.5300129903345634], [0.06409293206487447, 0.9285803982267335], [0.06409293206487447, -1.8613914570220111]], [[0.06409293206487447, -0.6656892333424664], [-0.12818586412974894, 1.7257152140086887], [0.0, 0.9285803982267335]]], [[[-0.16935925473236063, 0.7847495017085017], [0.04233981368309016, 0.7847495017085017], [-0.2116990684137475, 1.5818843174876653]], [[-0.2116990684137475, -2.0052223535265554], [0.08467962736618032, 0.3861820938189051], [0.04233981368309016, -0.8095201298606983]], [[-0.2116990684137475, 1.5818843174876653], [-0.16935925473236063, 1.183316909595365], [-0.1270194410475742, -1.2080875377584348]], [[0.2116990684137475, 0.7847495017085017], [-0.2116990684137475, 1.9804517253799652], [0.2116990684137475, -1.2080875377584348]], [[-0.08467962736618032, -0.4109527219656949], [-0.1270194410475742, 0.3861820938189051], [0.08467962736618032, -0.8095201298606983]], [[0.04233981368309016, -2.0052223535265554], [-0.2116990684137475, 1.183316909595365], [-0.1270194410475742, -1.2080875377584348]], [[0.2116990684137475, -1.6066549456507346], [0.1270194410475742, 0.3861820938189051], [-0.04233981368309016, 0.7847495017085017]], [[-0.04233981368309016, -0.8095201298606983], [-0.2116990684137475, -0.4109527219656949], [0.1270194410475742, 1.183316909595365]], [[0.16935925473236063, 1.5818843174876653], [0.0, 0.3861820938189051], [0.08467962736618032, 0.7847495017085017]], [[0.08467962736618032, -2.0052223535265554], [0.16935925473236063, -0.8095201298606983], [0.2116990684137475, -0.4109527219656949]], [[-0.2116990684137475, -0.012385314075743257], [0.0, -0.4109527219656949], [-0.04233981368309016, 0.7847495017085017]], [[0.0, -2.0052223535265554], [0.04233981368309016, 0.3861820938189051], [0.1270194410475742, 1.9804517253799652]]], [[[1.9862663322636415e+19, 49.913387874859666], [-4.965665830659367e+19, 49.913387874859666], [-2.9793994983942988e+19, 49.11625305906749]], [[-1.9862663322636415e+19, 51.50765750644301], [-1.9862663322636415e+19, 51.50765750644301], [3.972532664527283e+19, 49.51482046696335]], [[1.9862663322636415e+19, 50.31195528275556], [4.965665830659367e+19, 50.31195528275556], [-9.931331661318207e+18, 49.51482046696335]], [[0.0, 48.717685651171415], [-3.972532664527283e+19, 51.10909009854733], [-3.972532664527283e+19, 50.31195528275556]], [[3.972532664527283e+19, 49.913387874859666], [0.0, 49.913387874859666], [-2.9793994983942988e+19, 51.50765750644301]], [[0.0, 49.913387874859666], [2.9793994983942988e+19, 51.50765750644301], [-3.972532664527283e+19, 49.11625305906749]], [[-9.931331661318207e+18, 51.10909009854733], [-1.9862663322636415e+19, 50.31195528275556], [-4.965665830659367e+19, 48.717685651171415]], [[-4.965665830659367e+19, 50.31195528275556], [2.9793994983942988e+19, 49.11625305906749], [-4.965665830659367e+19, 52.304792322234896]], [[4.965665830659367e+19, 49.51482046696335], [-9.931331661318207e+18, 52.304792322234896], [-1.9862663322636415e+19, 49.51482046696335]], [[0.0, 51.90622491433904], [-9.931331661318207e+18, 52.304792322234896], [4.965665830659367e+19, 50.31195528275556]], [[3.972532664527283e+19, 49.51482046696335], [0.0, 50.31195528275556], [9.931331661318207e+18, 48.717685651171415]], [[1.9862663322636415e+19, 49.51482046696335], [-3.972532664527283e+19, 48.319118243275454], [9.931331661318207e+18, 48.319118243275454]]], [[[-30587.337148069262, 14.83025937690007], [-91762.01144423368, 14.43169196900905], [-152936.68574039705, 14.43169196900905]], [[122349.34859227705, 12.040287521660476], [122349.34859227705, 14.43169196900905], [30587.337148069262, 14.43169196900905]], [[-122349.34859227705, 12.837422337445352], [-152936.68574039705, 11.641720113768097], [152936.68574039705, 11.243152705875742]], [[61174.674296138524, 13.235989745336376], [0.0, 11.641720113768097], [122349.34859227705, 11.641720113768097]], [[-30587.337148069262, 12.040287521660476], [-61174.674296138524, 12.438854929552987], [-30587.337148069262, 13.634557153227115]], [[91762.01144423368, 11.641720113768097], [-152936.68574039705, 14.43169196900905], [61174.674296138524, 14.83025937690007]], [[-152936.68574039705, 13.634557153227115], [-61174.674296138524, 12.040287521660476], [91762.01144423368, 12.837422337445352]], [[-91762.01144423368, 12.438854929552987], [152936.68574039705, 12.438854929552987], [-61174.674296138524, 13.235989745336376]], [[91762.01144423368, 13.235989745336376], [152936.68574039705, 14.83025937690007], [122349.34859227705, 14.43169196900905]], [[30587.337148069262, 14.43169196900905], [0.0, 12.438854929552987], [-91762.01144423368, 12.040287521660476]], [[0.0, 13.235989745336376], [-91762.01144423368, 15.228826784790972], [91762.01144423368, 14.43169196900905]], [[91762.01144423368, 11.641720113768097], [30587.337148069262, 11.641720113768097], [61174.674296138524, 11.641720113768097]]], [[[0.7718906839621456, 2.970914195506775], [-0.5789180129718875, -0.21762506763139594], [-0.9648633549528477, -0.6161924755232985]], [[0.3859453419810728, 0.18094234026072503], [0.0, 0.18094234026072503], [0.3859453419810728, 0.5795097481534454]], [[-0.3859453419810728, 2.5723467876147303], [-0.7718906839621456, 0.5795097481534454], [-0.1929726709905364, -0.21762506763139594]], [[0.1929726709905364, 2.5723467876147303], [-0.3859453419810728, 0.9780771560456737], [0.9648633549528477, 0.9780771560456737]], [[-0.5789180129718875, 2.5723467876147303], [-0.5789180129718875, -0.6161924755232985], [0.1929726709905364, 1.3766445639396447]], [[0.7718906839621456, -0.21762506763139594], [0.5789180129718875, 0.5795097481534454], [-0.7718906839621456, 2.970914195506775]], [[0.3859453419810728, 2.5723467876147303], [0.0, 2.5723467876147303], [-0.9648633549528477, -0.21762506763139594]], [[-0.1929726709905364, 2.5723467876147303], [0.1929726709905364, 0.18094234026072503], [0.7718906839621456, 2.173779379722922]], [[-0.7718906839621456, 1.775211971831858], [0.1929726709905364, 1.775211971831858], [-0.3859453419810728, 0.18094234026072503]], [[0.5789180129718875, 3.36948160339879], [0.0, -0.6161924755232985], [-0.5789180129718875, 0.9780771560456737]], [[-0.1929726709905364, -0.6161924755232985], [-0.1929726709905364, 2.173779379722922], [-0.1929726709905364, 0.18094234026072503]], [[0.9648633549528477, 0.5795097481534454], [-0.7718906839621456, 2.970914195506775], [0.1929726709905364, 0.9780771560456737]]], [[[-0.15381349945670958, 1.5025162921934225], [0.15381349945670958, -1.2874555630526774], [-0.15381349945670958, -2.084590378837266]], [[-0.10254233297113992, -0.4903207482052365], [0.15381349945670958, -0.8888881553784359], [0.05127116648556996, -0.4903207482052365]], [[0.15381349945670958, -1.6860229709449774], [0.05127116648556996, -1.6860229709449774], [0.0, -0.09175334031293648]], [[0.05127116648556996, -2.084590378837266], [-0.2563558324278478, 0.7053814761907641], [0.05127116648556996, 1.1039488843011225]], [[-0.05127116648556996, -1.6860229709449774], [-0.10254233297113992, 1.5025162921934225], [-0.15381349945670958, -0.4903207482052365]], [[-0.15381349945670958, 1.5025162921934225], [0.20508466594227984, 0.7053814761907641], [0.2563558324278478, 0.7053814761907641]], [[-0.05127116648556996, 1.1039488843011225], [-0.10254233297113992, -2.084590378837266], [0.05127116648556996, 0.30681406757936347]], [[-0.15381349945670958, 0.7053814761907641], [0.0, -1.6860229709449774], [0.0, -0.4903207482052365]], [[0.0, 1.5025162921934225], [-0.10254233297113992, 0.30681406757936347], [-0.05127116648556996, -1.6860229709449774]], [[-0.20508466594227984, -0.8888881553784359], [0.2563558324278478, -0.09175334031293648], [0.05127116648556996, -1.6860229709449774]], [[-0.2563558324278478, 1.1039488843011225], [0.0, 0.7053814761907641], [-0.20508466594227984, 1.5025162921934225]], [[0.05127116648556996, -0.8888881553784359], [-0.10254233297113992, 1.1039488843011225], [-0.15381349945670958, -0.4903207482052365]]], [[[0.346672470586096, -0.58628280009141], [-0.06933449411721936, 0.21085201569319265], [0.20800348235165775, 2.2036890551546877]], [[-0.06933449411721936, -1.3834176158760094], [0.2773379764688774, 1.8051216472623905], [0.0, -0.58628280009141]], [[-0.06933449411721936, 0.6094194235854901], [0.346672470586096, 0.6094194235854901], [0.346672470586096, -0.18771539219910738]], [[0.2773379764688774, -0.18771539219910738], [0.1386689882344387, 1.0079868314777904], [0.06933449411721936, 2.2036890551546877]], [[-0.20800348235165775, -1.3834176158760094], [0.06933449411721936, 1.4065542393700905], [-0.1386689882344387, -0.9848502079837099]], [[0.2773379764688774, -0.58628280009141], [0.20800348235165775, 0.21085201569319265], [0.0, -0.9848502079837099]], [[0.346672470586096, 1.4065542393700905], [0.346672470586096, -0.9848502079837099], [0.2773379764688774, -0.58628280009141]], [[0.346672470586096, 2.2036890551546877], [-0.2773379764688774, 2.2036890551546877], [-0.346672470586096, -1.3834176158760094]], [[-0.1386689882344387, -1.3834176158760094], [-0.20800348235165775, -1.3834176158760094], [0.346672470586096, 0.21085201569319265]], [[-0.20800348235165775, 1.8051216472623905], [-0.20800348235165775, -0.18771539219910738], [-0.1386689882344387, 1.4065542393700905]], [[0.0, -0.58628280009141], [-0.346672470586096, 1.4065542393700905], [0.06933449411721936, 1.0079868314777904]], [[0.20800348235165775, 2.2036890551546877], [-0.1386689882344387, -0.58628280009141], [-0.06933449411721936, -0.9848502079837099]]]]

L_out_8_1H_3C = [[[141.86823916798969, 152.26650830815973]], [[3.850333285902043, -14.800084320510162]], [[9.605999981577876, 7.020279955998147]], [[-0.42446993983397174, 2.6555630421794287]], [[3.502793548542953, -24.163465117418045]], [[-1.8493724464287482, 10.448874039411432]], [[-3.5838485726558127, 18.033797640569503]], [[1274.6523630262654, 1246.1205501114507]]]


L_8_3H_3C = [[[[-0.5406577746154171, 2.77585759352452], [-0.7208770328163553, 1.5801553698660957], [-0.36043851640817764, 1.1815879619737955]], [[-0.7208770328163553, -0.8112490774987119], [0.5406577746154171, -0.4126816696067297], [0.5406577746154171, 2.3772901856322206]], [[0.5406577746154171, -0.4126816696067297], [-0.18021925820408882, 0.3844531461778702], [0.9010962910179003, 2.3772901856322206]], [[-0.9010962910179003, 2.3772901856322206], [-0.7208770328163553, -0.4126816696067297], [0.5406577746154171, -0.014114261712326203]], [[-0.9010962910179003, -0.014114261712326203], [-0.7208770328163553, 0.7830205540704883], [0.18021925820408882, -0.8112490774987119]], [[0.0, -0.8112490774987119], [0.9010962910179003, 2.3772901856322206], [0.18021925820408882, 1.5801553698660957]], [[0.0, 0.7830205540704883], [-0.18021925820408882, 2.77585759352452], [0.36043851640817764, 0.7830205540704883]], [[-0.7208770328163553, -0.8112490774987119], [-0.18021925820408882, 0.7830205540704883], [0.36043851640817764, -0.8112490774987119]], [[-0.36043851640817764, 1.5801553698660957], [0.0, 1.1815879619737955], [-0.18021925820408882, 1.1815879619737955]], [[0.5406577746154171, 0.3844531461778702], [-0.5406577746154171, 0.7830205540704883], [-0.7208770328163553, 1.5801553698660957]], [[0.9010962910179003, 2.3772901856322206], [0.7208770328163553, 1.5801553698660957], [-0.7208770328163553, 2.77585759352452]], [[0.5406577746154171, 1.1815879619737955], [-0.18021925820408882, -0.014114261712326203], [-0.7208770328163553, 0.3844531461778702]]], [[[0.05144771274874918, -0.8853929106067859], [0.10289542549749836, 0.3103093130709729], [-0.10289542549749836, -0.8853929106067859]], [[0.10289542549749836, -2.0810951342811745], [0.0, -0.08825809481727692], [0.0, 1.5060115367690439]], [[-0.05144771274874918, 1.107444128876744], [-0.10289542549749836, -0.8853929106067859], [-0.25723856374392956, 1.9045789446613441]], [[-0.10289542549749836, 0.3103093130709729], [0.25723856374392956, 1.5060115367690439], [0.20579085099499672, -1.2839603184770558]], [[-0.05144771274874918, -0.08825809481727692], [0.15434313824598145, -1.2839603184770558], [0.20579085099499672, -1.2839603184770558]], [[-0.20579085099499672, -0.8853929106067859], [0.15434313824598145, 1.9045789446613441], [-0.25723856374392956, -1.2839603184770558]], [[0.25723856374392956, -2.0810951342811745], [-0.25723856374392956, -0.8853929106067859], [0.15434313824598145, -1.2839603184770558]], [[-0.20579085099499672, -1.2839603184770558], [-0.05144771274874918, 1.5060115367690439], [0.20579085099499672, -0.08825809481727692]], [[-0.05144771274874918, 0.3103093130709729], [0.05144771274874918, -2.0810951342811745], [0.10289542549749836, -0.4868255027136575]], [[-0.25723856374392956, 0.3103093130709729], [0.25723856374392956, -0.08825809481727692], [-0.15434313824598145, -0.08825809481727692]], [[0.20579085099499672, -0.08825809481727692], [0.25723856374392956, 0.3103093130709729], [0.10289542549749836, -1.2839603184770558]], [[-0.20579085099499672, -2.0810951342811745], [0.10289542549749836, 1.107444128876744], [0.10289542549749836, -0.8853929106067859]]], [[[0.10396361437026973, -0.013192461959452007], [0.10396361437026973, 0.7839423538240269], [-0.20792722874053945, -0.013192461959452007]], [[-0.1559454215577522, -0.013192461959452007], [-0.10396361437026973, 1.5810771696155808], [-0.10396361437026973, 1.979644577507881]], [[0.0, 0.38537494593136856], [-0.051981807185134864, 0.38537494593136856], [-0.051981807185134864, 0.38537494593136856]], [[0.0, 1.182509761723281], [-0.20792722874053945, -0.013192461959452007], [-0.2599090359305242, 1.182509761723281]], [[0.10396361437026973, -0.41175986985323143], [0.10396361437026973, -2.006029501471607], [0.0, 1.5810771696155808]], [[0.10396361437026973, -1.6074620935228192], [-0.10396361437026973, 1.182509761723281], [0.2599090359305242, 1.979644577507881]], [[-0.20792722874053945, -0.013192461959452007], [0.2599090359305242, 0.38537494593136856], [0.10396361437026973, -0.8103272777451731]], [[0.0, 1.182509761723281], [-0.051981807185134864, -1.208894685630519], [-0.051981807185134864, -0.8103272777451731]], [[-0.20792722874053945, -0.013192461959452007], [-0.2599090359305242, -0.41175986985323143], [-0.20792722874053945, -0.41175986985323143]], [[-0.10396361437026973, -1.208894685630519], [0.2599090359305242, -1.208894685630519], [-0.10396361437026973, 1.182509761723281]], [[0.2599090359305242, -0.013192461959452007], [0.20792722874053945, -2.006029501471607], [0.0, 1.979644577507881]], [[0.10396361437026973, -2.006029501471607], [-0.1559454215577522, 0.38537494593136856], [0.2599090359305242, -2.006029501471607]]], [[[4601186.002582365, 19.93749685817734], [-18404744.01032946, 20.336064266063737], [18404744.01032946, 20.336064266063737]], [[-13803558.007737523, 16.350390187199856], [18404744.01032946, 16.350390187199856], [-13803558.007737523, 20.336064266063737]], [[-13803558.007737523, 19.140362042404313], [18404744.01032946, 17.546092410859195], [-13803558.007737523, 17.147525002973595]], [[-18404744.01032946, 19.538929450290887], [-18404744.01032946, 19.538929450290887], [9202372.00516473, 16.748957595086907]], [[0.0, 19.538929450290887], [4601186.002582365, 18.343227226631186], [9202372.00516473, 17.94465981874459]], [[18404744.01032946, 17.94465981874459], [-13803558.007737523, 16.748957595086907], [18404744.01032946, 18.343227226631186]], [[-18404744.01032946, 17.147525002973595], [-23005930.012900613, 19.93749685817734], [-23005930.012900613, 16.748957595086907]], [[4601186.002582365, 17.546092410859195], [9202372.00516473, 17.546092410859195], [4601186.002582365, 17.94465981874459]], [[-18404744.01032946, 19.93749685817734], [4601186.002582365, 18.343227226631186], [9202372.00516473, 17.147525002973595]], [[9202372.00516473, 19.93749685817734], [13803558.007737523, 19.93749685817734], [9202372.00516473, 17.546092410859195]], [[13803558.007737523, 18.343227226631186], [-9202372.00516473, 16.748957595086907], [-23005930.012900613, 17.94465981874459]], [[-18404744.01032946, 16.350390187199856], [23005930.012900613, 20.336064266063737], [-4601186.002582365, 19.93749685817734]]], [[[-60.324489494650734, 5.542716548811142], [90.48673424196015, 6.339851364595494], [90.48673424196015, 5.542716548811142]], [[60.324489494650734, 7.136986180380201], [90.48673424196015, 6.738418772488253], [-30.162244747325367, 6.738418772488253]], [[120.64897898930147, 6.339851364595494], [30.162244747325367, 7.5355535882721965], [0.0, 6.738418772488253]], [[-150.81122373669825, 4.347014325134281], [60.324489494650734, 4.347014325134281], [-90.48673424196015, 4.347014325134281]], [[30.162244747325367, 5.941283956703274], [-60.324489494650734, 8.332688404073158], [150.81122373669825, 8.332688404073158]], [[120.64897898930147, 7.5355535882721965], [60.324489494650734, 7.934120996164576], [120.64897898930147, 8.332688404073158]], [[-30.162244747325367, 5.542716548811142], [-120.64897898930147, 7.934120996164576], [-90.48673424196015, 4.745581733026648]], [[120.64897898930147, 6.738418772488253], [30.162244747325367, 4.347014325134281], [-30.162244747325367, 7.136986180380201]], [[60.324489494650734, 4.745581733026648], [150.81122373669825, 8.332688404073158], [120.64897898930147, 5.144149140918818]], [[30.162244747325367, 5.542716548811142], [-150.81122373669825, 8.332688404073158], [-120.64897898930147, 7.934120996164576]], [[60.324489494650734, 8.332688404073158], [-90.48673424196015, 7.136986180380201], [0.0, 7.136986180380201]], [[-60.324489494650734, 6.738418772488253], [150.81122373669825, 5.144149140918818], [30.162244747325367, 7.934120996164576]]], [[[0.1339179158354594, 0.988809029445813], [0.0, 1.3873764373380002], [-0.1339179158354594, 1.7859438452303]], [[0.3347947895887587, 1.3873764373380002], [-0.3347947895887587, 0.19167421366070236], [0.1339179158354594, -1.8011628258004562]], [[-0.3347947895887587, -1.4025954179081], [-0.1339179158354594, -0.605460602123387], [0.0669589579177297, -0.20689319423161096]], [[0.2678358316709188, 1.3873764373380002], [0.20087687375320834, -0.20689319423161096], [-0.2678358316709188, 0.5902416215535367]], [[0.2678358316709188, 1.7859438452303], [0.0669589579177297, 1.3873764373380002], [-0.1339179158354594, 1.7859438452303]], [[-0.3347947895887587, 2.1845112531230995], [0.3347947895887587, 1.3873764373380002], [0.3347947895887587, 1.3873764373380002]], [[0.20087687375320834, -1.4025954179081], [-0.1339179158354594, -1.004028010015816], [-0.1339179158354594, 0.988809029445813]], [[0.20087687375320834, 2.1845112531230995], [-0.0669589579177297, 0.5902416215535367], [-0.2678358316709188, 1.7859438452303]], [[0.20087687375320834, -1.8011628258004562], [-0.2678358316709188, -0.605460602123387], [0.3347947895887587, 2.1845112531230995]], [[0.3347947895887587, -0.605460602123387], [0.3347947895887587, 1.3873764373380002], [-0.20087687375320834, 2.1845112531230995]], [[0.0, 1.7859438452303], [0.1339179158354594, -1.004028010015816], [-0.0669589579177297, -0.605460602123387]], [[0.0669589579177297, 1.7859438452303], [0.2678358316709188, 2.1845112531230995], [0.0, 0.5902416215535367]]], [[[0.16620353913953437, -2.2945485733245063], [-0.08310176956976718, -1.4974137575399147], [-0.08310176956976718, 0.09685587402927638]], [[0.0, -1.8959811654322147], [0.12465265435465067, 0.8939906898138797], [0.08310176956976718, 0.4954232819215783]], [[-0.20775442392441834, -0.700278941755321], [-0.08310176956976718, 0.4954232819215783], [-0.04155088478488359, -0.700278941755321]], [[-0.12465265435465067, 1.6911255055984853], [0.04155088478488359, -1.8959811654322147], [0.0, -0.3017115338630227]], [[0.16620353913953437, -0.700278941755321], [-0.04155088478488359, -2.2945485733245063], [0.04155088478488359, 0.09685587402927638]], [[-0.16620353913953437, 1.2925580977061852], [0.12465265435465067, 0.4954232819215783], [0.20775442392441834, 0.8939906898138797]], [[-0.08310176956976718, 1.6911255055984853], [0.0, -1.0988463496476153], [0.12465265435465067, -1.0988463496476153]], [[-0.20775442392441834, -0.700278941755321], [0.08310176956976718, -1.0988463496476153], [-0.16620353913953437, 0.4954232819215783]], [[0.08310176956976718, -0.3017115338630227], [-0.12465265435465067, -1.4974137575399147], [-0.16620353913953437, -2.2945485733245063]], [[0.16620353913953437, -1.0988463496476153], [-0.16620353913953437, -1.8959811654322147], [-0.08310176956976718, -1.8959811654322147]], [[0.04155088478488359, -1.8959811654322147], [0.20775442392441834, 0.4954232819215783], [-0.04155088478488359, -1.0988463496476153]], [[-0.08310176956976718, -1.0988463496476153], [0.20775442392441834, 0.4954232819215783], [0.12465265435465067, -0.3017115338630227]]], [[[1934.8283594575812, 12.54020207384308], [-1934.8283594575812, 8.953095402812378], [5804.4850801164, 12.938769481735381]], [[3869.6567189151624, 11.743067258058478], [-1934.8283594575812, 8.953095402812378], [0.0, 12.54020207384308]], [[5804.4850801164, 10.94593244227388], [-3869.6567189151624, 10.547365034381583], [-3869.6567189151624, 11.743067258058478]], [[7739.313437830325, 8.953095402812378], [0.0, 10.148797626489282], [7739.313437830325, 12.14163466595078]], [[0.0, 9.35166281070468], [-9674.141799896881, 9.750230218596982], [-5804.4850801164, 10.148797626489282]], [[-7739.313437830325, 12.54020207384308], [-3869.6567189151624, 10.148797626489282], [-3869.6567189151624, 9.35166281070468]], [[9674.141799896881, 12.54020207384308], [-3869.6567189151624, 12.14163466595078], [1934.8283594575812, 8.953095402812378]], [[9674.141799896881, 10.94593244227388], [-5804.4850801164, 10.547365034381583], [1934.8283594575812, 12.54020207384308]], [[5804.4850801164, 9.35166281070468], [-9674.141799896881, 10.547365034381583], [1934.8283594575812, 11.34449985016618]], [[-9674.141799896881, 9.750230218596982], [1934.8283594575812, 11.743067258058478], [-9674.141799896881, 10.94593244227388]], [[-7739.313437830325, 11.34449985016618], [-1934.8283594575812, 11.34449985016618], [3869.6567189151624, 10.148797626489282]], [[-3869.6567189151624, 12.54020207384308], [7739.313437830325, 12.14163466595078], [7739.313437830325, 9.35166281070468]]]]

L_out_8_3H_3C = [[[123.26283267888157, 132.73948309509186]], [[0.31349060543578927, 8.575906359340905]], [[10.119064856393944, 5.172558125085294]], [[1.263383458003855, -3.5819370511587545]], [[0.6282744255221645, 0.2905367294500602]], [[-0.19843547152221072, 1.7576063457276707]], [[-0.06169051784436794, 0.41006156333619914]], [[1107.8942963138893, 1107.8266136880102]]]

L_8_1H_3C_V = [[[[6.45420001364116e-254, -583.1865578979153], [6.45420001364116e-254, -580.7951534511627], [4.302800009096638e-254, -580.3965860433701]], [[-6.45420001364116e-254, -581.5922882667478], [8.605600018193276e-254, -579.2008838199941], [2.151400004548319e-254, -579.2008838199941]], [[-4.302800009096638e-254, -579.599451227786], [-4.302800009096638e-254, -579.9980186355778], [-8.605600018193276e-254, -579.2008838199941]], [[4.302800009096638e-254, -579.599451227786], [-4.302800009096638e-254, -582.3894230823321], [-1.0757000022732345e-253, -582.3894230823321]], [[2.151400004548319e-254, -579.599451227786], [1.0757000022732345e-253, -579.9980186355778], [-2.151400004548319e-254, -579.599451227786]], [[-8.605600018193276e-254, -582.7879904901237], [-1.0757000022732345e-253, -582.7879904901237], [-4.302800009096638e-254, -579.2008838199941]], [[1.0757000022732345e-253, -581.1937208589553], [4.302800009096638e-254, -579.9980186355778], [-8.605600018193276e-254, -581.9908556745402]], [[-2.151400004548319e-254, -581.9908556745402], [2.151400004548319e-254, -582.7879904901237], [8.605600018193276e-254, -580.7951534511627]], [[4.302800009096638e-254, -581.5922882667478], [8.605600018193276e-254, -581.5922882667478], [1.0757000022732345e-253, -580.7951534511627]], [[-6.45420001364116e-254, -582.7879904901237], [-4.302800009096638e-254, -581.9908556745402], [0.0, -579.9980186355778]], [[8.605600018193276e-254, -580.7951534511627], [4.302800009096638e-254, -583.1865578979153], [1.0757000022732345e-253, -581.5922882667478]], [[-8.605600018193276e-254, -582.3894230823321], [4.302800009096638e-254, -579.9980186355778], [0.0, -581.5922882667478]]], [[[1.6794075584519444e-282, -648.0963159194711], [-1.6794075584519444e-282, -645.7049114727183], [-1.6794075584519444e-282, -645.7049114727183]], [[2.0992594480623342e-282, -646.9006136960941], [0.0, -648.8934507350542], [0.0, -645.3063440649269]], [[1.6794075584519444e-282, -648.8934507350542], [0.0, -647.6977485116787], [0.0, -645.7049114727183]], [[1.6794075584519444e-282, -646.1034788805098], [-4.198518896129861e-283, -646.1034788805098], [-4.198518896129861e-283, -648.8934507350542]], [[-1.2595556688372224e-282, -648.4948833272628], [-8.397037792259722e-283, -648.8934507350542], [2.0992594480623342e-282, -646.5020462883018]], [[-4.198518896129861e-283, -649.2920181428459], [4.198518896129861e-283, -647.6977485116787], [-4.198518896129861e-283, -645.7049114727183]], [[0.0, -645.7049114727183], [1.2595556688372224e-282, -646.1034788805098], [-8.397037792259722e-283, -647.6977485116787]], [[1.2595556688372224e-282, -645.3063440649269], [4.198518896129861e-283, -646.1034788805098], [-8.397037792259722e-283, -647.2991811038864]], [[4.198518896129861e-283, -646.5020462883018], [-1.6794075584519444e-282, -648.8934507350542], [1.2595556688372224e-282, -646.9006136960941]], [[-8.397037792259722e-283, -646.1034788805098], [-4.198518896129861e-283, -646.5020462883018], [2.0992594480623342e-282, -646.1034788805098]], [[-1.6794075584519444e-282, -647.6977485116787], [4.198518896129861e-283, -645.7049114727183], [-2.0992594480623342e-282, -647.6977485116787]], [[8.397037792259722e-283, -646.1034788805098], [-1.2595556688372224e-282, -645.7049114727183], [4.198518896129861e-283, -646.5020462883018]]], [[[0.0, -0.8218916078333985], [-0.05462398995222663, -1.2204590157259492], [0.0, 1.9680802474124508]], [[0.05462398995222663, 0.3738106158430865], [0.21849595980890651, -1.2204590157259492], [0.0, 0.7723780237358014]], [[-0.2731199497610156, -0.8218916078333985], [0.10924797990445326, 0.7723780237358014], [0.05462398995222663, 0.3738106158430865]], [[-0.16387196985671157, -0.8218916078333985], [0.05462398995222663, -1.2204590157259492], [0.05462398995222663, 1.9680802474124508]], [[0.21849595980890651, 0.3738106158430865], [0.10924797990445326, -2.0175938315106077], [0.16387196985671157, -1.619026423618249]], [[0.16387196985671157, 1.1709454316278507], [0.10924797990445326, 1.569512839520151], [-0.16387196985671157, 1.1709454316278507]], [[-0.05462398995222663, -1.619026423618249], [-0.2731199497610156, -1.619026423618249], [-0.2731199497610156, -0.024756792203960834]], [[0.2731199497610156, 1.1709454316278507], [-0.10924797990445326, -0.8218916078333985], [-0.05462398995222663, 1.1709454316278507]], [[0.05462398995222663, -0.8218916078333985], [-0.05462398995222663, 1.569512839520151], [0.21849595980890651, -0.8218916078333985]], [[-0.16387196985671157, -2.0175938315106077], [-0.21849595980890651, 0.3738106158430865], [0.21849595980890651, -0.8218916078333985]], [[-0.21849595980890651, -0.4233241999415135], [0.0, -0.4233241999415135], [0.10924797990445326, -2.0175938315106077]], [[-0.10924797990445326, 1.569512839520151], [0.16387196985671157, 1.9680802474124508], [0.05462398995222663, -2.0175938315106077]]], [[[1203758064.3117707, 22.580207284068447], [-963006451.4498293, 21.783072468344358], [-240751612.86245733, 21.783072468344358]], [[481503225.72491467, 20.985937652621732], [-1203758064.3117707, 23.377342099794767], [-963006451.4498293, 20.985937652621732]], [[-963006451.4498293, 21.783072468344358], [-1203758064.3117707, 22.181639875900192], [-722254838.5871906, 20.58737024475929]], [[0.0, 20.188802836897775], [-240751612.86245733, 24.174476915516788], [-722254838.5871906, 22.181639875900192]], [[-963006451.4498293, 21.783072468344358], [-240751612.86245733, 21.384505060484], [-722254838.5871906, 20.58737024475929]], [[963006451.4498293, 20.985937652621732], [722254838.5871906, 20.58737024475929], [963006451.4498293, 22.181639875900192]], [[240751612.86245733, 21.384505060484], [-722254838.5871906, 20.985937652621732], [-240751612.86245733, 20.188802836897775]], [[-1203758064.3117707, 21.384505060484], [-963006451.4498293, 24.174476915516788], [0.0, 20.985937652621732]], [[240751612.86245733, 20.985937652621732], [-481503225.72491467, 20.985937652621732], [240751612.86245733, 22.580207284068447]], [[963006451.4498293, 20.985937652621732], [-963006451.4498293, 20.985937652621732], [722254838.5871906, 24.174476915516788]], [[0.0, 22.580207284068447], [240751612.86245733, 20.188802836897775], [963006451.4498293, 24.174476915516788]], [[963006451.4498293, 20.58737024475929], [-481503225.72491467, 20.188802836897775], [-240751612.86245733, 22.580207284068447]]], [[[-0.2603515690244903, 1.1264731124487224], [0.20828125521929972, 1.525040520319312], [-0.10414062760964986, 1.923607928190018]], [[-0.20828125521929972, 0.32933829669332754], [-0.15621094141463077, 0.7279057045832011], [-0.05207031380482493, -0.4677965190931079]], [[-0.15621094141463077, -1.6634987427840084], [0.15621094141463077, 1.525040520319312], [0.0, -2.0620661506823192]], [[0.15621094141463077, 1.923607928190018], [-0.10414062760964986, -1.6634987427840084], [-0.10414062760964986, -1.2649313348832243]], [[-0.20828125521929972, -0.06922911150891503], [-0.20828125521929972, -0.06922911150891503], [-0.15621094141463077, 1.525040520319312]], [[0.0, 0.32933829669332754], [0.20828125521929972, -0.4677965190931079], [0.10414062760964986, -1.2649313348832243]], [[-0.05207031380482493, -1.2649313348832243], [-0.20828125521929972, -0.06922911150891503], [-0.05207031380482493, -0.866363926985525]], [[-0.15621094141463077, -0.4677965190931079], [-0.15621094141463077, -2.0620661506823192], [0.0, -0.4677965190931079]], [[0.0, -1.6634987427840084], [0.0, -0.866363926985525], [-0.2603515690244903, -0.866363926985525]], [[0.05207031380482493, 1.923607928190018], [-0.10414062760964986, 1.525040520319312], [-0.2603515690244903, -0.4677965190931079]], [[-0.05207031380482493, 1.1264731124487224], [-0.10414062760964986, 1.1264731124487224], [-0.10414062760964986, 1.525040520319312]], [[-0.20828125521929972, -0.06922911150891503], [0.2603515690244903, 1.923607928190018], [0.10414062760964986, -2.0620661506823192]]], [[[-1.8212554087778668e-201, -458.7159307575451], [2.276569260973129e-201, -461.505902574673], [1.3659415565837442e-201, -461.505902574673]], [[4.553138521944667e-202, -460.7087677697805], [2.276569260973129e-201, -461.505902574673], [-9.106277043889334e-202, -460.7087677697805]], [[1.3659415565837442e-201, -460.7087677697805], [-1.3659415565837442e-201, -462.3030373795672], [4.553138521944667e-202, -461.9044699771203]], [[-4.553138521944667e-202, -462.7016047820174], [9.106277043889334e-202, -461.9044699771203], [1.8212554087778668e-201, -458.7159307575451]], [[-9.106277043889334e-202, -460.7087677697805], [4.553138521944667e-202, -459.9116329648849], [-4.553138521944667e-202, -459.51306556243685]], [[1.3659415565837442e-201, -461.505902574673], [4.553138521944667e-202, -461.9044699771203], [9.106277043889334e-202, -461.1073351722271]], [[-1.3659415565837442e-201, -459.51306556243685], [0.0, -459.51306556243685], [-1.8212554087778668e-201, -462.3030373795672]], [[1.8212554087778668e-201, -460.3102003673338], [4.553138521944667e-202, -461.9044699771203], [-1.8212554087778668e-201, -461.9044699771203]], [[9.106277043889334e-202, -461.9044699771203], [-1.3659415565837442e-201, -460.3102003673338], [2.276569260973129e-201, -461.505902574673]], [[9.106277043889334e-202, -461.9044699771203], [-1.3659415565837442e-201, -459.1144981599905], [9.106277043889334e-202, -462.7016047820174]], [[-1.3659415565837442e-201, -462.3030373795672], [2.276569260973129e-201, -459.51306556243685], [1.3659415565837442e-201, -459.1144981599905]], [[2.276569260973129e-201, -461.9044699771203], [1.3659415565837442e-201, -462.3030373795672], [-9.106277043889334e-202, -461.1073351722271]]], [[[0.0, -498.6780462297888], [-2.000243364912157e-218, -497.88091143401425], [1.3334955766079916e-218, -499.475181025565]], [[-2.6669911532159833e-218, -498.6780462297888], [-2.6669911532159833e-218, -500.272315821341], [-1.3334955766079916e-218, -499.0766136276765]], [[0.0, -498.6780462297888], [-2.000243364912157e-218, -498.6780462297888], [-1.3334955766079916e-218, -497.48234403612736]], [[-1.3334955766079916e-218, -498.6780462297888], [-3.3337389415195495e-218, -498.6780462297888], [-3.3337389415195495e-218, -500.67088321922756]], [[2.000243364912157e-218, -498.6780462297888], [3.3337389415195495e-218, -499.87374842345304], [-2.000243364912157e-218, -499.0766136276765]], [[-3.3337389415195495e-218, -501.4680180150015], [-1.3334955766079916e-218, -497.48234403612736], [-3.3337389415195495e-218, -500.67088321922756]], [[3.3337389415195495e-218, -497.48234403612736], [2.000243364912157e-218, -497.48234403612736], [-2.6669911532159833e-218, -497.88091143401425]], [[1.3334955766079916e-218, -499.87374842345304], [-1.3334955766079916e-218, -500.67088321922756], [-3.3337389415195495e-218, -500.67088321922756]], [[3.3337389415195495e-218, -499.87374842345304], [-2.6669911532159833e-218, -499.87374842345304], [-2.000243364912157e-218, -500.67088321922756]], [[6.667477883039958e-219, -500.272315821341], [0.0, -501.06945061711474], [-2.6669911532159833e-218, -497.48234403612736]], [[-1.3334955766079916e-218, -497.88091143401425], [-6.667477883039958e-219, -501.4680180150015], [-6.667477883039958e-219, -500.272315821341]], [[1.3334955766079916e-218, -497.48234403612736], [2.000243364912157e-218, -501.06945061711474], [-1.3334955766079916e-218, -499.87374842345304]]], [[[0.0, -1.6394687485223112], [0.2685228584123697, -0.8423339327368161], [-0.10740914336486103, -0.04519911695244333]], [[-0.10740914336486103, 1.947637922508389], [-0.21481828672972206, -0.04519911695244333], [-0.21481828672972206, -1.240901340630011]], [[-0.053704571682430514, -1.240901340630011], [-0.2685228584123697, 1.947637922508389], [-0.053704571682430514, 0.3533682909399697]], [[0.10740914336486103, 1.150503106723789], [0.0, -1.6394687485223112], [-0.10740914336486103, 1.5490705146160888]], [[-0.053704571682430514, 1.150503106723789], [-0.21481828672972206, -0.04519911695244333], [-0.16111371504730418, 1.5490705146160888]], [[-0.16111371504730418, 0.7519356988323839], [-0.16111371504730418, -1.6394687485223112], [0.10740914336486103, 1.947637922508389]], [[0.16111371504730418, 1.150503106723789], [0.053704571682430514, 1.5490705146160888], [-0.16111371504730418, 1.150503106723789]], [[0.0, -0.4437665248446303], [0.053704571682430514, -1.240901340630011], [0.2685228584123697, -0.4437665248446303]], [[-0.16111371504730418, 1.5490705146160888], [-0.16111371504730418, 0.7519356988323839], [-0.053704571682430514, -2.0380361564134493]], [[0.0, -1.240901340630011], [0.16111371504730418, -0.04519911695244333], [-0.053704571682430514, -2.0380361564134493]], [[0.21481828672972206, -1.240901340630011], [-0.10740914336486103, 1.5490705146160888], [0.2685228584123697, 1.150503106723789]], [[0.16111371504730418, 0.7519356988323839], [0.2685228584123697, -0.8423339327368161], [0.0, 0.7519356988323839]]]]

L_out_8_1H_3C_V = [[[-5.87911519718363, -4.702445083287098e-05]], [[-0.6967773858860616, -9.198509965006825e-05]], [[-0.6312657190663883, 4.8095539687654725]], [[-9.712637858130444, -0.00011139773513808383]], [[4.435082654285687, -16.834558573881367]], [[-15.524617787543102, -3.0346996688762932e-05]], [[-7.52079634152472, -0.0001877389960680673]], [[-1.6455107884940043, 10.911172613240478]]]


L_8_3H_3C_V = [[[[-0.05289129139658483,-0.059842931929298654], [0.2644564569830538, -1.2555451570730924], [0.2644564569830538, -1.2555451570730924]], [[-0.05289129139658483, -0.8569777481171045], [0.2115651655863393, -0.8569777481171045], [0.15867387418964057, -0.8569777481171045]], [[0.05289129139658483, -0.059842931929298654], [-0.10578258279316965, -2.0526799737044237], [-0.2115651655863393, -0.059842931929298654]], [[0.2644564569830538, -0.8569777481171045], [-0.10578258279316965, -0.8569777481171045], [-0.05289129139658483, -0.8569777481171045]], [[0.05289129139658483, -1.2555451570730924], [0.0, 1.5344266999161598], [0.0, 0.737291884062972]], [[-0.2644564569830538, -0.8569777481171045], [-0.15867387418964057, -0.8569777481171045], [0.2115651655863393, 1.5344266999161598]], [[-0.10578258279316965, -1.2555451570730924], [-0.2644564569830538, -0.059842931929298654], [0.10578258279316965, -0.4584103399275539]], [[0.10578258279316965, -0.4584103399275539], [-0.10578258279316965, -0.059842931929298654], [-0.05289129139658483, 0.737291884062972]], [[0.2644564569830538, 1.135859292049391], [-0.15867387418964057, 0.33872447607272477], [0.2644564569830538, -2.0526799737044237]], [[0.0, -0.4584103399275539], [0.2115651655863393, 0.33872447607272477], [-0.2115651655863393, -1.6541125657221798]], [[0.15867387418964057, 1.5344266999161598], [-0.2644564569830538, 0.33872447607272477], [-0.2115651655863393, -1.6541125657221798]], [[-0.15867387418964057, 1.5344266999161598], [-0.10578258279316965, -2.0526799737044237], [-0.15867387418964057, 0.737291884062972]]], [[[0.21448417251310098, -0.4431142973526597], [0.10724208625655049, 1.5497227421076707], [-0.16086312938481465, 1.5497227421076707]], [[-0.21448417251310098, -2.0373839289185325], [0.053621043128275245, 0.35402051843194027], [0.0, 0.75258792632449]], [[-0.21448417251310098, -2.0373839289185325], [-0.2681052156408963, 1.5497227421076707], [-0.053621043128275245, 1.948290149999971]], [[-0.16086312938481465, 0.35402051843194027], [-0.053621043128275245, -1.6388165210307293], [-0.21448417251310098, 1.1511553342153709]], [[-0.16086312938481465, -0.4431142973526597], [0.16086312938481465, 1.5497227421076707], [0.16086312938481465, -1.240249113138429]], [[0.2681052156408963, -2.0373839289185325], [0.0, -1.240249113138429], [-0.10724208625655049, 0.75258792632449]], [[0.21448417251310098, -0.04454688946036591], [0.10724208625655049, -1.240249113138429], [-0.10724208625655049, 0.35402051843194027]], [[0.0, -0.04454688946036591], [-0.2681052156408963, 1.5497227421076707], [-0.053621043128275245, -0.04454688946036591]], [[-0.21448417251310098, -2.0373839289185325], [-0.10724208625655049, 0.35402051843194027], [0.16086312938481465, -1.6388165210307293]], [[0.0, -1.6388165210307293], [0.16086312938481465, 0.35402051843194027], [0.053621043128275245, 1.948290149999971]], [[-0.053621043128275245, 1.5497227421076707], [-0.21448417251310098, -0.04454688946036591], [-0.10724208625655049, -0.84168170524471]], [[-0.053621043128275245, 1.1511553342153709], [-0.10724208625655049, -0.4431142973526597], [-0.10724208625655049, 1.1511553342153709]]], [[[553430130567.1202, 27.737379736354878], [553430130567.1202, 29.33164936767977], [276715065283.5601, 27.338812328522852]], [[-691787663209.1627, 30.527351591172778], [553430130567.1202, 28.13594714418785], [276715065283.5601, 29.33164936767977]], [[138357532641.78006, 28.13594714418785], [-691787663209.1627, 30.128784183341807], [-415072597925.4804, 26.94024492069202]], [[-138357532641.78006, 28.13594714418785], [0.0, 30.128784183341807], [-415072597925.4804, 27.338812328522852]], [[415072597925.4804, 26.541677512862215], [276715065283.5601, 28.933081959848828], [-138357532641.78006, 29.33164936767977]], [[-276715065283.5601, 30.527351591172778], [553430130567.1202, 28.933081959848828], [-276715065283.5601, 27.338812328522852]], [[-553430130567.1202, 26.94024492069202], [-415072597925.4804, 29.33164936767977], [0.0, 29.730216775511074]], [[0.0, 30.527351591172778], [138357532641.78006, 26.541677512862215], [691787663209.1627, 27.737379736354878]], [[-691787663209.1627, 26.94024492069202], [276715065283.5601, 28.13594714418785], [-691787663209.1627, 30.128784183341807]], [[-138357532641.78006, 27.737379736354878], [691787663209.1627, 30.527351591172778], [691787663209.1627, 27.338812328522852]], [[415072597925.4804, 26.94024492069202], [138357532641.78006, 29.33164936767977], [-553430130567.1202, 29.730216775511074]], [[0.0, 27.338812328522852], [691787663209.1627, 28.13594714418785], [-553430130567.1202, 27.338812328522852]]], [[[0.21375714686588143, -0.44732692638308913], [0.2671964335821126, 1.9440775209703678], [0.21375714686588143, -2.0415965579542594]], [[-0.2671964335821126, -2.0415965579542594], [-0.05343928671647036, -1.2444617421680322], [0.16031786014936353, 0.34980788940151086]], [[-0.05343928671647036, 1.1469427051857677], [-0.10687857343294072, 1.1469427051857677], [-0.05343928671647036, 1.5455101130780675]], [[0.0, 0.34980788940151086], [-0.10687857343294072, 1.9440775209703678], [0.2671964335821126, 0.7483752972938824]], [[-0.05343928671647036, -0.8458943342753176], [0.05343928671647036, -1.6430291500603325], [0.05343928671647036, -0.04875951848292577]], [[0.05343928671647036, -2.0415965579542594], [-0.21375714686588143, 0.34980788940151086], [0.0, -0.44732692638308913]], [[-0.05343928671647036, -0.44732692638308913], [-0.10687857343294072, -1.2444617421680322], [-0.16031786014936353, -0.44732692638308913]], [[0.05343928671647036, 0.34980788940151086], [-0.05343928671647036, -0.04875951848292577], [0.16031786014936353, -1.6430291500603325]], [[-0.10687857343294072, 0.7483752972938824], [0.10687857343294072, 1.1469427051857677], [-0.21375714686588143, 1.1469427051857677]], [[-0.10687857343294072, -1.6430291500603325], [-0.2671964335821126, 1.1469427051857677], [0.16031786014936353, 0.7483752972938824]], [[-0.16031786014936353, -0.04875951848292577], [-0.2671964335821126, 1.1469427051857677], [-0.21375714686588143, -0.44732692638308913]], [[-0.16031786014936353, 0.34980788940151086], [-0.21375714686588143, -1.2444617421680322], [-0.21375714686588143, 1.5455101130780675]]], [[[5.021716334902961e-07, -15.223658369871528], [-3.0130298009408507e-07, -13.629388735817296], [1.0043432669803396e-07, -14.426523551745607]], [[3.0130298009408507e-07, -12.035119104094061], [3.0130298009408507e-07, -11.237984288233601], [-5.021716334902961e-07, -15.223658369871528]], [[2.0086865339606793e-07, -13.23082132788648], [2.0086865339606793e-07, -14.426523551745607], [-2.0086865339606793e-07, -12.83225391995581]], [[-1.0043432669803396e-07, -14.027956143748161], [1.0043432669803396e-07, -13.23082132788648], [2.0086865339606793e-07, -14.027956143748161]], [[3.0130298009408507e-07, -12.433686512024952], [3.0130298009408507e-07, -11.636551696163862], [5.021716334902961e-07, -11.237984288233601]], [[1.0043432669803396e-07, -14.027956143748161], [4.0173730679213586e-07, -12.433686512024952], [1.0043432669803396e-07, -12.83225391995581]], [[-3.0130298009408507e-07, -13.629388735817296], [-2.0086865339606793e-07, -11.636551696163862], [-5.021716334902961e-07, -12.83225391995581]], [[3.0130298009408507e-07, -14.825090960808604], [-1.0043432669803396e-07, -12.433686512024952], [2.0086865339606793e-07, -14.426523551745607]], [[5.021716334902961e-07, -13.23082132788648], [-3.0130298009408507e-07, -14.426523551745607], [-5.021716334902961e-07, -12.433686512024952]], [[3.0130298009408507e-07, -15.223658369871528], [3.0130298009408507e-07, -13.629388735817296], [4.0173730679213586e-07, -11.237984288233601]], [[5.021716334902961e-07, -12.433686512024952], [0.0, -11.636551696163862], [-4.0173730679213586e-07, -15.223658369871528]], [[-3.0130298009408507e-07, -14.426523551745607], [0.0, -12.035119104094061], [5.021716334902961e-07, -14.426523551745607]]], [[[0.15747135069774904, -0.06634578112077477], [-0.05249045023265095, -0.46491318901319834], [0.26245225116280135, -1.6606154126895833]], [[0.15747135069774904, -0.06634578112077477], [-0.1049809004653019, -0.8634805969063812], [-0.05249045023265095, 1.1293564425565163]], [[0.1049809004653019, -0.8634805969063812], [0.26245225116280135, 1.9264912583411165], [0.05249045023265095, -0.46491318901319834]], [[0.0, -1.6606154126895833], [-0.26245225116280135, -0.46491318901319834], [-0.26245225116280135, -1.6606154126895833]], [[0.0, 1.1293564425565163], [-0.26245225116280135, 1.1293564425565163], [0.1049809004653019, -1.2620480047972835]], [[0.26245225116280135, 0.7307890346628187], [0.15747135069774904, 0.7307890346628187], [0.26245225116280135, -0.06634578112077477]], [[0.2099618009306038, -0.06634578112077477], [0.2099618009306038, 0.33222162677140166], [0.05249045023265095, -0.46491318901319834]], [[0.0, -0.46491318901319834], [0.0, -0.06634578112077477], [0.0, -1.2620480047972835]], [[0.2099618009306038, 0.7307890346628187], [-0.15747135069774904, -0.8634805969063812], [-0.26245225116280135, -2.0591828205826253]], [[0.05249045023265095, 1.1293564425565163], [0.05249045023265095, 1.9264912583411165], [-0.2099618009306038, -0.46491318901319834]], [[-0.05249045023265095, 1.1293564425565163], [0.15747135069774904, 1.5279238504488166], [-0.15747135069774904, -1.6606154126895833]], [[-0.15747135069774904, -2.0591828205826253], [0.0, 1.1293564425565163], [0.0, -0.8634805969063812]]], [[[0.21075659767296648, 0.7334598069316773], [-0.10537829883648324, -0.4622424152230096], [-0.15806744825496216, -2.0565120446905003]], [[-0.15806744825496216, 1.1320272148832216], [0.2634457470912298, 0.33489239897839235], [0.15806744825496216, 0.7334598069316773]], [[-0.10537829883648324, -0.8608098212953852], [0.21075659767296648, -2.0565120446905003], [-0.21075659767296648, -0.4622424152230096]], [[-0.2634457470912298, -2.0565120446905003], [-0.05268914941824162, 1.9291620307539887], [0.15806744825496216, 1.5305946228185754]], [[0.2634457470912298, -0.4622424152230096], [-0.05268914941824162, -1.259377229097608], [0.0, -2.0565120446905003]], [[0.21075659767296648, -0.8608098212953852], [-0.10537829883648324, 0.33489239897839235], [-0.2634457470912298, -1.657944636895411]], [[-0.21075659767296648, -0.4622424152230096], [0.15806744825496216, 0.33489239897839235], [-0.21075659767296648, -1.259377229097608]], [[-0.21075659767296648, 1.1320272148832216], [0.10537829883648324, 0.7334598069316773], [0.21075659767296648, 1.9291620307539887]], [[-0.15806744825496216, -2.0565120446905003], [0.15806744825496216, 0.7334598069316773], [0.15806744825496216, 1.1320272148832216]], [[-0.2634457470912298, -0.8608098212953852], [-0.10537829883648324, -1.657944636895411], [0.0, -2.0565120446905003]], [[-0.05268914941824162, 0.7334598069316773], [-0.10537829883648324, -0.8608098212953852], [0.0, -1.657944636895411]], [[-0.21075659767296648, 1.9291620307539887], [-0.2634457470912298, -1.259377229097608], [-0.15806744825496216, 1.5305946228185754]]], [[[-0.26971768973643573, 1.5533137504905699], [-0.16183061384186534, 1.95188115838287], [0.26971768973643573, -2.033792920540473]], [[0.107887075894587, -0.04095588107821344], [-0.16183061384186534, -0.4395232889705965], [0.16183061384186534, 1.5533137504905699]], [[0.107887075894587, -2.033792920540473], [-0.16183061384186534, 1.5533137504905699], [0.26971768973643573, 1.1547463425982698]], [[0.0539435379472935, 1.95188115838287], [0.0539435379472935, -0.8380906968629787], [0.16183061384186534, 1.1547463425982698]], [[0.215774151789174, 0.3576115268140035], [-0.26971768973643573, -1.6352255126478301], [0.107887075894587, 0.3576115268140035]], [[0.0, -1.6352255126478301], [0.16183061384186534, -0.4395232889705965], [0.215774151789174, 1.95188115838287]], [[0.0, -0.04095588107821344], [0.16183061384186534, 0.3576115268140035], [0.26971768973643573, -0.04095588107821344]], [[-0.107887075894587, -2.033792920540473], [0.0539435379472935, -2.033792920540473], [-0.0539435379472935, 1.5533137504905699]], [[0.0539435379472935, -1.23665810475553], [-0.0539435379472935, -2.033792920540473], [0.16183061384186534, -0.4395232889705965]], [[-0.215774151789174, -0.4395232889705965], [-0.215774151789174, -0.4395232889705965], [-0.16183061384186534, -0.04095588107821344]], [[-0.0539435379472935, 1.5533137504905699], [0.215774151789174, 1.95188115838287], [-0.107887075894587, -0.8380906968629787]], [[-0.107887075894587, 1.95188115838287], [0.0539435379472935, -1.6352255126478301], [0.0539435379472935, -0.04095588107821344]]]]

L_out_8_3H_3C_V = [[[0.35634289773560823, -1.331075838081925]], [[0.3209247059546997, -1.59019799070723]], [[-2.044910546831035, -5.721717015062867e-05]], [[0.542108577305682, -2.9304972135286724]], [[-0.2855960030846932, -8.603684982176502e-05]], [[0.08138474330520783, 0.20499675408982343]], [[0.5697996592179015, -2.3238561211266244]], [[0.024169393731210846, 0.3040729128089291]]]


L_8_5H_3C = [[[[0.0, 1.1958542551231024], [-0.11241458842472324, 1.1958542551231024], [0.0, 1.5944216630154024]], [[0.0, -1.5941176001229975], [0.11241458842472324, -0.796982784313542], [-0.22482917684944648, 0.797286847255658]], [[0.2810364710664424, 0.3987194393537913], [-0.16862188263503777, 1.5944216630154024], [0.2810364710664424, -1.9926850080152976]], [[-0.11241458842472324, 0.797286847255658], [0.16862188263503777, -0.796982784313542], [-0.05620729421236162, -1.9926850080152976]], [[0.22482917684944648, -1.1955501922306975], [-0.05620729421236162, -1.5941176001229975], [0.0, 0.3987194393537913]], [[-0.05620729421236162, 0.797286847255658], [0.05620729421236162, 1.9929890709077025], [0.16862188263503777, -0.796982784313542]], [[-0.22482917684944648, 1.5944216630154024], [0.22482917684944648, 1.9929890709077025], [0.22482917684944648, -1.1955501922306975]], [[-0.16862188263503777, -0.796982784313542], [-0.16862188263503777, 1.1958542551231024], [0.16862188263503777, -1.1955501922306975]], [[-0.05620729421236162, -0.796982784313542], [0.05620729421236162, -1.9926850080152976], [0.2810364710664424, -1.1955501922306975]], [[0.05620729421236162, -1.1955501922306975], [-0.16862188263503777, 0.797286847255658], [0.05620729421236162, -1.1955501922306975]], [[-0.05620729421236162, -1.1955501922306975], [0.22482917684944648, -1.9926850080152976], [0.0, 0.00015203146385726137]], [[0.22482917684944648, 1.9929890709077025], [-0.2810364710664424, 0.00015203146385726137], [-0.2810364710664424, -0.796982784313542]]], [[[-0.0561984993293249, 1.5942651785408104], [0.2809924966476083, 0.797130362735745], [-0.2247939973172996, -4.453042712477464e-06]], [[0.2247939973172996, 0.797130362735745], [0.0, -1.9928414924898896], [0.2247939973172996, -4.453042712477464e-06]], [[0.2247939973172996, 0.3985629548430815], [-0.1123969986586498, 1.9928325864331105], [0.16859549798863688, 1.9928325864331105]], [[-0.0561984993293249, 0.797130362735745], [0.2247939973172996, -0.797139268833455], [-0.0561984993293249, 1.9928325864331105]], [[0.2809924966476083, 1.9928325864331105], [0.0561984993293249, 1.1956977706485103], [-0.2809924966476083, -0.797139268833455]], [[0.0, -0.3985718609415185], [-0.2247939973172996, -4.453042712477464e-06], [0.2247939973172996, 1.1956977706485103]], [[0.2247939973172996, 1.1956977706485103], [-0.2809924966476083, -1.9928414924898896], [-0.16859549798863688, 1.9928325864331105]], [[-0.2247939973172996, -1.1957066767052895], [-0.16859549798863688, -1.9928414924898896], [-0.2809924966476083, 1.9928325864331105]], [[-0.0561984993293249, -1.9928414924898896], [0.0561984993293249, -4.453042712477464e-06], [-0.16859549798863688, -4.453042712477464e-06]], [[0.16859549798863688, -0.3985718609415185], [-0.1123969986586498, 1.5942651785408104], [0.16859549798863688, -1.5942740845975896]], [[-0.1123969986586498, -1.1957066767052895], [-0.2809924966476083, 1.5942651785408104], [0.2809924966476083, 1.1956977706485103]], [[0.2247939973172996, 0.797130362735745], [-0.2809924966476083, -4.453042712477464e-06], [-0.0561984993293249, 1.9928325864331105]]], [[[0.2809962872868551, -1.594260594498384], [-0.16859777237832688, -0.3985583708346631], [-0.11239851491649167, -0.7971257787156903]], [[-0.2809962872868551, -1.594260594498384], [-0.05619925745824583, 1.5942786686400159], [-0.05619925745824583, -1.594260594498384]], [[-0.2809962872868551, 1.992846076532316], [-0.11239851491649167, -1.594260594498384], [-0.11239851491649167, -0.7971257787156903]], [[0.16859777237832688, -1.594260594498384], [0.11239851491649167, -0.3985583708346631], [0.11239851491649167, 1.1957112607477158]], [[-0.05619925745824583, 0.3985764449499369], [-0.11239851491649167, -0.3985583708346631], [-0.11239851491649167, 1.992846076532316]], [[0.11239851491649167, -0.7971257787156903], [-0.11239851491649167, -1.195693186606084], [-0.11239851491649167, 9.037055934464193e-06]], [[-0.11239851491649167, -1.195693186606084], [-0.2809962872868551, 1.1957112607477158], [-0.11239851491649167, -0.7971257787156903]], [[0.2809962872868551, 1.5942786686400159], [0.16859777237832688, -1.9928280023906841], [0.0, 9.037055934464193e-06]], [[-0.2809962872868551, 1.1957112607477158], [-0.22479702983298333, 0.3985764449499369], [-0.05619925745824583, 1.5942786686400159]], [[-0.05619925745824583, -0.7971257787156903], [-0.11239851491649167, 1.1957112607477158], [-0.16859777237832688, -1.9928280023906841]], [[0.2809962872868551, 1.992846076532316], [-0.2809962872868551, 0.7971438528535096], [0.05619925745824583, 1.992846076532316]], [[0.2809962872868551, 1.992846076532316], [0.11239851491649167, -0.7971257787156903], [0.05619925745824583, 9.037055934464193e-06]]], [[[0.225187428324237, -1.9910689376061472], [0.0, 1.1974703255317336], [-0.2814842854053705, -0.39679930603917646]], [[-0.2814842854053705, 1.1974703255317336], [-0.1688905712432459, 1.9946051413163337], [0.0, -1.1939341218220663]], [[0.1688905712432459, 0.40033550974542353], [0.225187428324237, 1.5960377334240334], [0.1688905712432459, -1.9910689376061472]], [[0.05629685708105925, -1.1939341218220663], [-0.2814842854053705, 0.7989029176377409], [0.0, -1.5925015297143665]], [[-0.2814842854053705, -1.1939341218220663], [0.1125937141621185, 0.40033550974542353], [0.2814842854053705, 1.1974703255317336]], [[0.225187428324237, -0.7953667139314591], [0.1688905712432459, -1.1939341218220663], [0.05629685708105925, 1.9946051413163337]], [[-0.1688905712432459, 1.1974703255317336], [0.1688905712432459, 0.0017681018532761026], [0.1125937141621185, -1.1939341218220663]], [[0.05629685708105925, 0.40033550974542353], [-0.1688905712432459, -1.1939341218220663], [-0.1125937141621185, -1.9910689376061472]], [[0.1125937141621185, -1.5925015297143665], [-0.1688905712432459, -1.9910689376061472], [0.05629685708105925, -1.5925015297143665]], [[0.0, -0.7953667139314591], [-0.1125937141621185, -1.9910689376061472], [0.1688905712432459, 1.9946051413163337]], [[0.1688905712432459, 0.40033550974542353], [0.2814842854053705, -1.5925015297143665], [0.225187428324237, 1.1974703255317336]], [[-0.225187428324237, -0.7953667139314591], [-0.1125937141621185, 0.7989029176377409], [0.1688905712432459, 0.7989029176377409]]], [[[0.11315075889494781, 1.6009827026264898], [0.22630151778989563, 0.8038478868418503], [0.11315075889494781, -1.18898915261961]], [[-0.11315075889494781, -0.7904217447273497], [0.22630151778989563, -1.18898915261961], [0.0, 1.20241529473419]], [[0.22630151778989563, 1.6009827026264898], [0.0, 0.00671307105590263], [-0.2828768972375136, -1.986123968403709]], [[0.22630151778989563, -0.7904217447273497], [-0.2828768972375136, -0.3918543368354732], [-0.16972613834257042, 1.6009827026264898]], [[0.22630151778989563, 0.00671307105590263], [-0.22630151778989563, 1.99955011051879], [-0.11315075889494781, -1.5875565605119102]], [[-0.11315075889494781, 1.20241529473419], [0.16972613834257042, -1.5875565605119102], [0.0, -0.3918543368354732]], [[0.05657537944747391, 1.20241529473419], [0.05657537944747391, 0.4052804789491268], [-0.16972613834257042, -1.5875565605119102]], [[-0.2828768972375136, -0.3918543368354732], [0.16972613834257042, 0.8038478868418503], [0.05657537944747391, -0.3918543368354732]], [[0.2828768972375136, 1.99955011051879], [-0.11315075889494781, -1.18898915261961], [0.22630151778989563, 0.8038478868418503]], [[-0.16972613834257042, -1.986123968403709], [-0.11315075889494781, 0.4052804789491268], [0.22630151778989563, 1.6009827026264898]], [[0.0, 1.20241529473419], [0.22630151778989563, -0.7904217447273497], [0.0, -0.7904217447273497]], [[0.05657537944747391, 1.20241529473419], [-0.11315075889494781, 1.20241529473419], [0.2828768972375136, 0.4052804789491268]]], [[[0.26663924512631354, -0.8495617906402919], [-0.26663924512631354, 1.541842656713602], [-0.21331139610116903, -2.045264014315459]], [[-0.21331139610116903, 0.7447078409289081], [-0.26663924512631354, -0.8495617906402919], [-0.26663924512631354, -1.6466966064247979]], [[0.10665569805058452, -1.6466966064247979], [0.26663924512631354, -0.05242697485575611], [-0.1599835470758076, -2.045264014315459]], [[0.26663924512631354, 1.940410064605902], [-0.21331139610116903, 0.34614043303671954], [-0.05332784902529226, -0.45099438274788045]], [[0.21331139610116903, -1.6466966064247979], [-0.05332784902529226, -1.6466966064247979], [0.21331139610116903, -0.45099438274788045]], [[-0.05332784902529226, -1.248129198532498], [-0.26663924512631354, -1.248129198532498], [0.26663924512631354, -0.8495617906402919]], [[-0.10665569805058452, 0.7447078409289081], [-0.05332784902529226, 1.541842656713602], [0.05332784902529226, -0.45099438274788045]], [[-0.1599835470758076, 1.541842656713602], [0.21331139610116903, -2.045264014315459], [0.26663924512631354, -1.6466966064247979]], [[-0.10665569805058452, 1.541842656713602], [0.1599835470758076, 1.1432752488213018], [-0.21331139610116903, -0.05242697485575611]], [[-0.10665569805058452, 1.1432752488213018], [0.26663924512631354, -0.8495617906402919], [0.0, -1.6466966064247979]], [[0.21331139610116903, -1.248129198532498], [0.21331139610116903, 1.940410064605902], [-0.1599835470758076, 1.1432752488213018]], [[-0.21331139610116903, 0.34614043303671954], [0.05332784902529226, -0.8495617906402919], [0.26663924512631354, -0.05242697485575611]]], [[[-0.2126395249390981, -0.0555423940093843], [0.10631976246954905, -0.8526772097936091], [-0.10631976246954905, -1.6498120255783115]], [[0.10631976246954905, -0.0555423940093843], [0.1594796437044462, 0.3430250138824881], [-0.10631976246954905, 1.9372946454523883]], [[-0.1594796437044462, 1.5387272375600884], [0.1594796437044462, -0.0555423940093843], [0.10631976246954905, 1.9372946454523883]], [[0.0, -2.048379433471607], [0.2126395249390981, 1.5387272375600884], [0.1594796437044462, 1.5387272375600884]], [[-0.053159881234774525, 0.7415924217755909], [-0.2126395249390981, 1.9372946454523883], [0.1594796437044462, -0.4541098019021119]], [[-0.053159881234774525, 1.5387272375600884], [0.26579940617367565, 1.1401598296677882], [0.053159881234774525, -1.2512446176860117]], [[0.2126395249390981, 0.3430250138824881], [0.10631976246954905, 1.5387272375600884], [-0.053159881234774525, 0.7415924217755909]], [[0.053159881234774525, -2.048379433471607], [-0.053159881234774525, -2.048379433471607], [0.2126395249390981, -1.2512446176860117]], [[-0.1594796437044462, 0.3430250138824881], [0.10631976246954905, -0.8526772097936091], [0.10631976246954905, -2.048379433471607]], [[-0.053159881234774525, 0.3430250138824881], [-0.2126395249390981, 1.5387272375600884], [-0.10631976246954905, 1.1401598296677882]], [[-0.2126395249390981, 1.1401598296677882], [-0.2126395249390981, -0.0555423940093843], [-0.1594796437044462, 1.1401598296677882]], [[0.26579940617367565, 0.3430250138824881], [-0.2126395249390981, 1.5387272375600884], [0.26579940617367565, -1.6498120255783115]]], [[[0.28140991005107163, -1.1942222494796368], [-0.05628198201011107, 0.40004738209918966], [0.0, -1.592789657371937]], [[-0.2251279280404443, -1.991357065264237], [0.28140991005107163, 0.7986147900010668], [0.2251279280404443, -0.7956548415681332]], [[0.11256396402022215, -1.991357065264237], [-0.1688459460301133, 0.7986147900010668], [0.11256396402022215, 1.9943170136587631]], [[0.28140991005107163, -1.592789657371937], [-0.05628198201011107, -1.592789657371937], [-0.11256396402022215, -0.39708743368541033]], [[-0.28140991005107163, -1.1942222494796368], [-0.28140991005107163, -0.39708743368541033], [0.11256396402022215, 0.7986147900010668]], [[-0.28140991005107163, -1.991357065264237], [0.05628198201011107, -1.1942222494796368], [0.28140991005107163, -1.991357065264237]], [[-0.2251279280404443, -0.39708743368541033], [0.11256396402022215, -1.592789657371937], [0.1688459460301133, 0.7986147900010668]], [[-0.2251279280404443, -1.592789657371937], [0.05628198201011107, 1.595749605766463], [0.0, 0.0014799742118151264]], [[0.11256396402022215, -0.39708743368541033], [0.05628198201011107, -1.592789657371937], [-0.11256396402022215, -1.991357065264237]], [[-0.05628198201011107, -0.7956548415681332], [0.11256396402022215, 0.40004738209918966], [-0.28140991005107163, -1.991357065264237]], [[0.0, 1.595749605766463], [0.1688459460301133, -0.39708743368541033], [-0.1688459460301133, 0.0014799742118151264]], [[0.05628198201011107, -0.7956548415681332], [-0.2251279280404443, -1.592789657371937], [0.05628198201011107, 0.0014799742118151264]]]]

L_out_8_5H_3C = [[[158.3585282674395, 154.93425950015592]], [[1.167132368748029, 1.2289383741403266]], [[13.427324028688226, 14.870734015900066]], [[0.8552334290077109, -1.5966208768064787]], [[0.6561214756987744, -0.09935728751419293]], [[-0.014547303237030079, 0.9387241858978639]], [[0.07987794536528053, -0.3226108557312603]], [[1451.7863738628444, 1451.5356643949822]]]


L_8_5H_3C_V = [[[[0.0, 1.1387571569400674], [-0.10617583717179478, 1.1387571569400674], [0.0, 1.5373245648323675]], [[0.0, -1.6512146983060325], [0.10617583717179478, -0.8540798825214484], [-0.21235167434358956, 0.7401897490477516]], [[0.26543959292947755, 0.34162234115535933], [-0.1592637557576922, 1.5373245648323675], [0.26543959292947755, -2.0497821061977537]], [[-0.10617583717179478, 0.7401897490477516], [0.1592637557576922, -0.8540798825214484], [-0.05308791858589739, -2.0497821061977537]], [[0.21235167434358956, -1.2526472904137325], [-0.05308791858589739, -1.6512146983060325], [0.0, 0.34162234115535933]], [[-0.05308791858589739, 0.7401897490477516], [0.05308791858589739, 1.9358919727246675], [0.1592637557576922, -0.8540798825214484]], [[-0.21235167434358956, 1.5373245648323675], [0.21235167434358956, 1.9358919727246675], [0.21235167434358956, -1.2526472904137325]], [[-0.1592637557576922, -0.8540798825214484], [-0.1592637557576922, 1.1387571569400674], [0.1592637557576922, -1.2526472904137325]], [[-0.05308791858589739, -0.8540798825214484], [0.05308791858589739, -2.0497821061977537], [0.26543959292947755, -1.2526472904137325]], [[0.05308791858589739, -1.2526472904137325], [-0.1592637557576922, 0.7401897490477516], [0.05308791858589739, -1.2526472904137325]], [[-0.05308791858589739, -1.2526472904137325], [0.21235167434358956, -2.0497821061977537], [0.0, -0.05694506674955645]], [[0.21235167434358956, 1.9358919727246675], [-0.26543959292947755, -0.05694506674955645], [-0.26543959292947755, -0.8540798825214484]]], [[[-0.05372949822611177, 1.54933733758963], [0.26864749113065267, 0.7522025218045723], [-0.21491799290444707, -0.044932293980477406]], [[0.21491799290444707, 0.7522025218045723], [0.0, -2.037769333440558], [0.21491799290444707, -0.044932293980477406]], [[0.21491799290444707, 0.35363511391203595], [-0.10745899645222354, 1.94790474548193], [0.16118849467836485, 1.94790474548193]], [[-0.05372949822611177, 0.7522025218045723], [0.21491799290444707, -0.8420671097646277], [-0.05372949822611177, 1.94790474548193]], [[0.26864749113065267, 1.94790474548193], [0.05372949822611177, 1.15076992969733], [-0.26864749113065267, -0.8420671097646277]], [[0.0, -0.44349970187256404], [-0.21491799290444707, -0.044932293980477406], [0.21491799290444707, 1.15076992969733]], [[0.21491799290444707, 1.15076992969733], [-0.26864749113065267, -2.037769333440558], [-0.16118849467836485, 1.94790474548193]], [[-0.21491799290444707, -1.24063451765647], [-0.16118849467836485, -2.037769333440558], [-0.26864749113065267, 1.94790474548193]], [[-0.05372949822611177, -2.037769333440558], [0.05372949822611177, -0.044932293980477406], [-0.16118849467836485, -0.044932293980477406]], [[0.16118849467836485, -0.44349970187256404], [-0.10745899645222354, 1.54933733758963], [0.16118849467836485, -1.63920192554877]], [[-0.10745899645222354, -1.24063451765647], [-0.26864749113065267, 1.54933733758963], [0.26864749113065267, 1.15076992969733]], [[0.21491799290444707, 0.7522025218045723], [-0.26864749113065267, -0.044932293980477406], [-0.05372949822611177, 1.94790474548193]]], [[[0.26564290246485533, -1.6504490571851598], [-0.15938574147890835, -0.45474683350828754], [-0.10625716098593649, -0.8533142414005955]], [[-0.26564290246485533, -1.6504490571851598], [-0.053128580492968244, 1.5380902059532402], [-0.053128580492968244, -1.6504490571851598]], [[-0.26564290246485533, 1.9366576138455402], [-0.10625716098593649, -1.6504490571851598], [-0.10625716098593649, -0.8533142414005955]], [[0.15938574147890835, -1.6504490571851598], [0.10625716098593649, -0.45474683350828754], [0.10625716098593649, 1.1395227980609401]], [[-0.053128580492968244, 0.34238798227631245], [-0.10625716098593649, -0.45474683350828754], [-0.10625716098593649, 1.9366576138455402]], [[0.10625716098593649, -0.8533142414005955], [-0.10625716098593649, -1.2518816492928597], [-0.10625716098593649, -0.05617942561600718]], [[-0.10625716098593649, -1.2518816492928597], [-0.26564290246485533, 1.1395227980609401], [-0.10625716098593649, -0.8533142414005955]], [[0.26564290246485533, 1.5380902059532402], [0.15938574147890835, -2.049016465076916], [0.0, -0.05617942561600718]], [[-0.26564290246485533, 1.1395227980609401], [-0.21251432197187298, 0.34238798227631245], [-0.053128580492968244, 1.5380902059532402]], [[-0.053128580492968244, -0.8533142414005955], [-0.10625716098593649, 1.1395227980609401], [-0.15938574147890835, -2.049016465076916]], [[0.26564290246485533, 1.9366576138455402], [-0.26564290246485533, 0.7409553901686045], [0.053128580492968244, 1.9366576138455402]], [[0.26564290246485533, 1.9366576138455402], [0.10625716098593649, -0.8533142414005955], [0.053128580492968244, -0.05617942561600718]]], [[[0.21318109482137512, -2.0458837936535637], [0.0, 1.1426554694848965], [-0.26647636852671375, -0.4516141620842919]], [[-0.26647636852671375, 1.1426554694848965], [-0.159885821116026, 1.9397902852694966], [0.0, -1.2487489778689034]], [[0.159885821116026, 0.3455206537003081], [0.21318109482137512, 1.5412228773771965], [0.159885821116026, -2.0458837936535637]], [[0.05329527370534378, -1.2487489778689034], [-0.26647636852671375, 0.7440880615926329], [0.0, -1.6473163857612034]], [[-0.26647636852671375, -1.2487489778689034], [0.10659054741068756, 0.3455206537003081], [0.26647636852671375, 1.1426554694848965]], [[0.21318109482137512, -0.8501815699765671], [0.159885821116026, -1.2487489778689034], [0.05329527370534378, 1.9397902852694966]], [[-0.159885821116026, 1.1426554694848965], [0.159885821116026, -0.05304675374434752], [0.10659054741068756, -1.2487489778689034]], [[0.05329527370534378, 0.3455206537003081], [-0.159885821116026, -1.2487489778689034], [-0.10659054741068756, -2.0458837936535637]], [[0.10659054741068756, -1.6473163857612034], [-0.159885821116026, -2.0458837936535637], [0.05329527370534378, -1.6473163857612034]], [[0.0, -0.8501815699765671], [-0.10659054741068756, -2.0458837936535637], [0.159885821116026, 1.9397902852694966]], [[0.159885821116026, 0.3455206537003081], [0.26647636852671375, -1.6473163857612034], [0.21318109482137512, 1.1426554694848965]], [[-0.21318109482137512, -0.8501815699765671], [-0.10659054741068756, 0.7440880615926329], [0.159885821116026, 0.7440880615926329]]], [[[0.10642282248641804, 1.5396480316389831], [0.2128456449728361, 0.742513215853795], [0.10642282248641804, -1.2503238236071168]], [[-0.10642282248641804, -0.851756415715405], [0.2128456449728361, -1.2503238236071168], [0.0, 1.141080623746683]], [[0.2128456449728361, 1.5396480316389831], [0.0, -0.05462159993231858], [-0.26605705621616116, -2.0474586393906162]], [[0.2128456449728361, -0.851756415715405], [-0.26605705621616116, -0.4531890078234187], [-0.15963423372966273, 1.5396480316389831]], [[0.2128456449728361, -0.05462159993231858], [-0.2128456449728361, 1.9382154395312832], [-0.10642282248641804, -1.6488912314994169]], [[-0.10642282248641804, 1.141080623746683], [0.15963423372966273, -1.6488912314994169], [0.0, -0.4531890078234187]], [[0.05321141124320902, 1.141080623746683], [0.05321141124320902, 0.3439458079611813], [-0.15963423372966273, -1.6488912314994169]], [[-0.26605705621616116, -0.4531890078234187], [0.15963423372966273, 0.742513215853795], [0.05321141124320902, -0.4531890078234187]], [[0.26605705621616116, 1.9382154395312832], [-0.10642282248641804, -1.2503238236071168], [0.2128456449728361, 0.742513215853795]], [[-0.15963423372966273, -2.0474586393906162], [-0.10642282248641804, 0.3439458079611813], [0.2128456449728361, 1.5396480316389831]], [[0.0, 1.141080623746683], [0.2128456449728361, -0.851756415715405], [0.0, -0.851756415715405]], [[0.05321141124320902, 1.141080623746683], [-0.10642282248641804, 1.141080623746683], [0.26605705621616116, 0.3439458079611813]]], [[[0.26259516366792207, -0.8648535451325846], [-0.26259516366792207, 1.5265509022211594], [-0.210076130934332, -2.0605557688095897]], [[-0.210076130934332, 0.7294160864366154], [-0.26259516366792207, -0.8648535451325846], [-0.26259516366792207, -1.6619883609172406]], [[0.105038065467166, -1.6619883609172406], [0.26259516366792207, -0.06771872934804836], [-0.15755709820073807, -2.0605557688095897]], [[0.26259516366792207, 1.9251183101134595], [-0.210076130934332, 0.3308486785442562], [-0.052519032733583, -0.4662861372403438]], [[0.210076130934332, -1.6619883609172406], [-0.052519032733583, -1.6619883609172406], [0.210076130934332, -0.4662861372403438]], [[-0.052519032733583, -1.2634209530249405], [-0.26259516366792207, -1.2634209530249405], [0.26259516366792207, -0.8648535451325846]], [[-0.105038065467166, 0.7294160864366154], [-0.052519032733583, 1.5265509022211594], [0.052519032733583, -0.4662861372403438]], [[-0.15755709820073807, 1.5265509022211594], [0.210076130934332, -2.0605557688095897], [0.26259516366792207, -1.6619883609172406]], [[-0.105038065467166, 1.5265509022211594], [0.15755709820073807, 1.1279834943288594], [-0.210076130934332, -0.06771872934804836]], [[-0.105038065467166, 1.1279834943288594], [0.26259516366792207, -0.8648535451325846], [0.0, -1.6619883609172406]], [[0.210076130934332, -1.2634209530249405], [0.210076130934332, 1.9251183101134595], [-0.15755709820073807, 1.1279834943288594]], [[-0.210076130934332, 0.3308486785442562], [0.052519032733583, -0.8648535451325846], [0.26259516366792207, -0.06771872934804836]]], [[[-0.21426818638162962, -0.04796034060426001], [0.10713409319081481, -0.8450951563885483], [-0.10713409319081481, -1.6422299721740061]], [[0.10713409319081481, -0.04796034060426001], [0.1607011397862351, 0.35060706728786784], [-0.10713409319081481, 1.944876698856694]], [[-0.1607011397862351, 1.5463092909643938], [0.1607011397862351, -0.04796034060426001], [0.10713409319081481, 1.944876698856694]], [[0.0, -2.040797380066068], [0.21426818638162962, 1.5463092909643938], [0.1607011397862351, 1.5463092909643938]], [[-0.053567046595407404, 0.7491744751806517], [-0.21426818638162962, 1.944876698856694], [0.1607011397862351, -0.44652774849673216]], [[-0.053567046595407404, 1.5463092909643938], [0.2678352329769596, 1.1477418830720938], [0.053567046595407404, -1.243662564281706]], [[0.21426818638162962, 0.35060706728786784], [0.10713409319081481, 1.5463092909643938], [-0.053567046595407404, 0.7491744751806517]], [[0.053567046595407404, -2.040797380066068], [-0.053567046595407404, -2.040797380066068], [0.21426818638162962, -1.243662564281706]], [[-0.1607011397862351, 0.35060706728786784], [0.10713409319081481, -0.8450951563885483], [0.10713409319081481, -2.040797380066068]], [[-0.053567046595407404, 0.35060706728786784], [-0.21426818638162962, 1.5463092909643938], [-0.10713409319081481, 1.1477418830720938]], [[-0.21426818638162962, 1.1477418830720938], [-0.21426818638162962, -0.04796034060426001], [-0.1607011397862351, 1.1477418830720938]], [[0.2678352329769596, 0.35060706728786784], [-0.21426818638162962, 1.5463092909643938], [0.2678352329769596, -1.6422299721740061]]], [[[0.26498007904154613, -1.2543799315023736], [-0.05299601580830933, 0.33988970006681796], [0.0, -1.6529473393946736]], [[-0.21198406323323732, -2.0515147472863906], [0.26498007904154613, 0.7384571079591049], [0.21198406323323732, -0.8558125236100951]], [[0.10599203161661866, -2.0515147472863906], [-0.15898804742493589, 0.7384571079591049], [0.10599203161661866, 1.9341593316360264]], [[0.26498007904154613, -1.6529473393946736], [-0.05299601580830933, -1.6529473393946736], [-0.10599203161661866, -0.45724511571778204]], [[-0.26498007904154613, -1.2543799315023736], [-0.26498007904154613, -0.45724511571778204], [0.10599203161661866, 0.7384571079591049]], [[-0.26498007904154613, -2.0515147472863906], [0.05299601580830933, -1.2543799315023736], [0.26498007904154613, -2.0515147472863906]], [[-0.21198406323323732, -0.45724511571778204], [0.10599203161661866, -1.6529473393946736], [0.15898804742493589, 0.7384571079591049]], [[-0.21198406323323732, -1.6529473393946736], [0.05299601580830933, 1.5355919237437263], [0.0, -0.05867770782540888]], [[0.10599203161661866, -0.45724511571778204], [0.05299601580830933, -1.6529473393946736], [-0.10599203161661866, -2.0515147472863906]], [[-0.05299601580830933, -0.8558125236100951], [0.10599203161661866, 0.33988970006681796], [-0.26498007904154613, -2.0515147472863906]], [[0.0, 1.5355919237437263], [0.15898804742493589, -0.45724511571778204], [-0.15898804742493589, -0.05867770782540888]], [[0.05299601580830933, -0.8558125236100951], [-0.21198406323323732, -1.6529473393946736], [0.05299601580830933, -0.05867770782540888]]]]

L_out_8_5H_3C_V = [[[0.6392080592906091, -2.7850607079919056]], [[0.05241282767567028, 0.11421883306797348]], [[-0.14126654793524795, 1.3021434392771563]], [[0.4271486188498037, -2.0247056869661852]], [[0.18539441491727832, -0.5700843482954929]], [[-0.08291017514938584, 0.8703613139853456]], [[0.11697492744435621, -0.2855138736523473]], [[0.11968627979233967, -0.1310231880694322]]]


L_8_24H_3C = [[[[-0.05620675496672916, -1.9926946019301182], [-0.2810337748327571, -1.9926946019301182], [0.2810337748327571, -1.594127194037818]], [[0.11241350993345832, 0.7972772533250094], [-0.11241350993345832, 1.1958446612082818], [-0.22482701986691664, -1.9926946019301182]], [[0.05620675496672916, -0.7969923782441906], [0.11241350993345832, 0.3987098454329517], [0.11241350993345832, 1.1958446612082818]], [[-0.11241350993345832, 1.5944120691005819], [0.0, 0.7972772533250094], [0.11241350993345832, 0.3987098454329517]], [[-0.11241350993345832, 1.992979476992882], [0.0, -1.9926946019301182], [-0.11241350993345832, 0.3987098454329517]], [[0.11241350993345832, -1.594127194037818], [0.0, -1.195559786145518], [0.11241350993345832, 0.0001424375462297389]], [[0.0, 0.0001424375462297389], [0.22482701986691664, -0.7969923782441906], [-0.2810337748327571, -0.3984249703516483]], [[0.22482701986691664, -1.594127194037818], [0.1686202649019366, -0.3984249703516483], [0.1686202649019366, -1.195559786145518]], [[-0.2810337748327571, -1.594127194037818], [0.2810337748327571, -0.3984249703516483], [-0.22482701986691664, 1.5944120691005819]], [[0.05620675496672916, -1.195559786145518], [-0.05620675496672916, -1.594127194037818], [-0.1686202649019366, 1.992979476992882]], [[0.11241350993345832, -1.9926946019301182], [-0.22482701986691664, 0.3987098454329517], [-0.22482701986691664, -0.3984249703516483]], [[0.2810337748327571, 0.3987098454329517], [0.0, 0.3987098454329517], [0.05620675496672916, 0.0001424375462297389]]], [[[-0.056199465347192494, 1.273626125475264e-05], [-0.11239893069438499, 1.5942823678041687], [0.22479786138876998, -1.1956894874419313]], [[0.2809973267358997, 1.5942823678041687], [-0.1685983960415691, -1.5942568953342313], [-0.11239893069438499, -1.9928243032265314]], [[-0.056199465347192494, 1.1957149599118686], [-0.11239893069438499, -1.5942568953342313], [0.22479786138876998, 1.1957149599118686]], [[0.2809973267358997, 0.39858014415563603], [0.056199465347192494, 1.1957149599118686], [0.11239893069438499, -0.7971220795221071]], [[0.1685983960415691, 1.5942823678041687], [0.22479786138876998, 1.273626125475264e-05], [0.2809973267358997, 0.39858014415563603]], [[-0.1685983960415691, 1.5942823678041687], [-0.2809973267358997, 0.39858014415563603], [-0.1685983960415691, 1.1957149599118686]], [[0.2809973267358997, 0.7971475520470929], [0.056199465347192494, -0.39855467162896396], [0.11239893069438499, 0.7971475520470929]], [[0.1685983960415691, -1.5942568953342313], [-0.2809973267358997, -0.39855467162896396], [-0.22479786138876998, -0.7971220795221071]], [[-0.11239893069438499, -0.39855467162896396], [0.11239893069438499, 0.7971475520470929], [-0.056199465347192494, 0.39858014415563603]], [[0.0, -0.39855467162896396], [-0.1685983960415691, 1.5942823678041687], [0.1685983960415691, 0.7971475520470929]], [[0.0, 1.9928497756964687], [-0.2809973267358997, 1.1957149599118686], [-0.1685983960415691, 1.1957149599118686]], [[0.1685983960415691, -1.1956894874419313], [-0.11239893069438499, -1.5942568953342313], [-0.22479786138876998, 0.39858014415563603]]], [[[0.112398158527994, 1.5942754978205291], [-0.112398158527994, -1.5942637653178708], [0.16859723779230204, 5.8662893458422394e-06]], [[-0.056199079263997, 1.9928429057128292], [-0.16859723779230204, 5.8662893458422394e-06], [0.112398158527994, 1.5942754978205291]], [[-0.2809953963178048, 0.3985732741799896], [0.224796317055988, 5.8662893458422394e-06], [-0.112398158527994, 0.7971406820907624]], [[0.112398158527994, -1.992831173210171], [0.224796317055988, -0.3985615416046104], [0.16859723779230204, -1.992831173210171]], [[0.224796317055988, -0.7971289494784376], [0.0, 1.195708089928229], [0.2809953963178048, 0.3985732741799896]], [[-0.056199079263997, -0.3985615416046104], [-0.16859723779230204, 1.5942754978205291], [0.112398158527994, 1.9928429057128292]], [[0.224796317055988, -1.1956963574255708], [0.224796317055988, 0.7971406820907624], [0.2809953963178048, 0.3985732741799896]], [[-0.2809953963178048, 1.195708089928229], [0.16859723779230204, 1.5942754978205291], [0.2809953963178048, 1.9928429057128292]], [[-0.2809953963178048, 1.9928429057128292], [-0.056199079263997, -1.5942637653178708], [-0.056199079263997, 0.7971406820907624]], [[0.2809953963178048, 1.5942754978205291], [-0.056199079263997, 5.8662893458422394e-06], [-0.224796317055988, -1.1956963574255708]], [[-0.16859723779230204, -1.992831173210171], [0.2809953963178048, -0.3985615416046104], [-0.224796317055988, 5.8662893458422394e-06]], [[0.056199079263997, 1.9928429057128292], [0.2809953963178048, 1.5942754978205291], [0.112398158527994, -1.5942637653178708]]], [[[-0.05616998486683021, -1.9933485661544164], [-0.22467993946732084, 0.7966232890974324], [-0.22467993946732084, -1.1962137503698163]], [[-0.16850995460052529, 0.7966232890974324], [0.05616998486683021, 1.1951906969839836], [-0.05616998486683021, 0.3980558812057634]], [[0.22467993946732084, 1.9923255127685837], [-0.16850995460052529, 0.7966232890974324], [-0.22467993946732084, 1.1951906969839836]], [[-0.05616998486683021, 1.5937581048762834], [0.22467993946732084, 0.7966232890974324], [0.11233996973366042, -0.0005115266871925787]], [[-0.22467993946732084, 0.3980558812057634], [-0.05616998486683021, -1.5947811582621165], [0.16850995460052529, 1.1951906969839836]], [[-0.05616998486683021, -1.5947811582621165], [-0.11233996973366042, 0.3980558812057634], [-0.22467993946732084, -1.1962137503698163]], [[0.05616998486683021, -1.9933485661544164], [0.16850995460052529, 1.5937581048762834], [0.22467993946732084, 1.1951906969839836]], [[-0.05616998486683021, 1.1951906969839836], [-0.28084992433411227, -1.5947811582621165], [-0.22467993946732084, -1.9933485661544164]], [[0.28084992433411227, 0.3980558812057634], [-0.28084992433411227, 1.5937581048762834], [0.0, -0.3990789345788366]], [[-0.28084992433411227, -1.9933485661544164], [0.05616998486683021, -1.9933485661544164], [0.16850995460052529, 1.9923255127685837]], [[-0.05616998486683021, -1.5947811582621165], [-0.28084992433411227, 1.9923255127685837], [-0.05616998486683021, -1.9933485661544164]], [[0.0, -0.3990789345788366], [-0.28084992433411227, -1.5947811582621165], [-0.11233996973366042, 0.7966232890974324]]], [[[-0.28101863177813385, 0.7972239638695985], [-0.1686111790669127, 0.7972239638695985], [-0.28101863177813385, -1.5941804834848914]], [[-0.22481490542257143, -1.1956130755925911], [0.11240745271128572, 1.1957913717612088], [-0.1686111790669127, 1.9929261875458089]], [[0.22481490542257143, 1.9929261875458089], [-0.28101863177813385, -1.9927478913771912], [0.28101863177813385, 1.1957913717612088]], [[-0.05620372635564286, -1.9927478913771912], [-0.11240745271128572, 0.7972239638695985], [-0.1686111790669127, 1.5943587796535086]], [[0.0, 1.9929261875458089], [0.28101863177813385, -1.1956130755925911], [0.0, 1.5943587796535086]], [[-0.22481490542257143, -0.39847825980777823], [0.1686111790669127, -0.7970456676996015], [-0.1686111790669127, -1.5941804834848914]], [[0.0, -1.1956130755925911], [0.28101863177813385, 1.1957913717612088], [0.05620372635564286, 0.39865655597682176]], [[-0.1686111790669127, 1.9929261875458089], [0.28101863177813385, -1.9927478913771912], [-0.28101863177813385, 1.5943587796535086]], [[0.05620372635564286, 1.9929261875458089], [0.28101863177813385, 0.39865655597682176], [-0.28101863177813385, -1.5941804834848914]], [[0.28101863177813385, 0.7972239638695985], [0.28101863177813385, -0.7970456676996015], [-0.1686111790669127, -0.39847825980777823]], [[0.0, -1.5941804834848914], [-0.22481490542257143, 0.39865655597682176], [0.28101863177813385, -0.39847825980777823]], [[-0.28101863177813385, 1.1957913717612088], [0.1686111790669127, -0.7970456676996015], [-0.22481490542257143, -0.7970456676996015]]], [[[-0.21044518197528855, 0.7312244747528288], [-0.21044518197528855, 0.7312244747528288], [-0.15783388648157737, 1.92692669842996]], [[0.21044518197528855, 1.5283592905376602], [0.05261129549382214, 0.3326570668609144], [-0.10522259098764428, -0.06591034103140977]], [[0.0, -1.6601799726007398], [0.15783388648157737, 0.7312244747528288], [0.0, -2.058747380491362]], [[-0.05261129549382214, -2.058747380491362], [0.21044518197528855, 1.92692669842996], [-0.21044518197528855, -1.6601799726007398]], [[-0.15783388648157737, 1.12979188264536], [0.05261129549382214, 1.92692669842996], [-0.15783388648157737, -2.058747380491362]], [[0.10522259098764428, -0.8630451568163712], [0.2630564774690765, 0.7312244747528288], [0.21044518197528855, -1.26161256470844]], [[0.2630564774690765, 0.3326570668609144], [0.05261129549382214, -0.8630451568163712], [0.10522259098764428, -1.26161256470844]], [[0.15783388648157737, -0.4644777489236856], [-0.10522259098764428, 0.7312244747528288], [-0.2630564774690765, -2.058747380491362]], [[0.2630564774690765, -2.058747380491362], [-0.15783388648157737, -1.6601799726007398], [-0.2630564774690765, 0.7312244747528288]], [[0.15783388648157737, -2.058747380491362], [-0.2630564774690765, -0.06591034103140977], [0.10522259098764428, 1.92692669842996]], [[0.05261129549382214, -0.06591034103140977], [-0.21044518197528855, 1.12979188264536], [-0.2630564774690765, -0.8630451568163712]], [[0.15783388648157737, -2.058747380491362], [-0.05261129549382214, -2.058747380491362], [0.21044518197528855, -2.058747380491362]]], [[[0.15541501876656322, -0.8785424197420629], [-0.2590250312776449, -0.47997501184995084], [0.0, -2.074244643417956]], [[0.10361001251103416, -1.2771098276340276], [-0.2590250312776449, 0.31715980393464915], [0.05180500625551708, -0.8785424197420629]], [[-0.05180500625551708, -0.8785424197420629], [-0.15541501876656322, 1.1142946197197723], [-0.05180500625551708, -0.8785424197420629]], [[0.10361001251103416, 0.31715980393464915], [0.15541501876656322, -0.08140760395776662], [-0.15541501876656322, -0.8785424197420629]], [[-0.20722002502206832, -0.08140760395776662], [-0.05180500625551708, -1.2771098276340276], [-0.15541501876656322, 1.1142946197197723]], [[-0.10361001251103416, 1.5128620276120723], [-0.20722002502206832, -0.47997501184995084], [0.2590250312776449, -0.47997501184995084]], [[-0.15541501876656322, -2.074244643417956], [0.15541501876656322, -0.08140760395776662], [0.0, 0.31715980393464915]], [[0.2590250312776449, -0.47997501184995084], [0.20722002502206832, -0.8785424197420629], [0.05180500625551708, -0.8785424197420629]], [[0.15541501876656322, 0.715727211827137], [0.20722002502206832, -0.8785424197420629], [0.20722002502206832, -0.47997501184995084]], [[0.10361001251103416, -0.8785424197420629], [0.15541501876656322, 1.1142946197197723], [-0.20722002502206832, -0.08140760395776662]], [[0.20722002502206832, 0.715727211827137], [-0.2590250312776449, 1.1142946197197723], [0.20722002502206832, -1.6756772355263276]], [[-0.20722002502206832, 0.31715980393464915], [0.15541501876656322, 1.5128620276120723], [-0.05180500625551708, -0.8785424197420629]]], [[[-0.05627410215688079, 1.1970422062869357], [-0.22509640862752317, 0.7984747984154534], [0.11254820431376159, -0.7957948331537466]], [[-0.16882230647073151, -0.7957948331537466], [-0.16882230647073151, -0.3972274252553905], [0.05627410215688079, -1.9914970568514643]], [[-0.16882230647073151, -0.7957948331537466], [-0.22509640862752317, 1.9941770220715358], [0.16882230647073151, 1.9941770220715358]], [[0.05627410215688079, 0.7984747984154534], [0.16882230647073151, 0.7984747984154534], [0.05627410215688079, 1.9941770220715358]], [[0.16882230647073151, 0.0013399826318459547], [-0.2813705107844013, 0.7984747984154534], [0.2813705107844013, 1.9941770220715358]], [[0.2813705107844013, -1.592929648959164], [-0.2813705107844013, 0.7984747984154534], [-0.05627410215688079, -1.9914970568514643]], [[0.11254820431376159, 0.3999073905292095], [-0.11254820431376159, -1.9914970568514643], [-0.05627410215688079, 1.1970422062869357]], [[0.22509640862752317, 0.7984747984154534], [0.11254820431376159, -1.592929648959164], [-0.2813705107844013, -1.9914970568514643]], [[0.22509640862752317, 1.595609614179236], [0.16882230647073151, -0.7957948331537466], [0.2813705107844013, -1.1943622410668642]], [[-0.16882230647073151, 1.1970422062869357], [0.16882230647073151, -1.592929648959164], [0.0, -0.3972274252553905]], [[-0.05627410215688079, 0.3999073905292095], [-0.2813705107844013, -1.592929648959164], [0.16882230647073151, -1.9914970568514643]], [[0.16882230647073151, 1.9941770220715358], [0.16882230647073151, -1.9914970568514643], [-0.11254820431376159, -0.3972274252553905]]]]

L_out_8_24H_3C = [[[147.88691653979646, 147.3753789818573]], [[1.6373470559224907, -0.37270577536500377]], [[10.29267655313166, 10.331989226563376]], [[0.9570457418147688, -0.2357223208677493]], [[0.6505505976089713, 1.5251291965480007]], [[0.10129388077793676, 0.3159734915032931]], [[0.07992651930106566, -0.26745699565834596]], [[1345.822775559229, 1343.2843658285312]]]

L_8_24H_3C_V = [[[[-0.05319266623231917, -2.0478109640132804], [-0.26596333066882744, -2.0478109640132804], [0.26596333066882744, -1.6492435482309264]], [[0.10638533246463834, 0.7421608991254831], [-0.10638533246463834, 1.1407283070151735], [-0.21277066492927668, -2.0478109640132804]], [[0.05319266623231917, -0.8521087324437169], [0.10638533246463834, 0.34359349123577776], [0.10638533246463834, 1.1407283070151735]], [[-0.10638533246463834, 1.5392957149074735], [0.0, 0.7421608991254831], [0.10638533246463834, 0.34359349123577776]], [[-0.10638533246463834, 1.9378631227997736], [0.0, -2.0478109640132804], [-0.10638533246463834, 0.34359349123577776]], [[0.10638533246463834, -1.6492435482309264], [0.0, -1.2506761403386264], [0.10638533246463834, -0.0549739169549548]], [[0.0, -0.0549739169549548], [0.21277066492927668, -0.8521087324437169], [-0.26596333066882744, -0.45354132454882223]], [[0.21277066492927668, -1.6492435482309264], [0.15957799820288052, -0.45354132454882223], [0.15957799820288052, -1.2506761403386264]], [[-0.26596333066882744, -1.6492435482309264], [0.26596333066882744, -0.45354132454882223], [-0.21277066492927668, 1.5392957149074735]], [[0.05319266623231917, -1.2506761403386264], [-0.05319266623231917, -1.6492435482309264], [-0.15957799820288052, 1.9378631227997736]], [[0.10638533246463834, -2.0478109640132804], [-0.21277066492927668, 0.34359349123577776], [-0.21277066492927668, -0.45354132454882223]], [[0.26596333066882744, 0.34359349123577776], [0.0, 0.34359349123577776], [0.05319266623231917, -0.0549739169549548]]], [[[-0.05348815616648456, -0.049434205770129164], [-0.10697631233296911, 1.5448354257652674], [0.21395262466593823, -1.2451364294808325]], [[0.26744078083146205, 1.5448354257652674], [-0.16046446849933296, -1.6437038373731325], [-0.10697631233296911, -2.042271245232566]], [[-0.05348815616648456, 1.1462680178729674], [-0.10697631233296911, -1.6437038373731325], [0.21395262466593823, 1.1462680178729674]], [[0.26744078083146205, 0.3491332021296925], [0.05348815616648456, 1.1462680178729674], [0.10697631233296911, -0.8465690215648233]], [[0.16046446849933296, 1.5448354257652674], [0.21395262466593823, -0.049434205770129164], [0.26744078083146205, 0.3491332021296925]], [[-0.16046446849933296, 1.5448354257652674], [-0.26744078083146205, 0.3491332021296925], [-0.16046446849933296, 1.1462680178729674]], [[0.26744078083146205, 0.7477006100043767], [0.05348815616648456, -0.4480016136549075], [0.10697631233296911, 0.7477006100043767]], [[0.16046446849933296, -1.6437038373731325], [-0.26744078083146205, -0.4480016136549075], [-0.21395262466593823, -0.8465690215648233]], [[-0.10697631233296911, -0.4480016136549075], [0.10697631233296911, 0.7477006100043767], [-0.05348815616648456, 0.3491332021296925]], [[0.0, -0.4480016136549075], [-0.16046446849933296, 1.5448354257652674], [0.16046446849933296, 0.7477006100043767]], [[0.0, 1.9434028336575675], [-0.26744078083146205, 1.1462680178729674], [-0.16046446849933296, 1.1462680178729674]], [[0.16046446849933296, -1.2451364294808325], [-0.10697631233296911, -1.6437038373731325], [-0.21395262466593823, 0.3491332021296925]]], [[[0.1073922302998391, 1.5487158275317292], [-0.1073922302998391, -1.6398234356066708], [0.16108834548903722, -0.04555380379375862]], [[-0.05369611514991955, 1.9472832354240293], [-0.16108834548903722, -0.04555380379375862], [0.1073922302998391, 1.5487158275317292]], [[-0.2684805757683688, 0.35301360407944704], [0.2147844605996782, -0.04555380379375862], [-0.1073922302998391, 0.7515810119819322]], [[0.1073922302998391, -2.038390844478039], [0.2147844605996782, -0.44412121170515295], [0.16108834548903722, -2.038390844478039]], [[0.2147844605996782, -0.8426886195872678], [0.0, 1.1501484196394292], [0.2684805757683688, 0.35301360407944704]], [[-0.05369611514991955, -0.44412121170515295], [-0.16108834548903722, 1.5487158275317292], [0.1073922302998391, 1.9472832354240293]], [[0.2147844605996782, -1.2412560277143707], [0.2147844605996782, 0.7515810119819322], [0.2684805757683688, 0.35301360407944704]], [[-0.2684805757683688, 1.1501484196394292], [0.16108834548903722, 1.5487158275317292], [0.2684805757683688, 1.9472832354240293]], [[-0.2684805757683688, 1.9472832354240293], [-0.05369611514991955, -1.6398234356066708], [-0.05369611514991955, 0.7515810119819322]], [[0.2684805757683688, 1.5487158275317292], [-0.05369611514991955, -0.04555380379375862], [-0.2147844605996782, -1.2412560277143707]], [[-0.16108834548903722, -2.038390844478039], [0.2684805757683688, -0.44412121170515295], [-0.2147844605996782, -0.04555380379375862]], [[0.05369611514991955, 1.9472832354240293], [0.2684805757683688, 1.5487158275317292], [0.1073922302998391, -1.6398234356066708]]], [[[-0.05303490580137364, -2.0507810409436855], [-0.21213962320549457, 0.7391908142901137], [-0.21213962320549457, -1.2536462251744174]], [[-0.15910471740656928, 0.7391908142901137], [0.05303490580137364, 1.1377582221793825], [-0.05303490580137364, 0.3406234063988748]], [[0.21213962320549457, 1.9348930379639826], [-0.15910471740656928, 0.7391908142901137], [-0.21213962320549457, 1.1377582221793825]], [[-0.05303490580137364, 1.5363256300716823], [0.21213962320549457, 0.7391908142901137], [0.10606981160274728, -0.05794400149458976]], [[-0.21213962320549457, 0.3406234063988748], [-0.05303490580137364, -1.6522136330667176], [0.15910471740656928, 1.1377582221793825]], [[-0.05303490580137364, -1.6522136330667176], [-0.10606981160274728, 0.3406234063988748], [-0.21213962320549457, -1.2536462251744174]], [[0.05303490580137364, -2.0507810409436855], [0.15910471740656928, 1.5363256300716823], [0.21213962320549457, 1.1377582221793825]], [[-0.05303490580137364, 1.1377582221793825], [-0.26517452900062594, -1.6522136330667176], [-0.21213962320549457, -2.0507810409436855]], [[0.26517452900062594, 0.3406234063988748], [-0.26517452900062594, 1.5363256300716823], [0.0, -0.4565114093857252]], [[-0.26517452900062594, -2.0507810409436855], [0.05303490580137364, -2.0507810409436855], [0.15910471740656928, 1.9348930379639826]], [[-0.05303490580137364, -1.6522136330667176], [-0.26517452900062594, 1.9348930379639826], [-0.05303490580137364, -2.0507810409436855]], [[0.0, -0.4565114093857252], [-0.26517452900062594, -1.6522136330667176], [-0.10606981160274728, 0.7391908142901137]]], [[[-0.2658651229807066, 0.7417917768783365], [-0.159519073788456, 0.7417917768783365], [-0.2658651229807066, -1.6496126704753988]], [[-0.21269209838461478, -1.2510452625830988], [0.10634604919230739, 1.140359184770701], [-0.159519073788456, 1.9374940005553012]], [[0.21269209838461478, 1.9374940005553012], [-0.2658651229807066, -2.0481800783671584], [0.2658651229807066, 1.140359184770701]], [[-0.053173024596153695, -2.0481800783671584], [-0.10634604919230739, 0.7417917768783365], [-0.159519073788456, 1.5389265926630011]], [[0.0, 1.9374940005553012], [0.2658651229807066, -1.2510452625830988], [0.0, 1.5389265926630011]], [[-0.21269209838461478, -0.45391044679905623], [0.159519073788456, -0.8524778546908635], [-0.159519073788456, -1.6496126704753988]], [[0.0, -1.2510452625830988], [0.2658651229807066, 1.140359184770701], [0.053173024596153695, 0.34322436898554376]], [[-0.159519073788456, 1.9374940005553012], [0.2658651229807066, -2.0481800783671584], [-0.2658651229807066, 1.5389265926630011]], [[0.053173024596153695, 1.9374940005553012], [0.2658651229807066, 0.34322436898554376], [-0.2658651229807066, -1.6496126704753988]], [[0.2658651229807066, 0.7417917768783365], [0.2658651229807066, -0.8524778546908635], [-0.159519073788456, -0.45391044679905623]], [[0.0, -1.6496126704753988], [-0.21269209838461478, 0.34322436898554376], [0.2658651229807066, -0.45391044679905623]], [[-0.2658651229807066, 1.140359184770701], [0.159519073788456, -0.8524778546908635], [-0.21269209838461478, -0.8524778546908635]]], [[[-0.20813404413440917, 0.7201460566024707], [-0.20813404413440917, 0.7201460566024707], [-0.15610053309927438, 1.9158482802758552]], [[0.20813404413440917, 1.5172808723835554], [0.05203351103360229, 0.3215786487112566], [-0.10406702206720458, -0.07698875918155315]], [[0.0, -1.6712583907548446], [0.15610053309927438, 0.7201460566024707], [0.0, -2.069825798631215]], [[-0.05203351103360229, -2.069825798631215], [0.20813404413440917, 1.9158482802758552], [-0.20813404413440917, -1.6712583907548446]], [[-0.15610053309927438, 1.118713464491255], [0.05203351103360229, 1.9158482802758552], [-0.15610053309927438, -2.069825798631215]], [[0.10406702206720458, -0.8741235749667293], [0.2601675551660786, 0.7201460566024707], [0.20813404413440917, -1.2726909828625448]], [[0.2601675551660786, 0.3215786487112566], [0.05203351103360229, -0.8741235749667293], [0.10406702206720458, -1.2726909828625448]], [[0.15610053309927438, -0.4755561670733434], [-0.10406702206720458, 0.7201460566024707], [-0.2601675551660786, -2.069825798631215]], [[0.2601675551660786, -2.069825798631215], [-0.15610053309927438, -1.6712583907548446], [-0.2601675551660786, 0.7201460566024707]], [[0.15610053309927438, -2.069825798631215], [-0.2601675551660786, -0.07698875918155315], [0.10406702206720458, 1.9158482802758552]], [[0.05203351103360229, -0.07698875918155315], [-0.20813404413440917, 1.118713464491255], [-0.2601675551660786, -0.8741235749667293]], [[0.15610053309927438, -2.069825798631215], [-0.05203351103360229, -2.069825798631215], [0.20813404413440917, -2.069825798631215]]], [[[0.1580979926400785, -0.8614264429722185], [-0.26349665440013986, -0.4628590350799528], [0.0, -2.057128666649092]], [[0.10539866176005652, -1.259993850864529], [-0.26349665440013986, 0.3342757807046472], [0.05269933088002826, -0.8614264429722185]], [[-0.05269933088002826, -0.8614264429722185], [-0.1580979926400785, 1.131410596489271], [-0.05269933088002826, -0.8614264429722185]], [[0.10539866176005652, 0.3342757807046472], [0.1580979926400785, -0.06429162718762757], [-0.1580979926400785, -0.8614264429722185]], [[-0.21079732352011304, -0.06429162718762757], [-0.05269933088002826, -1.259993850864529], [-0.1580979926400785, 1.131410596489271]], [[-0.10539866176005652, 1.529978004381571], [-0.21079732352011304, -0.4628590350799528], [0.26349665440013986, -0.4628590350799528]], [[-0.1580979926400785, -2.057128666649092], [0.1580979926400785, -0.06429162718762757], [0.0, 0.3342757807046472]], [[0.26349665440013986, -0.4628590350799528], [0.21079732352011304, -0.8614264429722185], [0.05269933088002826, -0.8614264429722185]], [[0.1580979926400785, 0.7328431885969815], [0.21079732352011304, -0.8614264429722185], [0.21079732352011304, -0.4628590350799528]], [[0.10539866176005652, -0.8614264429722185], [0.1580979926400785, 1.131410596489271], [-0.21079732352011304, -0.06429162718762757]], [[0.21079732352011304, 0.7328431885969815], [-0.26349665440013986, 1.131410596489271], [0.21079732352011304, -1.658561258756829]], [[-0.21079732352011304, 0.3342757807046472], [0.1580979926400785, 1.529978004381571], [-0.05269933088002826, -0.8614264429722185]]], [[[-0.053182424286900495, 1.140535765393052], [-0.21272969714760198, 0.7419683573360273], [0.10636484857380099, -0.8523012742331727]], [[-0.1595472728752513, -0.8523012742331727], [-0.1595472728752513, -0.4537338663231827], [0.053182424286900495, -2.048003498252584]], [[-0.1595472728752513, -0.8523012742331727], [-0.21272969714760198, 1.937670581177652], [0.1595472728752513, 1.937670581177652]], [[0.053182424286900495, 0.7419683573360273], [0.1595472728752513, 0.7419683573360273], [0.053182424286900495, 1.937670581177652]], [[0.1595472728752513, -0.055166458676791146], [-0.265912121480033, 0.7419683573360273], [0.265912121480033, 1.937670581177652]], [[0.265912121480033, -1.649436089853048], [-0.265912121480033, 0.7419683573360273], [-0.053182424286900495, -2.048003498252584]], [[0.10636484857380099, 0.3434009494614173], [-0.10636484857380099, -2.048003498252584], [-0.053182424286900495, 1.140535765393052]], [[0.21272969714760198, 0.7419683573360273], [0.10636484857380099, -1.649436089853048], [-0.265912121480033, -2.048003498252584]], [[0.21272969714760198, 1.539103173285352], [0.1595472728752513, -0.8523012742331727], [0.265912121480033, -1.250868681960748]], [[-0.1595472728752513, 1.140535765393052], [0.1595472728752513, -1.649436089853048], [0.0, -0.4537338663231827]], [[-0.053182424286900495, 0.3434009494614173], [-0.265912121480033, -1.649436089853048], [0.1595472728752513, -2.048003498252584]], [[0.1595472728752513, 1.937670581177652], [0.1595472728752513, -2.048003498252584], [-0.10636484857380099, -0.4537338663231827]]]]

L_out_8_24H_3C_V = [[[0.15049773877730038, -0.3610398191390057]], [[0.33735401047805796, -1.6726988208105948]], [[0.05623363233012343, 0.0955463057619437]], [[0.2638920405988692, -0.9288760220821802]], [[-0.05555377994987121, 0.8190248189892624]], [[0.060123367499395994, 0.274802978224768]], [[0.15015482257292523, -0.19722869238648782]], [[0.460447782961795, -2.077961945932593]]]


L_8_1H_3C_AV = copy.deepcopy(L_8_1H_3C)
L_out_8_1H_3C_AV = copy.deepcopy(L_out_8_1H_3C)

L_8_3H_3C_AV = copy.deepcopy(L_8_3H_3C)
L_out_8_3H_3C_AV = copy.deepcopy(L_out_8_3H_3C)

L_8_1H_3C_V_AV = copy.deepcopy(L_8_1H_3C_V)
L_out_8_1H_3C_V_AV = copy.deepcopy(L_out_8_1H_3C_V)

L_8_3H_3C_V_AV = copy.deepcopy(L_8_1H_3C_V)
L_out_8_3H_3C_V_AV = copy.deepcopy(L_out_8_1H_3C_V)

L_8_5H_3C_AV = copy.deepcopy(L_8_5H_3C)
L_out_8_5H_3C_AV = copy.deepcopy(L_out_8_5H_3C)

L_8_5H_3C_V_AV = copy.deepcopy(L_8_5H_3C_V)
L_out_8_5H_3C_V_AV = copy.deepcopy(L_out_8_5H_3C_V)

L_8_24H_3C_AV = copy.deepcopy(L_8_24H_3C)
L_out_8_24H_3C_AV = copy.deepcopy(L_out_8_24H_3C)

L_8_24H_3C_V_AV = copy.deepcopy(L_8_24H_3C_V)
L_out_8_24H_3C_V_AV = copy.deepcopy(L_out_8_24H_3C_V)

epochs = 100
learning_rate = 0.00001
L_donnees_1H = jeu_donnees_jeu_loaded
L_donnees_3H = jeu_donnees_jeu_loaded_3H
L_donnees_5H = jeu_donnees_jeu_loaded_5H
L_donnees_24H = jeu_donnees_jeu_loaded_24H
L_donnees_1H_V = prep_V(jeu_donnees_jeu_loaded)
L_donnees_3H_V = prep_V(jeu_donnees_jeu_loaded_3H)
L_donnees_5H_V = prep_V(jeu_donnees_jeu_loaded_5H)
L_donnees_24H_V = prep_V(jeu_donnees_jeu_loaded_24H)

time_beg = datetime.utcnow()
print("Time beg = ", time_beg)


print()         #Pour l'apprentissage 
    #Pour 1H:
#aprend_8(L_8_1H_3C,L_out_8_1H_3C,jeu_donnees_teste_loaded,L_donnees_1H,epochs,learning_rate)
    #Pour 3H
#aprend_8(L_8_3H_3C,L_out_8_3H_3C,jeu_donnees_teste_loaded_3H,L_donnees_3H,epochs,learning_rate)
    #Pour 1H, variation
#aprend_8(L_8_1H_3C_V,L_out_8_1H_3C_V,jeu_teste_V_1H,L_donnees_1H_V,epochs,learning_rate)
    #Pour 3H, variation
#aprend_8(L_8_3H_3C_V,L_out_8_3H_3C_V,jeu_teste_V_3H,L_donnees_3H_V,epochs,learning_rate)
    #Pour 5H
#aprend_8(L_8_5H_3C,L_out_8_5H_3C,jeu_donnees_teste_loaded_5H,L_donnees_5H,epochs,learning_rate)
    #Pour 5H, variation
#aprend_8(L_8_5H_3C_V,L_out_8_5H_3C_V,jeu_teste_V_5H,L_donnees_5H_V,epochs,learning_rate)
    #Pour 24H
#aprend_8(L_8_24H_3C,L_out_8_24H_3C,jeu_donnees_teste_loaded_24H,L_donnees_24H,epochs,learning_rate)
    #Pour 24H, variation
#aprend_8(L_8_24H_3C_V,L_out_8_24H_3C_V,jeu_teste_V_24H,L_donnees_24H_V,epochs,learning_rate)

time_finish = datetime.utcnow()
time = time_finish-time_beg

print()         #Pour la durée de l'apprentissage 
    #Pour 1H et 1H,variation
#print("Pour",epochs,"boucles et",len(L_donnees_1H),"exemples, il faut",time)
    #Pour 3H et 3H,variation
#print("Pour",epochs,"boucles et",len(L_donnees_3H),"exemples, il faut",time)
    #Pour 5H et 5H, variation
#print("Pour",epochs,"boucles et",len(L_donnees_5H),"exemples, il faut",time)
    #Pour 24H et 24H, variation
#print("Pour",epochs,"boucles et",len(L_donnees_24H),"exemples, il faut",time)
print()
                #Pourcentage changement
    #Pour 1H:
#pourcentage_changement(L_8_1H_3C, L_out_8_1H_3C, L_8_1H_3C_AV, L_out_8_1H_3C_AV )
    #Pour 3H:
#pourcentage_changement(L_8_3H_3C, L_out_8_3H_3C, L_8_3H_3C_AV, L_out_8_3H_3C_AV )
    #Pour 1H, varaiation
#pourcentage_changement(L_8_1H_3C_V, L_out_8_1H_3C_V, L_8_1H_3C_V_AV, L_out_8_1H_3C_V_AV )
    #Pour 3H, varaiation
#pourcentage_changement(L_8_3H_3C_V, L_out_8_3H_3C_V, L_8_3H_3C_V_AV, L_out_8_3H_3C_V_AV )
    #Pour 5H
#pourcentage_changement(L_8_5H_3C, L_out_8_5H_3C, L_8_5H_3C_AV, L_out_8_5H_3C_AV )
    #Pour 5H, variation
#pourcentage_changement(L_8_5H_3C_V, L_out_8_5H_3C_V, L_8_5H_3C_V_AV, L_out_8_5H_3C_V_AV )
    #Pour 24H
#pourcentage_changement(L_8_24H_3C, L_out_8_24H_3C, L_8_24H_3C_AV, L_out_8_24H_3C_AV )
    #Pour 24H, variation
#pourcentage_changement(L_8_24H_3C_V, L_out_8_24H_3C_V, L_8_24H_3C_V_AV, L_out_8_24H_3C_V_AV )

                #Testes
    #Pour 1H:
#teste_8(L_8_1H_3C,L_out_8_1H_3C,jeu_donnees_teste_loaded)
    #Pour 3H:
#teste_8(L_8_3H_3C,L_out_8_3H_3C,jeu_donnees_teste_loaded_3H)
    #Pour 1H,variation:
#teste_8(L_8_1H_3C_V,L_out_8_1H_3C_V,jeu_teste_V_1H)
    #Pour 3H,variation:
#teste_8(L_8_3H_3C_V,L_out_8_3H_3C_V,jeu_teste_V_3H)
    #Pour 5H
#teste_8(L_8_5H_3C,L_out_8_5H_3C,jeu_donnees_teste_loaded_5H)
    #Pour 5H,variation
#teste_8(L_8_5H_3C_V,L_out_8_5H_3C_V,jeu_teste_V_5H)
    #Pour 24H
#teste_8(L_8_24H_3C,L_out_8_24H_3C,jeu_donnees_teste_loaded_24H)
    #Pour 5H,variation
#teste_8(L_8_24H_3C_V,L_out_8_24H_3C_V,jeu_teste_V_24H)
        
                #Verif_prediction
    #Pour 1H:  
#verif_prediction_8(L_8_1H_3C,L_out_8_1H_3C,api_key,city_name,1)
    #Pour 3H:
#verif_prediction_8(L_8_3H_3C,L_out_8_3H_3C,api_key,city_name,3)
    #Pour 1H,variation:
#verif_prediction_8_V(L_8_1H_3C_V,L_out_8_1H_3C_V,api_key,city_name,1)
    #Pour 3H,variation:
#verif_prediction_8_V(L_8_3H_3C_V,L_out_8_3H_3C_V,api_key,city_name,3)
    #Pour 5H:
#verif_prediction_8(L_8_5H_3C,L_out_8_5H_3C,api_key,city_name,5)
    #Pour 5H, variation
#verif_prediction_8_V(L_8_5H_3C_V,L_out_8_5H_3C_V,api_key,city_name,5)
    #Pour 24H:
#verif_prediction_8(L_8_24H_3C,L_out_8_24H_3C,api_key,city_name,24)
    #Pour 24H, variation
#verif_prediction_8_V(L_8_24H_3C_V,L_out_8_24H_3C_V,api_key,city_name,24)



    #Pour l'exel:  
#verif_prediction_8(L_8_1H_3C,L_out_8_1H_3C,api_key,city_name,20*24+5)




                #Mesures sur les temps pour les différents réseaux
#Pour 1H 5C:
# Pour 10 000 boucles et 2200 exemples, il faut: 4:40:58.363606 
# Pour 1000 boucles et 500 exemples, il faut 0:06:25.402592
# Pour 100 boucles et 4220 exemples, il faut 0:05:16.929002
# Pour 100 boucles et 5260 exemples, il faut 0:06:42.429948

#Pour 1H 3C:
# Pour 100 boucles et 5260 exemples, il faut 0:04:11.626115
# Pour 100 boucles et 5260 exemples, il faut 0:04:09.579353

#Pour 1H 2C:
# Pour 100 boucles et 5660 exemples, il faut 0:03:08.268818
    
#Pour 8AI 1H 3C:
# Pour 100 boucles et 6060 exemples, il faut 0:07:33.063433
# Pour 100 boucles et 6460 exemples, il faut 0:08:08.401933
# Pour 100 boucles et 6860 exemples, il faut 0:08:33.70449# 
# Pour 3000 boucles et 6860 exemples, il faut 4:16:16.995727
# Pour 100 boucles et 7260 exemples, il faut 0:09:12.346098
# Pour 100 boucles et 7660 exemples, il faut 0:09:46.723137
# Pour 100 boucles et 11175 exemples, il faut 0:14:01.379346
# Pour 10 boucles et 17108 exemples, il faut 0:02:10.904377

#Pour 8AI 3H 3C:
# Pour 100 boucles et 301 exemples, il faut 0:00:22.301961
# Pour 100 boucles et 702 exemples, il faut 0:00:53.075615
# Pour 1000 boucles et 702 exemples, il faut 0:08:49.463414
# Pour 100 boucles et 1103 exemples, il faut 0:01:23.469814
# Pour 100 boucles et 20708 exemples, il faut 0:26:08.176869

#Pour 8AI 1H 3C variation
# Pour 1000 boucles et 11175 exemples, il faut 2:22:44.144414
    
#Pour 8AI 3H 3C variation

#Pour 8AI 5H:
    
#Pour 8AI 5H, variation: