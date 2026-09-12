from math import exp
from random import randint,shuffle
import csv
import pyowm
from datetime import datetime, timedelta
import csv

def init_params(nb_couche, nb_neurones_couches, nb_neurones_out):
    L = [[[randint(-5, 5)*0.061, randint(-5, 5)*0.81] for _ in range(nb_couche)] for _ in range(nb_neurones_couches)]
    L_out = [[randint(0, 50)*0.41, randint(0, 50)*0.301] for _ in range(nb_neurones_out)]
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
        if comparer_a_pres(result[i], attendu[i], 1):
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
    gradient_poids_cachees = [[0 for _ in range(len(L[0]))] for _ in range(len(L))]
    gradient_biais_cachees = [[0 for _ in range(len(L[0]))] for _ in range(len(L))]
    for neurone_cache in range(len(L[0])):
        for neurone_entree in range(len(L)):
            somme_gradient_poids = 0
            somme_gradient_biais = 0
            for neurone_sortie in range(len(L_out)):
                delta = 2 * (L_result[neurone_sortie] - L_verif[neurone_sortie]) * sigmoid_deriv(L_result[neurone_sortie])
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
    weather_data = get_weather_data_teste(api_key, city_name,delta)
    
    donnees_meteo = weather_data_to_jeu_donnees_pour_1_jeu(weather_data)
    
    expected = donnees_meteo[0][1]
    
    teste(L,L_out,donnees_meteo)
    
    prediction =  predict(L, L_out, donnees_meteo[0][0])
    L_ou = prediction[0]
    for i in range(len(L_ou)):
        print("Prediction : ",L_ou[i],"/Expected : ",expected[i])


def apprentissage_8(L, L_out, learning_rate, epochs, jeu_donnees, jeu_teste):
    print("Apprentissage en cour...")
    for boucle in range(1,epochs+1):
        if boucle % 10 == 0:
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

            accu = accuracy(predi_out, L_verif)
            total_accuracy += accu

    avg_loss = total_loss / len(jeu_teste)
    avg_accuracy = total_accuracy / len(jeu_teste)

    print("Average loss =", avg_loss)
    print("Average accuracy =", avg_accuracy)

def aprend_8(L, L_out, jeu_donnees_teste_loaded, jeu_donnees_jeu_loaded, epochs, learning_rate):
    print("L Avant")
    print()
    print(L)
    print()
    print("L_out Avant")
    print()
    print(L_out)
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
    weather_data = get_weather_data_teste(api_key, city_name,delta)
    
    donnees_meteo = weather_data_to_jeu_donnees_pour_1_jeu(weather_data)
    
    expected = donnees_meteo[0][1]
    
    teste_8(L,L_out,donnees_meteo)
    L_ou = []
    for AI in range(len(L)):
        prediction =  predict(L[AI], L_out[AI], donnees_meteo[0][0])
        L_ou.append(prediction[0][0])
    
    for i in range(len(L_ou)):
        print("Prediction : ",L_ou[i],"/Expected : ",expected[i])


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

def get_weather_data_teste(api_key, city_name,delta):
    # Initialisation de la liste pour stocker les données
    weather_data = []

    # Initialisation de l'objet OWM
    owm = pyowm.OWM(api_key)
    mgr = owm.weather_manager()

    # Récupération de la date et heure actuelle
    end_date = datetime(2024, 4, 21)
    # Définir la date de début 
    start_date = end_date - timedelta(hours = delta)

    # Boucle à travers chaque heure dans la plage spécifiée
    current_date = start_date
    while current_date <= end_date:
        # Récupérer les données météorologiques pour la ville et la date actuelle
        observation = mgr.weather_at_place(city_name)
        weather = observation.weather

        # Extraire les données météorologiques nécessaires pour l'heure actuelle
        current_weather_entry = {
            "pressure": weather.pressure['press'],
            "temp": weather.temperature('celsius')['temp'],
            "humidity": weather.humidity,
            "wind_direction": weather.wind()['deg'],
            "wind_speed": weather.wind()['speed'],
            "clouds": weather.clouds,
            "rain": weather.rain.get('1h', 0),
            "visibility": weather.visibility_distance,
            "hour": current_date.hour,
            "minute": 0,
            "day": current_date.day,
            "month": current_date.month
        }
        
        # Récupérer les prévisions météorologiques horaires pour une heure suivante
        forecast = mgr.forecast_at_place(city_name, '3h')
        # Get the forecast for the next 3 hours
        forecast = forecast.forecast.weathers
        next_hour_date = current_date + timedelta(hours=1)
        next_hour_weather = None
        for weather in forecast:
            if weather.reference_time('date').hour == next_hour_date.hour:
                next_hour_weather = {
                    "pressure": weather.pressure['press'],
                    "temp": weather.temperature('celsius')['temp'],
                    "humidity": weather.humidity,
                    "wind_direction": weather.wind()['deg'],
                    "wind_speed": weather.wind()['speed'],
                    "clouds": weather.clouds,
                    "rain": weather.rain.get('1h', 0),
                    "visibility": weather.visibility_distance,
                    #"hour": next_hour_date.hour,
                    #"minute": next_hour_date.minute,
                    #"day": next_hour_date.day,
                    #"month": next_hour_date.month
                }
                # Ajouter l'ensemble de données actuel et les prévisions pour l'heure suivante à la liste
                weather_data = [current_weather_entry, next_hour_weather]
                break

        # Avancer d'une heure
        current_date += timedelta(hours=1)

    return weather_data

def weather_data_to_jeu_donnees_pour_1_jeu(weather_data):
    jeu_donnees = []
    L_jeu = [weather_data[0][cle] for cle in weather_data[0]]
    L_teste = [weather_data[1][cle] for cle in weather_data[1]]
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

shuffle(jeu_donnees_jeu_loaded)

#Initialisation jeu de données
for jeu in jeu_donnees_jeu_loaded:
    jeu[0][9] == 0
    
shuffle(jeu_donnees_jeu_loaded)

for i in range(100):
    jeu_donnees_teste_loaded.append(jeu_donnees_jeu_loaded[0])
    jeu_donnees_jeu_loaded.pop(0)
    
city_name = 'Cherbourg'
api_key = '141a3167c551e41755c9588eeb66ce17'



#On créer 8 IA avec 2 couches et 1 output et chacune d'entre elle va donner une données.
L_1H_3C, L_out_1H_3C = init_params(2, 12, 1)

L_8_1H_3C = []
L_out_8_1H_3C = []
for i in range(8):
    L_1H_3C, L_out_1H_3C = init_params(2, 12, 1)
    
    L_8_1H_3C.append(L_1H_3C)
    
    L_out_8_1H_3C.append(L_out_1H_3C)
    
L_8_1H_3C = [[[[0.0, 0.81], [0.061, 2.43]], [[0.244, -2.43], [0.122, 0.0]], [[0.305, -0.81], [0.244, 1.62]], [[-0.183, 0.0], [0.0, 0.0]], [[-0.244, 0.0], [-0.122, -1.62]], [[0.061, -0.81], [-0.305, -0.81]], [[-0.122, 4.050000000000001], [0.244, 3.24]], [[-0.061, 4.050000000000001], [-0.122, 0.0]], [[0.0, 0.81], [0.305, 0.0]], [[-0.244, 4.050000000000001], [-0.244, -4.050000000000001]], [[-0.183, 0.0], [0.183, -4.050000000000001]], [[0.0, 1.62], [0.061, -4.050000000000001]]], [[[0.3049882153769847, 1.6199613615198365], [-0.1219952861507688, -1.6200386384801637]], [[-0.18299292922621002, -1.6200386384801637], [-0.0609976430753844, -2.43003863849249]], [[-0.2439905723015376, -1.6200386384801637], [0.2439905723015376, -3.24003863849249]], [[-0.18299292922621002, -3.24003863849249], [0.3049882153769847, -0.8100386384860226]], [[-0.0609976430753844, 0.8099613615139775], [0.3049882153769847, 0.8099613615139775]], [[0.18299292922621002, 1.6199613615198365], [-0.2439905723015376, -1.6200386384801637]], [[0.3049882153769847, -3.863848498617202e-05], [0.18299292922621002, 4.049961361506676]], [[0.18299292922621002, -3.863848498617202e-05], [-0.0609976430753844, 0.8099613615139775]], [[0.3049882153769847, 1.6199613615198365], [0.0609976430753844, -2.43003863849249]], [[0.0, -3.24003863849249], [0.2439905723015376, 3.2399613615075102]], [[0.0, 0.8099613615139775], [-0.3049882153769847, -0.8100386384860226]], [[0.0609976430753844, 3.2399613615075102], [0.2439905723015376, 3.2399613615075102]]], [[[0.122, -0.81], [-0.061, 3.24]], [[0.183, -1.62], [0.122, -1.62]], [[-0.244, 0.0], [-0.244, -0.81]], [[-0.244, 2.43], [0.061, 2.43]], [[0.0, -1.62], [-0.061, 3.24]], [[-0.061, 0.0], [0.0, -0.81]], [[-0.305, 0.0], [-0.183, -2.43]], [[-0.183, 2.43], [-0.244, -2.43]], [[-0.305, 1.62], [-0.061, -4.050000000000001]], [[-0.061, 0.81], [0.244, 3.24]], [[0.183, -2.43], [0.061, -0.81]], [[-0.305, 3.24], [0.061, -3.24]]], [[[-0.06100009695449645, 0.8100015894167387], [0.18300029086348885, -2.4299984105832695]], [[0.1220001939089929, 1.620001589416736], [-0.06100009695449645, 1.620001589416736]], [[0.0, -3.2399984105832695], [-0.1220001939089929, 4.050001589416739]], [[-0.18300029086348885, 1.5894167379524418e-06], [0.18300029086348885, -4.049998410583262]], [[-0.06100009695449645, -1.619998410583264], [-0.2440003878179858, 2.430001589416731]], [[-0.18300029086348885, 1.620001589416736], [0.1220001939089929, 1.5894167379524418e-06]], [[0.0, 1.5894167379524418e-06], [-0.3050004847724821, -2.4299984105832695]], [[-0.3050004847724821, -2.4299984105832695], [0.1220001939089929, 1.620001589416736]], [[-0.1220001939089929, -0.8099984105832614], [0.0, 2.430001589416731]], [[0.3050004847724821, -0.8099984105832614], [-0.06100009695449645, -2.4299984105832695]], [[-0.3050004847724821, 2.430001589416731], [0.1220001939089929, -2.4299984105832695]], [[0.1220001939089929, 3.240001589416731], [0.2440003878179858, 2.430001589416731]]], [[[0.06092659119010746, -2.4312036142263467], [0.0, -2.4312036142263467]], [[0.24370636476042984, -3.2412036142263467], [-0.12185318238021492, -0.8112036142263446]], [[0.18277977357030673, -0.8112036142263446], [0.3046329559505224, 3.2387963857736537]], [[0.3046329559505224, -2.4312036142263467], [-0.24370636476042984, 0.8087963857736555]], [[0.12185318238021492, 2.4287963857736536], [-0.3046329559505224, -3.2412036142263467]], [[0.06092659119010746, -0.8112036142263446], [-0.24370636476042984, -2.4312036142263467]], [[-0.18277977357030673, 2.4287963857736536], [-0.3046329559505224, -0.8112036142263446]], [[-0.18277977357030673, 2.4287963857736536], [-0.06092659119010746, -0.001203614226575608]], [[0.0, -2.4312036142263467], [0.24370636476042984, 4.048796385784975]], [[-0.24370636476042984, 3.2387963857736537], [-0.12185318238021492, -3.2412036142263467]], [[0.18277977357030673, -4.0512036142150265], [0.3046329559505224, -2.4312036142263467]], [[0.18277977357030673, 3.2387963857736537], [-0.06092659119010746, -0.8112036142263446]]], [[[-0.305, 0.81], [0.061, 1.62]], [[-0.244, -0.81], [0.244, 0.0]], [[0.183, 4.050000000000001], [-0.244, 2.43]], [[-0.305, 0.81], [0.183, 0.0]], [[-0.061, -0.81], [0.305, -1.62]], [[-0.122, 4.050000000000001], [-0.183, -1.62]], [[0.122, 0.81], [-0.061, 4.050000000000001]], [[0.244, -2.43], [0.122, -2.43]], [[-0.061, -0.81], [-0.122, 0.81]], [[-0.122, 3.24], [-0.183, -1.62]], [[0.305, 0.0], [-0.183, 4.050000000000001]], [[-0.305, 2.43], [0.0, -2.43]]], [[[0.05679510755133488, -3.311423571310734], [-0.11359021510266976, -1.691423571311285]], [[0.0, -0.07142357131188609], [0.283975537756705, -0.07142357131188609]], [[0.05679510755133488, -0.881423571311552], [-0.2271804302053395, -2.501423571310734]], [[-0.05679510755133488, -2.501423571310734], [0.1703853226540107, 1.5485764286887151]], [[0.1703853226540107, -0.07142357131188609], [0.11359021510266976, -0.881423571311552]], [[-0.05679510755133488, 1.5485764286887151], [0.283975537756705, -2.501423571310734]], [[-0.2271804302053395, -3.311423571310734], [0.05679510755133488, 2.3585764286892665]], [[-0.05679510755133488, 3.1685764286892666], [0.11359021510266976, 3.1685764286892666]], [[0.2271804302053395, 3.1685764286892666], [-0.05679510755133488, -0.881423571311552]], [[-0.283975537756705, -0.881423571311552], [-0.05679510755133488, 3.978576428689331]], [[0.2271804302053395, -1.691423571311285], [-0.11359021510266976, -3.311423571310734]], [[0.11359021510266976, 0.7385764286884481], [-0.283975537756705, -0.881423571311552]]], [[[-0.244, -2.43], [0.305, 2.43]], [[0.305, 2.43], [0.122, -4.050000000000001]], [[0.183, -3.24], [0.0, -1.62]], [[-0.061, 3.24], [-0.061, -0.81]], [[0.0, 2.43], [0.244, -2.43]], [[0.061, -2.43], [-0.305, -0.81]], [[0.122, -1.62], [0.061, 0.0]], [[-0.183, 4.050000000000001], [0.0, 4.050000000000001]], [[0.244, 3.24], [0.122, 0.81]], [[0.244, -0.81], [-0.244, -4.050000000000001]], [[-0.183, -1.62], [0.305, -3.24]], [[0.061, 1.62], [0.244, 2.43]]]]
L_out_8_1H_3C = [[[188.48498517503026, 178.64498517503017]], [[1.747129967184622, -4.335870032830271]], [[10.08973394376059, 11.054733943760608]], [[11.412519980171645, 22.850519980185634]], [[0.605873597625503, -0.6451264023745238]], [[13.527897498205371, 4.891897498184338]], [[-0.4239999999999474, 2.544000000001311]], [[1944.0522239487022, 1935.9142239486991]]]


epochs = 1300
learning_rate = 0.00001
L_donnees = jeu_donnees_jeu_loaded

time_beg = datetime.utcnow()
print("Time beg = ", time_beg)
print()

#aprend_8(L_8_1H_3C,L_out_8_1H_3C,jeu_donnees_teste_loaded,L_donnees,epochs,learning_rate)

time_finish = datetime.utcnow()

time = time_finish-time_beg
print()
print("Pour",epochs,"boucles et",len(L_donnees),"exemples, il faut",time)
print()


#teste_8(L_8_1H_3C,L_out_8_1H_3C,jeu_donnees_teste_loaded)
    
verif_prediction_8(L_8_1H_3C,L_out_8_1H_3C,api_key,city_name,1)

#Pour 1H 5C:
# Pour 10 000 boucles et 2200 exemples, il faut: 4:40:58.363606 
# Pour 500 boucles et 300 exemples, il faut: 0:01:53.685946 
# Pour 1000 boucles et 500 exemples, il faut 0:06:25.402592
# Pour 100 boucles et 2620 exemples, il faut 0:03:21.353533
# Pour 100 boucles et 2940 exemples, il faut 0:03:46.249218
# Pour 100 boucles et 3580 exemples, il faut 0:04:35.831866
# Pour 100 boucles et 3900 exemples, il faut 0:04:54.635055
# Pour 100 boucles et 4220 exemples, il faut 0:05:16.929002
# Pour 100 boucles et 5260 exemples, il faut 0:06:42.429948

#Pour 1H 3C:
# Pour 100 boucles et 5260 exemples, il faut 0:04:11.626115
# Pour 100 boucles et 5260 exemples, il faut 0:04:09.579353

#Pour 1H 2C:
# Pour 100 boucles et 5660 exemples, il faut 0:03:08.268818
    
#Pour 8AI 1H 3C:
# Pour 50 boucles et 5660 exemples, il faut 0:02:31.884578
# Pour 10 boucles et 5660 exemples, il faut 0:00:29.556829
# Pour 100 boucles et 6060 exemples, il faut 0:05:21.569745
# Pour 1300 boucles et 6060 exemples, il faut 1:10:22.095140
