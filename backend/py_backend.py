import numpy as np
import json
from json import JSONEncoder
import os
from collections import defaultdict
# import AccidentPropensity as AP

import pandas as pd
import math
import numpy as np
import json

import datetime
from meteostat import Stations,Daily,Hourly
import pickle
from scipy import stats

# function to calculate the distance between two points
def distance(point1, point2):
    lat1, lon1 = point1
    lat2, lon2 = point2
    km_per_lat = 110.574
    km_per_lon = 111.320
    dx = (lon2 - lon1) * km_per_lon * math.cos((lat1 + lat2) / 2)
    dy = (lat2 - lat1) * km_per_lat
    return math.sqrt(dx**2 + dy**2)

def make_agg2(data):
    acc_dic_list = []
    # num_accs = 0
    for _,i in data.items():
        acc_dic_list.append(i['accident_time'])
        # num_accs+=i["num_accidents"]

    result_dict = defaultdict(int)
    for i in acc_dic_list:
        for key, value in i.items():
            result_dict[key] += value
    return dict(result_dict)

def sum_num_accidents(data):
    total_accidents = 0
    for segment in data:
        total_accidents += data[segment]["num_accidents"]
    
    return total_accidents

# function to calculate the distance between a point and a line segment
def distance_to_segment(point, segment_start, segment_end):
    px, py = point
    x1, y1 = segment_start
    x2, y2 = segment_end
    dx, dy = x2 - x1, y2 - y1
    segment_length_squared = dx*dx + dy*dy
    if segment_length_squared == 0:
        return distance(point, segment_start)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / segment_length_squared))
    x = x1 + t * dx
    y = y1 + t * dy
    return distance(point, (x, y))

# function to find accidents on a given route within a maximum distance
def find_accidents_on_route(start_point, end_point, all_relevant_accidents):
    # maximal distance of accidents from route in kilometers
    max_distance = 0.05
    # create a mask for accidents that are within the maximum distance from the route
    mask = all_relevant_accidents.apply(lambda row: distance_to_segment((row['Start_Lat'], row['Start_Lng']), start_point, end_point) <= max_distance, axis=1)

    # return the accidents that match the mask
    accidents = all_relevant_accidents.loc[mask]
    return accidents

# main function that splits route data into 5 segments and finds corresponding accidents, calculates metrics, and generates an output json file
def find_accidents(json_data):
    # load route data from json file
    # with open(route_data) as f:
    #     json_data = json.load(f)

    # create pandas DataFrame from loaded route data with columns lat, lng, assign (with the 0 or 1)
    data_dict_list = [{"lat": item[0]["lat"], "lng": item[0]["lng"], "assign": item[1]} for item in json_data]
    route_data = pd.DataFrame(data_dict_list)

    # split the route DataFrame into 5 equally sized parts
    df_list = np.array_split(route_data, 5)

    # loop through the 5 route segments
    route_dict = {}
    accident_dict = {}
    for i, df in enumerate(df_list):
        
        # safe each split dataframe as a new route_df_{i} dataframe
        route_dict[f"route_df_{i+1}"] = df

        # reduce route data to those rows that are 100m points & create a new DataFrame to store the results
        route_data_assigned = df[df['assign'] == 1]
        route_data_lat_lng = route_data_assigned[['lat','lng']]
        accidents_df = pd.DataFrame()
        
        # loop through the pairs of subsequent coordinates per route segment
        for j in range(len(route_data_lat_lng) - 1):
            # get start and end point by using the subsequent coordinate pair
            start_point = j
            end_point = j + 1

            # retrieve relevant accident data; uses the midpoint between start and end point of a coordinate pair
            point_lat = (route_data_lat_lng.iloc[end_point]['lat'] + route_data_lat_lng.iloc[start_point]['lat']) / 2
            point_lng = (route_data_lat_lng.iloc[end_point]['lng'] + route_data_lat_lng.iloc[start_point]['lng']) / 2
            dataset_id = np.char.add(np.char.add(np.char.mod('%s', point_lat.astype(str)[:4]), '_'), point_lng.astype(str)[:5])
            # loads file only if it exists (there might be route parts where no accident file exists as there are no accidents (i.e., on non-highway routes))
            filename = f'python_code/data/ga_accidents_{dataset_id}.csv' ##########################
            if os.path.isfile(filename):
                all_relevant_accidents = pd.read_csv(filename)
                # call the accident-retrieving function and append the results to the segment-specific accidents DataFrame
                accidents_df = pd.concat([accidents_df, find_accidents_on_route(route_data_lat_lng.iloc[start_point], route_data_lat_lng.iloc[end_point], all_relevant_accidents)], ignore_index=True)
        
        # drop duplicate rows from the accidents DataFrame; as we're searching for accidents around each segment within 50 meters, there could be duplicates in the accidents set
        accidents_df.drop_duplicates(inplace=True)
        
        # add segment-specific accident dataframe to the accident dictionary & assign a name that reflects the segment number
        accident_dict[f"accidents_df_{i+1}"] = accidents_df

    # calculate api per segment
    api_value_dict = {}
    accidents_dfs = list(accident_dict.values())
    for i, df in enumerate(accidents_dfs, start=1):
        api_value = df["Severity"].sum() / 6035011
        api_value_dict[f'api_{i}'] = round(api_value, 8)

    # convert api values into colors
    api_color_dict = api_value_dict.copy()
    min_api_value = min(api_color_dict.values())
    max_api_value = max(api_color_dict.values())
    for key in api_color_dict:
        api_color_dict[key] = (api_value_dict[key] - min_api_value) / (max_api_value - min_api_value)
    def get_hex_color(value):
        # convert a normalized value to a hex color code representing a gradient from green to red.
        r = int(255 * value)
        g = int(255 * (1 - value))
        b = 0
        return f'{r:02x}{g:02x}{b:02x}'
    for key in api_color_dict:
        api_color_dict[key] = get_hex_color(api_color_dict[key])

    # number of accidents per segment
    num_accidents_dict = {}
    accidents_dfs = list(accident_dict.values())
    for i, df in enumerate(accidents_dfs, start=1):
        num_accidents = df.shape[0]
        num_accidents_dict[f'acci_{i}'] = num_accidents

    # time of the day of accidents per segment
    accident_time_dict = {}
    accidents_dfs = list(accident_dict.values())
    for i, df in enumerate(accidents_dfs, start=1):
        try:
            df['Start_Time'] = pd.to_datetime(df['Start_Time'])
            hour_counts = df['Start_Time'].dt.hour.value_counts().sort_index().to_dict()
            for hour in range(24):
                if hour not in hour_counts:
                    hour_counts[hour] = 0
            hour_counts = dict(sorted(hour_counts.items()))
            accident_time_dict[f'time_{i}'] = hour_counts
        except:
            pass

    # top 5 weather conditions per segment
    weather_condition_dict = {}
    accidents_dfs = list(accident_dict.values())
    top_n = 5 # number of top weather conditions to include in the dictionary
    for i, df in enumerate(accidents_dfs, start=1):
        weather_counts = df['Weather_Condition'].value_counts().head(top_n).to_dict()
        weather_condition_dict[f'weather_{i}'] = weather_counts

    # create a dictionary to store the JSON data & loop over the segments and add the API and route data to the JSON dictionary
    json_dict = {}
    route_dfs = list(route_dict.values())   
    for i in range(5):
        segment_name = f"segment_{i+1}"
        json_dict[segment_name] = {}
        json_dict[segment_name]["api_value"] = api_value_dict[f"api_{i+1}"]
        json_dict[segment_name]["api_color"] = api_color_dict[f"api_{i+1}"]
        json_dict[segment_name]["num_accidents"] = num_accidents_dict[f"acci_{i+1}"]
        # json_dict[segment_name]["accident_time"] = accident_time_dict[f"time_{i+1}"]
        json_dict[segment_name]["weather_condition"] = weather_condition_dict[f"weather_{i+1}"]
        json_dict[segment_name]["route"] = route_dfs[i][["lat", "lng"]].to_dict(orient="records")
    
    
    
    # data_list = make_agg2(json_dict).items()

    # Create list of dictionaries for the histogram data
    # histogram_data = [{'x': k, 'y': v} for k, v in data_list]
    histogram_data = {}
    # Serialize histogram data as JSON
    histogram_json = json.dumps(histogram_data)

    with open("outputs/hour_acc_hist.json", "w") as f:
        json.dump(histogram_json, f)
    
    # num_accs = 0
    # for k, v in data_list:
    #     num_accs+=v
    # json_dict['total_accidents'] = num_accs
    
    # export the JSON file
    with open("outputs/backend_output.json", "w") as f:
        json.dump(json_dict, f)
    
    
    return json_dict

##################################Log Odds code starts##############################################



def haversine_distance(lat1, lon1, lat2, lon2):
    """
    :param lat1: latitude point 1
    :param lon1: longitude point 1
    :param lat2: latitude point 2
    :param lon2: longitude point 2
    :return: distance in Kms b/w 2 points
    """
    p = (np.pi/180)
    hav = 0.5 - np.cos((lat2-lat1)*p)/2 + np.cos(lat1*p)*np.cos(lat2*p) * (1-np.cos((lon2-lon1)*p)) / 2
    return 12742 * np.arcsin(np.sqrt(hav))

def closest_airport(lat,long):
    """
    :param lat: latitude current location
    :param long: longitude current location
    :return: closest weather station
    """
    latitude_longitude_repo = pd.read_csv("python_code/logOddsData/station_lat_longitude.csv", index_col=0)  ########Change addresses
    latitude_longitude_repo["distance"] = latitude_longitude_repo.apply(
        lambda x: haversine_distance(lat1=lat, lon1=long, \
                                     lat2=x["latitude"], lon2=x["longitude"]), axis=1)
    latitude_longitude_repo = latitude_longitude_repo.sort_values(by=["distance"])
    #     assert len(closest_aiport)==1, "More than 1 closest airport"
    return latitude_longitude_repo.iloc[0]["icao"], latitude_longitude_repo.index[0]

def get_discreet_time():
    """
    :return: current discretised time
    """
    time_now = datetime.datetime.now()
    hr_session = "day" if time_now.hour >= 9 and time_now.hour <= 21 else "night"
    season = "summer" if time_now.month >= 6 and time_now.month <= 9 else "winter"
    time_ad = f"{season}_{hr_session}"
    return time_ad

def get_realtime_weather(airport_api_id):
    """
    Get realtime weather at a location
    :param lat: point latitude
    :param long: point longitude
    :return: current weather
    """
    time_now = pd.Timestamp.now().round('60min').to_pydatetime()
    time_one = time_now + datetime.timedelta(hours=1)
    data = Hourly(airport_api_id, start=time_now, end=time_now)
    current_weather = data.fetch()
    return current_weather

def transform_weather(weather_avg_6):
    """
    Convert weather from metric to imperial
    :param weather_avg_6: point latitude
    :return: weather in imperial
    """
    temp_f = weather_avg_6[0] * 1.8 + 32
    hum = weather_avg_6[1]
    precip_in = weather_avg_6[2] / 25.4
    wspd_mph = weather_avg_6[4] / 1.609
    pres_in = weather_avg_6[5] * 0.02952998057228486
    return np.array([temp_f, hum, pres_in, precip_in, wspd_mph])

def weather_odds_helper(kde_pca_accidents,kde_pca_history,weather_now,if_bad=False,N=100):
    """
    :param kde_pca_accidents: KDE PCA dict for airport from accidents data
    :param kde_pca_history: KDE PCA dict for airport from history data
    :param weather_now: current weather
    :param N: number of trials for sample mean
    :return: Odds
    """
    #get discreet time
    time_ad = get_discreet_time()
    #get weather avg from history
    kde_kernel_accidents = kde_pca_accidents["kde"][time_ad]
    kde_kernel_accidents = stats.gaussian_kde(kde_kernel_accidents.dataset)
    kde_kernel_history = kde_pca_history["KDE"][time_ad]
    kde_kernel_history = stats.gaussian_kde(kde_kernel_history.dataset)
    weather_avg_samples = np.array([kde_kernel_history.resample(160).mean(axis=1) for i in range(N)])
    weather_avg_hist = np.mean(weather_avg_samples,axis=0)
    #densities from hist data
    density_weather_avg_hist = kde_kernel_history.evaluate(weather_avg_hist)[0]
    weather_cols_hist = ["temp","rhum","prcp","wdir","wspd","pres"]
    weather_now_pca_hist = kde_pca_history["PCA"].transform(weather_now[weather_cols_hist])[:,:2]
    density_weather_now_hist = kde_kernel_history.evaluate(weather_now_pca_hist)[0]
    #densities from accidents data
    weather_now_acc = transform_weather(weather_now[weather_cols_hist].values[0])
    weather_avg_6 = (weather_avg_hist @ kde_pca_history["PCA"].components_[:2,:]) + kde_pca_history["PCA"].mean_
    weather_avg_acc = transform_weather(weather_avg_6)
    weather_now_acc_pca = kde_pca_accidents["pca"].transform(weather_now_acc.reshape(1,-1))
    weather_avg_acc_pca = kde_pca_accidents["pca"].transform(weather_avg_acc.reshape(1,-1))
    density_weather_avg_acc = kde_kernel_accidents.evaluate(weather_avg_acc_pca)[0]
    density_weather_now_acc = kde_kernel_accidents.evaluate(weather_now_acc_pca)[0]
    if if_bad:
        density_weather_now_acc = density_weather_now_acc*1.1
    # print (f""" density_weather_now_acc:{density_weather_now_acc:.6f}
    # density_weather_avg_acc:{density_weather_avg_acc:.6f}
    # density_weather_avg_hist:{density_weather_avg_hist:.6f}
    # density_weather_now_hist:{density_weather_now_hist:.6f}""")
    # log_odds = (np.log(density_weather_now_acc) -\
    #             np.log(density_weather_avg_acc) +\
    #            np.log(density_weather_avg_hist) - \
    #            np.log(density_weather_now_hist))
    return np.exp(log_odds)

def get_odds(lat,long, N= 250):
    """
    :param lat: latitude for current location
    :param long: longitude for current location
    :param N: #trials for estimating KDE mean
    :return:
    """
    # find closest airport
    airport_icao, airport_api_id = closest_airport(lat, long)
    # print (f"Nearest airport weather stn to ({lat:.4f},{long:.4f}) is {airport_icao}")
    # find realtime weather
    # print ("Fetching current weather")
    weather_now = get_realtime_weather(airport_api_id)
    # read kde estimate for historical data
    with open('python_code/logOddsData/accidents_kde.pickle', 'rb') as f:  ######Change address
        # Load the data from the pickle file
        kde_accidents_dict = pickle.load(f)
    with open('python_code/logOddsData/kde_dict.pickle', 'rb') as f:  ##############Change address
        # Load the data from the pickle file
        kde_historical_dict = pickle.load(f)
    kde_pca_accidents = kde_accidents_dict[airport_icao]
    kde_pca_history = kde_historical_dict[airport_icao]
    # print ("Densities for weather now")
    weather_odds_normal = weather_odds_helper(kde_pca_accidents, kde_pca_history, weather_now,False,N)
    weather_now_bad = weather_now.copy()
    kde_pca_accidents = kde_accidents_dict["KBMG"]
    weather_bad_acc_samples = kde_pca_accidents["kde"][get_discreet_time()].resample(250)
    kde2 = stats.gaussian_kde(kde_pca_accidents["kde"][get_discreet_time()].dataset)
    weather_bad_acc_samples_density = kde2.evaluate(weather_bad_acc_samples)
    # print(weather_bad_acc_samples.shape,weather_bad_acc_samples_density.shape)
    weather_bad = weather_bad_acc_samples[:,np.argmax(weather_bad_acc_samples_density)]
    weather_bad_5cols = kde_pca_accidents["pca"].inverse_transform(weather_bad)
    weather_now_bad["prcp"] = weather_bad_5cols[3] * 25.4
    weather_now_bad["temp"] = (weather_bad_5cols[0]-32)/1.8
    weather_now_bad["pres"] = weather_bad_5cols[2] / 0.02952998057228486
    weather_now_bad["wspd"] = weather_bad_5cols[4] * 1.609
    # print("Densities for weather bad")
    weather_odds_bad = weather_odds_helper(kde_pca_accidents, kde_pca_history, weather_now_bad,True, N)
    # print(f"\nOdds for weather_now={weather_odds_normal:.5f} vs Odds for weather_bad={weather_odds_bad:.5f}")
    return (weather_odds_normal,weather_odds_bad)




##################################Log Odds code ends##############################################    
    


################################# Main code ####################################################

if __name__ == "__main__":
    # print(os.getcwd())
    # data = read_data()
    # processed_data = process_data(data)
    # colored_data =  dummy_function(processed_data)
    # write_color_to_file(colored_data)
    # print(os.getcwd())
    with open('python_code/data.json') as f:
        json_data = json.load(f)
    Acc_prop_return = find_accidents(json_data)
    # print(json_data[0])
    lat = Acc_prop_return["segment_1"]["route"][0]["lat"]
    lng = Acc_prop_return["segment_1"]["route"][0]["lng"]
    # print("Lattitude passed = ", lat, "Long passed = ",lng)
    odds_dict = {}
    try:
        odds1, odds2 = get_odds(lat,lng, N= 100)
        odds_dict['odds1'] = odds1
        odds_dict['odds2'] = odds2
        
    except:
        odds_dict['odds1'] = -1
        odds_dict['odds2'] = -1
    
    with open("outputs/logOdds_output.json", "w") as f:
        json.dump(odds_dict, f)


    # LogOdds.haversine_distance(30, 40, 50, 60)
    print("Done!")

# data = read_data()
# processed_data = process_data(data)
# colored_data =  dummy_function(processed_data)
# write_color_to_file(colored_data)
# print(os.getcwd())
# AP.find_accidents('data.json')
# # LogOdds.haversine_distance(30, 40, 50, 60)
# print("Done!", LogOdds.haversine_distance(30, 40, 50, 60))   