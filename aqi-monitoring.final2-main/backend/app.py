import os
import requests
from datetime import datetime, timedelta
import math
import joblib
import random
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# CORS setup for Frontend communication
CORS(app, resources={r"/*": {"origins": ["https://aqi-monitoring-final2.vercel.app/","http://localhost:5173"]}}) 

WAQI_TOKEN = os.getenv("WAQI_TOKEN")
OWM_KEY = os.getenv("OPENWEATHER_API_KEY")

# --- LOAD MODELS ---
try:
    model = joblib.load('aqi_7day_model.pkl')
    le = joblib.load('city_encoder.pkl')
    print("✅ MODEL & ENCODER LOADED SUCCESSFULLY")
except Exception as e:
    print(f"❌ ERROR LOADING MODELS: {e}")

# --- UTILS ---
BREAKPOINTS = {
    "pm2_5": [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(250,1000,401,500)],
    "pm10": [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(430,1000,401,500)],
    "no2": [(0,40,0,50),(41,80,51,100),(81,180,101,200),(181,280,201,300),(281,400,301,400),(400,1000,401,500)],
    "so2": [(0,40,0,50),(41,80,51,100),(81,380,101,200),(381,800,201,300),(801,1600,301,400),(1600,5000,401,500)],
    "co": [(0,1,0,50),(1.1,2,51,100),(2.1,10,101,200),(10,17,201,300),(17,34,301,400),(34,100,401,500)],
    "o3": [(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),(209,748,301,400),(748,2000,401,500)]
}

def get_sub_index(cp, pollutant):
    if cp is None: return 0
    for (blo, bhi, ilo, ihi) in BREAKPOINTS[pollutant]:
        if blo <= cp <= bhi:
            return ((ihi - ilo)/(bhi - blo))*(cp - blo) + ilo
    return 500

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R*(2*math.atan2(math.sqrt(a), math.sqrt(1-a)))

# --- ROUTES ---

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    lat, lon = data.get('lat'), data.get('lon')
    
    delta = 1.0
    try:
        url = f"https://api.waqi.info/map/bounds/?token={WAQI_TOKEN}&latlng={lat-delta},{lon-delta},{lat+delta},{lon+delta}"
        stations = requests.get(url).json().get('data', [])
        valid = [s for s in stations if s.get('aqi') and s['aqi']!='-' and int(s['aqi'])>0]
    except: valid = []

    owm_res = requests.get(f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OWM_KEY}").json()
    comp = owm_res['list'][0]['components'] if 'list' in owm_res else {}

    sub = {
        "pm2_5": get_sub_index(comp.get('pm2_5'),"pm2_5"),
        "pm10": get_sub_index(comp.get('pm10'),"pm10"),
        "no2": get_sub_index(comp.get('no2'),"no2"),
        "so2": get_sub_index(comp.get('so2'),"so2"),
        "co": get_sub_index(comp.get('co')/1000,"co"),
        "o3": get_sub_index(comp.get('o3'),"o3")
    }

    dominant = max(sub, key=sub.get) if sub else "pm2_5"
    owm_aqi = sub[dominant] if sub else 0
    final_aqi = owm_aqi
    min_dist = 999

    if valid:
        distances = [haversine(lat,lon,float(s['lat']),float(s['lon'])) for s in valid]
        min_dist = min(distances)
        weighted = sum(int(s['aqi'])/(d+0.5) for s,d in zip(valid,distances))
        total = sum(1/(d+0.5) for d in distances)
        station_avg = weighted/total
        w = 0.9 if min_dist < 5 else 0.6 if min_dist < 25 else 0.3
        final_aqi = (station_avg * w) + (owm_aqi * (1-w))
        final_aqi = max(min(final_aqi, station_avg+30), station_avg-30)

    # --- DYNAMIC REASONS ---
    reasons_map = {
        "pm2_5": "Fine particles from vehicle exhaust, burning of fuels, or forest fires.",
        "pm10": "Dust from construction, roads, and wind-blown sea salt or pollen.",
        "no2": "High traffic density and combustion from power plants or cars.",
        "so2": "Industrial emissions, coal burning, or oil refineries nearby.",
        "co": "Incomplete combustion of fuels, often due to heavy traffic or wood burning.",
        "o3": "Reaction of sunlight with pollutants from cars and industries (Smog)."
    }

    diff = abs(final_aqi - owm_aqi)
    confidence = "High" if min_dist < 5 and diff < 20 else "Medium" if min_dist < 15 and diff < 50 else "Low"
    
    def get_cat_data(aqi):
        if aqi <= 50: return "Good", "#10b981"
        if aqi <= 100: return "Satisfactory", "#84cc16"
        if aqi <= 200: return "Moderate", "#f59e0b"
        if aqi <= 300: return "Poor", "#f97316"
        return "Severe", "#ef4444"

    cat, color = get_cat_data(round(final_aqi))

    return jsonify({
        "aqi": round(final_aqi), 
        "category": cat, 
        "color": color,
        "dominant_pollutant": dominant.upper().replace("_","."),
        "pollution_reason": reasons_map.get(dominant, "General urban pollution sources."),
        "components": comp, 
        "confidence": confidence, 
        "source": "Hybrid Engine"
    })

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.json
        lat, lon = data.get('lat'), data.get('lon')
        state_name = data.get('state', 'Delhi')
        
        curr_pm25 = data.get('pm25', 60)
        curr_pm10 = data.get('pm10', 100)
        curr_no2 = data.get('no2', 40)
        curr_nh3 = data.get('nh3', 20)
        curr_so2 = data.get('so2', 10)
        curr_co = data.get('co', 1.5)
        curr_o3 = data.get('o3', 50)

        weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OWM_KEY}&units=metric"
        w_res = requests.get(weather_url).json()
        forecast_list = w_res.get('list', [])

        preds = []
        base_date = datetime.now()
        
        try:
            state_encoded = le.transform([state_name])[0]
        except:
            state_encoded = 0 
        
        for i in range(1, 8):
            future_date = base_date + timedelta(days=i)
            idx = min(i * 8, len(forecast_list) - 1)
            w_data = forecast_list[idx] if forecast_list else {}
            temp = w_data.get('main', {}).get('temp', 25)

            input_features = [
                state_encoded, curr_pm25, curr_pm10, curr_no2, 
                curr_nh3, curr_so2, curr_co, curr_o3,
                future_date.day, future_date.month, future_date.weekday()
            ]
            
            predicted_aqi = model.predict([input_features])[0]
            variation = random.uniform(-5, 5) 
            display_aqi = round(predicted_aqi + variation)

            preds.append({
                "day": f"Day {i}",
                "date": future_date.strftime('%Y-%m-%d'),
                "aqi": display_aqi,
                "temp": round(temp),
                "condition": w_data.get('weather', [{}])[0].get('main', 'Clear')
            })

            curr_pm25 = predicted_aqi * 0.6
            curr_pm10 = predicted_aqi * 0.9

        return jsonify({"forecast": preds})

    except Exception as e:
        print(f"Forecast Error: {e}")
        return jsonify({"error": str(e)})

@app.route('/get-ai-advice', methods=['POST'])
def get_ai_advice():
    data = request.json
    persona = data.get('persona', 'Adult')
    aqi = data.get('aqi', 0)

    def get_category(aqi_val):
        if aqi_val <= 50: return "Good"
        elif aqi_val <= 100: return "Moderate"
        elif aqi_val <= 200: return "Poor"
        elif aqi_val <= 300: return "Very Poor"
        else: return "Severe"

    category = get_category(aqi)

    advice_bank = {
        "Good": {
            "Kid": ["Safe air. Children can play outside."],
            "Aged People": ["Air is clean. Normal activities are safe."],
            "Pregnant Women": ["No major risk. Fresh air is safe."],
            "Adult": ["Good air quality. No precautions needed."],
            "Sensitive Skin": ["Low irritation risk."],
            "Respiratory Issues": ["Safe conditions for breathing."]
        },
        "Moderate": {
            "Kid": ["Limit long outdoor play."],
            "Aged People": ["Avoid prolonged outdoor exposure."],
            "Pregnant Women": ["Stay cautious in polluted areas."],
            "Adult": ["Avoid heavy outdoor exercise."],
            "Sensitive Skin": ["Mild irritation possible."],
            "Respiratory Issues": ["Mild breathing discomfort possible."]
        },
        "Poor": {
            "Kid": ["Avoid outdoor play. Pollution harms lungs."],
            "Aged People": ["Stay indoors. Breathing risk increases."],
            "Pregnant Women": ["Avoid outdoor exposure."],
            "Adult": ["Wear mask. Avoid outdoor activity."],
            "Sensitive Skin": ["Cover skin. Irritation risk high."],
            "Respiratory Issues": ["Carry inhaler. Avoid exposure."]
        },
        "Very Poor": {
            "Kid": ["Stay indoors. High risk for children."],
            "Aged People": ["Strictly avoid outdoor activity."],
            "Pregnant Women": ["Stay indoors. High pollution risk."],
            "Adult": ["Use N95 mask. Avoid going outside."],
            "Sensitive Skin": ["High irritation risk."],
            "Respiratory Issues": ["Serious breathing risk."]
        },
        "Severe": {
            "Kid": ["Do not go outside. Hazardous air."],
            "Aged People": ["Emergency level. Stay indoors."],
            "Pregnant Women": ["Avoid exposure completely."],
            "Adult": ["Stay indoors. Use purifier."],
            "Sensitive Skin": ["Severe skin damage risk."],
            "Respiratory Issues": ["Dangerous condition. Seek help."]
        }
    }

    options = advice_bank.get(category, {}).get(persona, ["Stay safe and monitor air quality."])
    advice = random.choice(options)

    return jsonify({
        "aqi": aqi,
        "category": category,
        "persona": persona,
        "advice": advice
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
