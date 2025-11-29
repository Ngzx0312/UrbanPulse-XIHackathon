import pandas as pd
import numpy as np

# CONFIG: Generate 10,000 scenarios representing KL streets
NUM_SAMPLES = 10000

# KL DISTRICT PROFILES (Source: OpenDOSM Proxy)
DISTRICTS = {
    "Bukit Bintang": {"density": 11000, "traffic_base": 8500}, # Commercial Hub
    "Cheras":        {"density": 9500,  "traffic_base": 7000}, # Residential
    "Bangsar":       {"density": 6500,  "traffic_base": 5500}, # Mixed
}

def generate_dataset():
    data = []
    print(f"🇲🇾 Generating {NUM_SAMPLES} scenarios based on KL topology...")

    for _ in range(NUM_SAMPLES):
        # 1. Random Context
        dist_name = np.random.choice(list(DISTRICTS.keys()))
        profile = DISTRICTS[dist_name]
        
        # Weather: 0=Rain (25%), 1=Cloudy (30%), 2=Sunny (45%)
        weather = np.random.choice([0, 1, 2], p=[0.25, 0.30, 0.45])
        
        # Intervention: 0=None, 1=Trees, 2=Bike Lane, 3=Emergency Lane
        intervention = np.random.randint(0, 4)
        
        # 2. Simulate Traffic (TomTom Logic)
        # Rain reduces volume slightly (people cancel trips) but slows speed
        vol_noise = np.random.normal(0, 500)
        traffic = profile["traffic_base"] + vol_noise
        if weather == 0: traffic *= 0.85 
        
        # 3. Calculate "Ground Truth" Logic
        
        # --- STRESS SCORE (0-100) ---
        # Trees (Int=1) cool the street -> Lower Stress
        # Rain (W=0) + Bike Lane (Int=2) = Wasted infrastructure -> Stress Penalty
        base_temp = 34.0 if weather == 2 else 27.0
        if intervention == 1: base_temp -= 2.0 
        
        traffic_load = traffic / 10000
        stress = (traffic_load * 50) + ((base_temp - 24) * 3)
        if intervention == 2 and weather == 0: stress += 15 # Planning failure
        stress = np.clip(stress, 20, 95)

        # --- ROI (RM Millions) ---
        # Based on MOH Asthma Prevalence (6.3%)
        pollution_drop = 0.0
        if intervention == 1: pollution_drop = 0.08
        if intervention == 2: pollution_drop = 0.15 if weather > 0 else 0.02
        
        cases_avoided = (profile["density"] * 0.063) * pollution_drop
        roi = (cases_avoided * 1500) / 1_000_000 # Monetize
        
        data.append([dist_name, traffic, weather, intervention, profile["density"], round(stress,1), round(roi,3)])

    # Save to CSV
    df = pd.DataFrame(data, columns=["district", "traffic", "weather", "intervention", "density", "stress", "roi"])
    df.to_csv("kl_training_data.csv", index=False)
    print("Data Generated: kl_training_data.csv")

if __name__ == "__main__":
    generate_dataset()