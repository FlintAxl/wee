"""
Disease-specific recommendations for TomatoGuard - UPDATED FOR YOUR DATASET
"""

RECOMMENDATIONS = {
    # LEAF DISEASES
    "bacterial_spot_speck": {
        "name": "Bacterial Spot/Speck",
        "severity": "High",
        "recommendations": [
            "Remove and destroy infected plants immediately",
            "Use copper-based bactericides weekly",
            "Avoid overhead watering to reduce spread",
            "Practice crop rotation (3-4 years)",
            "Use disease-free seeds and transplants",
            "Sterilize tools between plants"
        ],
        "prevention": [
            "Plant resistant varieties when available",
            "Ensure good air circulation",
            "Avoid working in wet fields",
            "Use drip irrigation instead of sprinklers"
        ],
        "organic_control": [
            "Apply copper soap fungicides",
            "Use Bacillus subtilis products",
            "Neem oil applications every 7-14 days"
        ]
    },
    
    "early_blight": {
        "name": "Early Blight",
        "severity": "Medium",
        "recommendations": [
            "Apply chlorothalonil or mancozeb fungicides",
            "Remove lower infected leaves",
            "Improve air circulation",
            "Water at base of plants only",
            "Apply fungicides preventatively in humid weather"
        ],
        "prevention": [
            "Mulch around plants to prevent soil splash",
            "Space plants properly (24-36 inches apart)",
            "Stake plants for better air flow",
            "Rotate crops annually"
        ],
        "organic_control": [
            "Apply copper fungicides",
            "Use Bacillus subtilis products",
            "Baking soda spray (1 tbsp per gallon)",
            "Neem oil applications"
        ]
    },
    
    "late_blight": {
        "name": "Late Blight",
        "severity": "Critical",
        "recommendations": [
            "IMMEDIATE ACTION REQUIRED: Destroy all infected plants",
            "Apply fungicides containing chlorothalonil or mancozeb",
            "Notify nearby farmers of outbreak",
            "Do NOT compost infected material",
            "Apply fungicides every 5-7 days in wet weather"
        ],
        "prevention": [
            "Plant resistant varieties",
            "Monitor weather for blight-favorable conditions",
            "Use protective fungicides before symptoms appear",
            "Destroy volunteer tomato and potato plants"
        ],
        "organic_control": [
            "Copper fungicides (may provide limited control)",
            "Remove and destroy infected plants immediately",
            "Improve air circulation dramatically"
        ]
    },
    
    "septoria_leaf_spot": {
        "name": "Septoria Leaf Spot",
        "severity": "Medium",
        "recommendations": [
            "Remove infected leaves immediately",
            "Apply fungicides containing chlorothalonil",
            "Improve air circulation",
            "Water in morning only",
            "Keep leaves as dry as possible"
        ],
        "prevention": [
            "Mulch heavily around plants",
            "Rotate crops (2-3 years without tomatoes)",
            "Space plants adequately",
            "Remove plant debris at season end"
        ],
        "organic_control": [
            "Copper-based fungicides",
            "Baking soda sprays",
            "Neem oil applications",
            "Remove infected leaves promptly"
        ]
    },
    
    # FRUIT DISEASES
    "anthracnose": {
        "name": "Anthracnose",
        "severity": "Medium-High",
        "recommendations": [
            "Harvest fruits promptly when ripe",
            "Remove and destroy infected fruits",
            "Apply fungicides containing chlorothalonil",
            "Avoid overhead irrigation",
            "Stake plants to keep fruits off ground"
        ],
        "prevention": [
            "Use disease-free seeds",
            "Mulch with black plastic or straw",
            "Rotate crops (3-year rotation)",
            "Control weeds that harbor fungus"
        ],
        "organic_control": [
            "Copper fungicides",
            "Bacillus subtilis products",
            "Remove infected fruits immediately",
            "Improve air circulation"
        ]
    },
    
    "gray_mold": {
        "name": "Gray Mold (Botrytis)",
        "severity": "Medium",
        "recommendations": [
            "Remove infected plant parts immediately",
            "Improve air circulation dramatically",
            "Reduce humidity around plants",
            "Apply fungicides containing iprodione or cyprodinil",
            "Avoid wounding plants during handling"
        ],
        "prevention": [
            "Space plants properly",
            "Water early in day",
            "Remove plant debris",
            "Sterilize pruning tools",
            "Avoid excess nitrogen fertilization"
        ],
        "organic_control": [
            "Baking soda sprays",
            "Potassium bicarbonate",
            "Improved air circulation",
            "Biological controls (Trichoderma spp.)"
        ]
    },
    
    "blossom_end_rot": {
        "name": "Blossom End Rot",
        "severity": "Low (Physiological)",
        "recommendations": [
            "Maintain consistent soil moisture",
            "Apply calcium nitrate or calcium chloride spray",
            "Mulch to conserve moisture",
            "Avoid excessive nitrogen fertilization",
            "Remove affected fruits"
        ],
        "prevention": [
            "Test soil pH (6.5-6.8 ideal)",
            "Add lime if soil pH is low",
            "Use balanced fertilizer",
            "Water consistently, avoid drought stress",
            "Improve drainage in heavy soils"
        ],
        "organic_control": [
            "Add crushed eggshells to soil",
            "Use calcium-rich organic amendments",
            "Consistent watering schedule",
            "Compost to improve soil structure"
        ]
    },
    
    "buckeye_rot": {
        "name": "Buckeye Rot",
        "severity": "Medium",
        "recommendations": [
            "Remove infected fruits immediately",
            "Stake plants to keep fruits off soil",
            "Mulch with plastic or straw",
            "Apply fungicides containing mefenoxam or metalaxyl",
            "Improve soil drainage"
        ],
        "prevention": [
            "Use drip irrigation",
            "Rotate crops (3-4 year rotation)",
            "Avoid planting in poorly drained areas",
            "Remove plant debris at season end"
        ],
        "organic_control": [
            "Copper fungicides",
            "Improve drainage",
            "Mulch heavily",
            "Remove infected fruits promptly"
        ]
    },
    
    # HEALTHY CLASSES
    "healthy": {
        "name": "Healthy Plant",
        "severity": "None",
        "recommendations": [
            "Plant appears healthy! Continue regular maintenance",
            "Monitor for any early signs of disease",
            "Maintain optimal growing conditions"
        ],
        "prevention": [
            "Continue current good practices",
            "Regular inspection (weekly recommended)",
            "Maintain proper spacing and air flow",
            "Use balanced fertilization"
        ],
        "organic_control": [
            "Continue organic practices",
            "Consider preventative organic sprays in high-risk periods",
            "Maintain soil health with compost"
        ]
    }
}

def get_recommendations(disease_name):
    """Get recommendations for a specific disease"""
    return RECOMMENDATIONS.get(disease_name, {
        "name": disease_name.replace('_', ' ').title(),
        "severity": "Unknown",
        "recommendations": ["Consult with agricultural expert for specific diagnosis"],
        "prevention": ["Practice good crop management and regular monitoring"],
        "organic_control": ["Use general organic practices and maintain plant health"]
    })