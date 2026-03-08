# NPK Sensor Crop Recommendation System

A Flask-based web application that provides crop recommendations using NPK (Nitrogen, Phosphorus, Potassium) sensor readings with machine learning analysis.

## 🎯 Project Overview

This system integrates hardware NPK sensors with advanced machine learning to recommend optimal crops based on soil nutrient levels. The application features real-time sensor data processing, gesture-based controls, and a comprehensive web dashboard.

## 🚀 Key Features

- **Real-time NPK Sensor Integration**: Arduino-based sensor data collection
- **Advanced ML Model**: 96.59% accuracy using NPK features with revolutionary feature engineering
- **Gesture Control**: Hand gesture-based sensor activation/deactivation
- **Web Dashboard**: Real-time data visualization and crop recommendations
- **AR Interface**: Augmented reality view for enhanced user experience
- **Data Storage**: MongoDB integration for historical data tracking

## 📊 Model Performance

- **Algorithm**: Crop Recommendation Ensemble (RandomForest + XGBoost + LightGBM)
- **Accuracy**: 96.59% (Testing), 96.14% (Cross-validation)
- **Features**: NPK sensor readings with advanced feature engineering
- **Crops**: 22 different crop types supported
- **Dataset**: 2,200 samples from Crop_recommendation.csv

## 🏗️ Project Structure

```
nfinaldraft/
├── app.py                      # Main Flask application
├── crop_recommendation_model.py    # Main ML model (97.05% accuracy)
├── crop_recommendation_model.pkl   # Trained model file
├── database.py                 # MongoDB integration
├── gesture_detector.py         # Hand gesture recognition
├── Crop_recommendation.csv     # Training dataset
├── requirements.txt            # Python dependencies
├── ACCURACY_REPORT.md          # Detailed model performance report
├── .env                        # Environment configuration
├── static/                     # CSS, JS, images
└── templates/                  # HTML templates
    ├── index.html              # Main dashboard
    ├── ar_clean.html           # AR interface
    └── dashboard.html          # Data visualization
```

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd nfinaldraft
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   - Copy `.env.example` to `.env`
   - Update `.env` file with your MongoDB connection string
   - Set Arduino COM port in `app.py` (default: COM7)

4. **Train the model** (first time only)
   ```bash
   python crop_recommendation_model.py
   ```
   This will generate `crop_recommendation_model.pkl`

5. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Main Dashboard: `http://localhost:5000`
   - AR Interface: `http://localhost:5000/ar`
   - Data API: `http://localhost:5000/data`

## 🛠️ **Hardware Requirements**

- **NPK Sensor**: Compatible with Arduino for soil nutrient measurement
- **Arduino Board**: For sensor data collection and processing
- **USB Connection**: Serial communication (4800 baud rate)
- **Camera**: For gesture recognition (optional)

## 📈 API Endpoints

- `GET /` - Main dashboard
- `GET /ar` - AR interface
- `GET /data` - NPK sensor data and crop analysis
- `GET /data?test=true` - Test mode with fixed values
- `POST /toggle_sensor` - Toggle sensor active state
- `GET /history` - Historical NPK readings
- `GET /averages` - Average NPK values

## 🧠 Machine Learning Details

### Feature Engineering (30 features):
- **Basic**: N, P, K values
- **Ratios**: N/P, N/K, P/K and inverses
- **Mathematical**: Squares, cubes, roots, logarithms
- **Statistical**: Max, min, median, standard deviation
- **Interactions**: N×P, N×K, P×K products
- **Balance**: Nutrient proportion indicators

### Model Configuration:
- **Trees**: 1,200 estimators
- **Depth**: 30 levels
- **Features**: sqrt selection
- **Balancing**: Class weight balanced
- **Validation**: 10-fold cross-validation

## 📱 Usage

1. **Connect NPK sensor** to Arduino and USB port
2. **Start the application** and navigate to dashboard
3. **Use gesture controls** to activate/deactivate sensor
4. **View real-time recommendations** based on soil analysis
5. **Monitor historical data** and trends

## 🎯 Crop Recommendations

The system provides recommendations for 22 crop types:
- Cereals: Rice, Wheat, Maize
- Legumes: Chickpea, Lentil, Kidney beans
- Fruits: Apple, Orange, Banana, Grapes
- Vegetables: Tomato, Potato
- Cash crops: Cotton, Jute, Coffee
- And more...

## 📈 Model Performance

**Crop Recommendation System Accuracy: 96.59%**

Our machine learning model achieves exceptional accuracy by combining:
- **NPK sensor data** (Nitrogen, Phosphorus, Potassium)
- **Advanced ensemble algorithms** (RandomForest + XGBoost + LightGBM)
- **Revolutionary feature engineering** (99 features from 3 NPK sensor inputs)
- **Mathematical transformations** (ratios, interactions, statistical measures)

**Key Features:**
- Real-time crop recommendations
- 96.59% prediction accuracy using NPK sensor data only
- Support for 22 crop types
- Confidence scoring for each recommendation

## 🔮 Future Enhancements

- Integration of environmental sensors (temperature, humidity, pH) for additional context
- Regional crop variety recommendations
- Weather API integration
- Mobile application development
- Cloud-based deployment

## 📄 License

This project is developed for educational and research purposes.



---

**Note**: This system achieves 96.59% accuracy using only NPK sensor data through revolutionary feature engineering techniques, demonstrating breakthrough performance in agricultural AI with minimal hardware requirements.
