import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             ExtraTreesClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
import os

class CropRecommendationModel:
    def __init__(self):
        self.model_path = 'crop_recommendation_model.pkl'
        self.dataset_path = 'Crop_recommendation.csv'
        self.model = None
        self.scaler = None
        self.label_encoder = None
        
        if os.path.exists(self.model_path):
            self._load_model()
        else:
            print("Crop Recommendation model not found. Training with NPK sensor data...")
            self._train_model()
    
    def _create_features(self, df):
        """Create comprehensive features from NPK + Environmental sensor data"""
        # Use NPK + Environmental features from the dataset
        enhanced_df = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'label']].copy()
        
        # Basic transformations for all features
        for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph']:
            enhanced_df[f'{col}_sq'] = enhanced_df[col] ** 2
            enhanced_df[f'{col}_cube'] = enhanced_df[col] ** 3
            enhanced_df[f'{col}_sqrt'] = np.sqrt(enhanced_df[col])
            enhanced_df[f'{col}_log'] = np.log1p(enhanced_df[col])
            enhanced_df[f'{col}_inv'] = 1 / (enhanced_df[col] + 0.01)
        
        # NPK ratios (most important for crops)
        enhanced_df['N_P_ratio'] = enhanced_df['N'] / (enhanced_df['P'] + 0.01)
        enhanced_df['N_K_ratio'] = enhanced_df['N'] / (enhanced_df['K'] + 0.01)
        enhanced_df['P_K_ratio'] = enhanced_df['P'] / (enhanced_df['K'] + 0.01)
        enhanced_df['P_N_ratio'] = enhanced_df['P'] / (enhanced_df['N'] + 0.01)
        enhanced_df['K_N_ratio'] = enhanced_df['K'] / (enhanced_df['N'] + 0.01)
        enhanced_df['K_P_ratio'] = enhanced_df['K'] / (enhanced_df['P'] + 0.01)
        
        # NPK statistics
        enhanced_df['NPK_total'] = enhanced_df['N'] + enhanced_df['P'] + enhanced_df['K']
        enhanced_df['NPK_mean'] = enhanced_df['NPK_total'] / 3
        enhanced_df['NPK_std'] = enhanced_df[['N', 'P', 'K']].std(axis=1)
        enhanced_df['NPK_var'] = enhanced_df[['N', 'P', 'K']].var(axis=1)
        enhanced_df['NPK_max'] = enhanced_df[['N', 'P', 'K']].max(axis=1)
        enhanced_df['NPK_min'] = enhanced_df[['N', 'P', 'K']].min(axis=1)
        enhanced_df['NPK_range'] = enhanced_df['NPK_max'] - enhanced_df['NPK_min']
        enhanced_df['NPK_median'] = enhanced_df[['N', 'P', 'K']].median(axis=1)
        
        # NPK proportions
        enhanced_df['N_prop'] = enhanced_df['N'] / enhanced_df['NPK_total']
        enhanced_df['P_prop'] = enhanced_df['P'] / enhanced_df['NPK_total']
        enhanced_df['K_prop'] = enhanced_df['K'] / enhanced_df['NPK_total']
        
        # NPK interactions
        enhanced_df['NP_product'] = enhanced_df['N'] * enhanced_df['P']
        enhanced_df['NK_product'] = enhanced_df['N'] * enhanced_df['K']
        enhanced_df['PK_product'] = enhanced_df['P'] * enhanced_df['K']
        enhanced_df['NPK_product'] = enhanced_df['N'] * enhanced_df['P'] * enhanced_df['K']
        
        # Geometric means
        enhanced_df['NP_geomean'] = np.sqrt(enhanced_df['N'] * enhanced_df['P'])
        enhanced_df['NK_geomean'] = np.sqrt(enhanced_df['N'] * enhanced_df['K'])
        enhanced_df['PK_geomean'] = np.sqrt(enhanced_df['P'] * enhanced_df['K'])
        enhanced_df['NPK_geomean'] = (enhanced_df['N'] * enhanced_df['P'] * enhanced_df['K']) ** (1/3)
        
        # Harmonic means
        enhanced_df['NP_harmean'] = 2 / (1/(enhanced_df['N']+0.01) + 1/(enhanced_df['P']+0.01))
        enhanced_df['NK_harmean'] = 2 / (1/(enhanced_df['N']+0.01) + 1/(enhanced_df['K']+0.01))
        enhanced_df['PK_harmean'] = 2 / (1/(enhanced_df['P']+0.01) + 1/(enhanced_df['K']+0.01))
        
        # Balance indicators
        enhanced_df['npk_balance'] = 1 - (abs(enhanced_df['N_prop'] - 0.33) + 
                                         abs(enhanced_df['P_prop'] - 0.33) + 
                                         abs(enhanced_df['K_prop'] - 0.33))
        
        # Dominant nutrient indicators
        dominant = enhanced_df[['N', 'P', 'K']].idxmax(axis=1)
        enhanced_df['N_dominant'] = (dominant == 'N').astype(int)
        enhanced_df['P_dominant'] = (dominant == 'P').astype(int)
        enhanced_df['K_dominant'] = (dominant == 'K').astype(int)
        
        # Advanced polynomial features
        enhanced_df['N2_P'] = enhanced_df['N']**2 * enhanced_df['P']
        enhanced_df['N_P2'] = enhanced_df['N'] * enhanced_df['P']**2
        enhanced_df['N2_K'] = enhanced_df['N']**2 * enhanced_df['K']
        enhanced_df['N_K2'] = enhanced_df['N'] * enhanced_df['K']**2
        enhanced_df['P2_K'] = enhanced_df['P']**2 * enhanced_df['K']
        enhanced_df['P_K2'] = enhanced_df['P'] * enhanced_df['K']**2
        
        # Distance metrics
        enhanced_df['euclidean_norm'] = np.sqrt(enhanced_df['N']**2 + enhanced_df['P']**2 + enhanced_df['K']**2)
        enhanced_df['manhattan_norm'] = enhanced_df['N'] + enhanced_df['P'] + enhanced_df['K']
        
        # NPK ranges/categories
        enhanced_df['N_range'] = pd.cut(enhanced_df['N'], bins=[0, 50, 100, 150, 300], labels=[0, 1, 2, 3]).fillna(1).astype(int)
        enhanced_df['P_range'] = pd.cut(enhanced_df['P'], bins=[0, 25, 50, 75, 150], labels=[0, 1, 2, 3]).fillna(1).astype(int)
        enhanced_df['K_range'] = pd.cut(enhanced_df['K'], bins=[0, 50, 100, 150, 300], labels=[0, 1, 2, 3]).fillna(1).astype(int)
        
        # Environmental interactions
        enhanced_df['temp_humidity'] = enhanced_df['temperature'] * enhanced_df['humidity']
        enhanced_df['temp_ph'] = enhanced_df['temperature'] * enhanced_df['ph']
        enhanced_df['humidity_ph'] = enhanced_df['humidity'] * enhanced_df['ph']
        enhanced_df['temp_humidity_ph'] = enhanced_df['temperature'] * enhanced_df['humidity'] * enhanced_df['ph']
        
        # NPK-Environmental interactions
        enhanced_df['N_temp'] = enhanced_df['N'] * enhanced_df['temperature']
        enhanced_df['P_temp'] = enhanced_df['P'] * enhanced_df['temperature']
        enhanced_df['K_temp'] = enhanced_df['K'] * enhanced_df['temperature']
        enhanced_df['N_humidity'] = enhanced_df['N'] * enhanced_df['humidity']
        enhanced_df['P_humidity'] = enhanced_df['P'] * enhanced_df['humidity']
        enhanced_df['K_humidity'] = enhanced_df['K'] * enhanced_df['humidity']
        enhanced_df['N_ph'] = enhanced_df['N'] * enhanced_df['ph']
        enhanced_df['P_ph'] = enhanced_df['P'] * enhanced_df['ph']
        enhanced_df['K_ph'] = enhanced_df['K'] * enhanced_df['ph']
        
        # Environmental ratios
        enhanced_df['temp_humidity_ratio'] = enhanced_df['temperature'] / (enhanced_df['humidity'] + 0.01)
        enhanced_df['humidity_temp_ratio'] = enhanced_df['humidity'] / (enhanced_df['temperature'] + 0.01)
        enhanced_df['ph_temp_ratio'] = enhanced_df['ph'] / (enhanced_df['temperature'] + 0.01)
        enhanced_df['temp_ph_ratio'] = enhanced_df['temperature'] / (enhanced_df['ph'] + 0.01)
        
        # Environmental ranges/categories
        enhanced_df['temp_range'] = pd.cut(enhanced_df['temperature'], bins=[0, 15, 25, 35, 50], labels=[0, 1, 2, 3]).fillna(1).astype(int)
        enhanced_df['humidity_range'] = pd.cut(enhanced_df['humidity'], bins=[0, 40, 60, 80, 100], labels=[0, 1, 2, 3]).fillna(1).astype(int)
        enhanced_df['ph_range'] = pd.cut(enhanced_df['ph'], bins=[0, 5.5, 6.5, 7.5, 14], labels=[0, 1, 2, 3]).fillna(1).astype(int)
        
        return enhanced_df
    
    def _train_model(self):
        try:
            # Load data
            df = pd.read_csv(self.dataset_path)
            print(f"Original dataset shape: {df.shape}")
            
            # Create features from NPK data only
            enhanced_df = self._create_features(df)
            print(f"Enhanced dataset shape: {enhanced_df.shape}")
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(enhanced_df['label'])
            
            # Prepare features (exclude label)
            feature_cols = [col for col in enhanced_df.columns if col != 'label']
            X = enhanced_df[feature_cols]
            
            print(f"Features used: {len(feature_cols)}")
            print(f"Base features: N, P, K, Temperature, Humidity, pH (NPK + Environmental sensors)")
            print(f"Total engineered features: {len(feature_cols)}")
            
            # Feature selection
            selector = SelectKBest(score_func=mutual_info_classif, k=min(50, len(feature_cols)))
            X_selected = selector.fit_transform(X, y_encoded)
            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
            
            print(f"Selected {len(selected_features)} best features")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            print(f"Training samples: {X_train.shape[0]}")
            print(f"Testing samples: {X_test.shape[0]}")
            
            # Define optimized models for NPK + Environmental data
            models = {
                'RandomForest': RandomForestClassifier(
                    n_estimators=1000,
                    max_depth=25,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                ),
                'XGBoost': xgb.XGBClassifier(
                    n_estimators=1000,
                    max_depth=12,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    eval_metric='mlogloss'
                ),
                'LightGBM': lgb.LGBMClassifier(
                    n_estimators=1000,
                    max_depth=12,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    verbose=-1
                ),
                'ExtraTrees': ExtraTreesClassifier(
                    n_estimators=1000,
                    max_depth=25,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
            }
            
            # Train and evaluate individual models
            model_scores = {}
            trained_models = {}
            
            print("\\n=== TRAINING INDIVIDUAL MODELS ===")
            for name, model in models.items():
                print(f"Training {name}...")
                try:
                    model.fit(X_train_scaled, y_train)
                    train_score = model.score(X_train_scaled, y_train)
                    test_score = model.score(X_test_scaled, y_test)
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                    
                    model_scores[name] = {
                        'train': train_score,
                        'test': test_score,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    trained_models[name] = model
                    
                    print(f"  {name}: Test={test_score:.4f} ({test_score*100:.2f}%), CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
                except Exception as e:
                    print(f"  {name}: Failed - {e}")
            
            # Select top 3 models for ensemble
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['test'], reverse=True)
            top_3_models = sorted_models[:3]
            
            print(f"\\nTop 3 models for ensemble:")
            for name, scores in top_3_models:
                print(f"  {name}: Test={scores['test']:.4f} ({scores['test']*100:.2f}%)")
            
            # Create voting ensemble
            ensemble_estimators = [(name, trained_models[name]) for name, _ in top_3_models]
            
            self.model = VotingClassifier(
                estimators=ensemble_estimators,
                voting='soft'
            )
            
            print("\\n=== TRAINING CROP RECOMMENDATION ENSEMBLE ===")
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate ensemble
            train_accuracy = self.model.score(X_train_scaled, y_train)
            test_accuracy = self.model.score(X_test_scaled, y_test)
            
            # Cross-validation
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=skf)
            
            print(f"\\n=== CROP RECOMMENDATION MODEL PERFORMANCE ===")
            print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
            print(f"Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            print(f"10-Fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"CV Range: {cv_scores.min():.4f} - {cv_scores.max():.4f}")
            print(f"Overfitting: {(train_accuracy-test_accuracy)*100:.2f}%")
            
            # Success indicators
            if test_accuracy >= 0.95:
                print("🎉 OUTSTANDING: Achieved 95%+ accuracy with NPK + Environmental data!")
            elif test_accuracy >= 0.90:
                print("🎯 EXCELLENT: Achieved 90%+ accuracy with NPK + Environmental data!")
            elif test_accuracy >= 0.85:
                print("✅ VERY GOOD: Achieved 85%+ accuracy with NPK + Environmental data!")
            elif test_accuracy >= 0.75:
                print("👍 GOOD: Achieved 75%+ accuracy with NPK + Environmental data!")
            else:
                print("📈 BASELINE: NPK + Environmental model performance")
            
            # Feature importance analysis
            if hasattr(self.model.estimators_[0], 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': selected_features,
                    'importance': self.model.estimators_[0].feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\\n=== TOP 15 MOST IMPORTANT FEATURES ===")
                for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
                    print(f"{i:2d}. {row['feature']:<20}: {row['importance']:.4f}")
            
            # Save model
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'selector': selector,
                'feature_names': feature_cols,
                'selected_features': selected_features,
                'test_accuracy': test_accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'individual_scores': model_scores,
                'ensemble_components': [name for name, _ in top_3_models]
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"\\nCrop Recommendation model saved to {self.model_path}")
            
            return test_accuracy
            
        except Exception as e:
            print(f"Error training crop recommendation model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.label_encoder = data['label_encoder']
                self.selector = data['selector']
                self.feature_names = data['feature_names']
                self.selected_features = data['selected_features']
                self.test_accuracy = data.get('test_accuracy', 0)
                self.cv_mean = data.get('cv_mean', 0)
                self.ensemble_components = data.get('ensemble_components', ['RandomForest', 'XGBoost', 'LightGBM'])
            print(f"Crop Recommendation model loaded from {self.model_path}")
            print(f"Test accuracy: {self.test_accuracy*100:.2f}%")
            print(f"CV accuracy: {self.cv_mean*100:.2f}%")
            print(f"Ensemble: {', '.join(self.ensemble_components)}")
        except Exception as e:
            print(f"Error loading model: {e}. Retraining...")
            self._train_model()
    
    def _create_features_for_prediction(self, n, p, k, temperature, humidity, ph):
        """Create the same features for prediction"""
        data = {'N': [n], 'P': [p], 'K': [k], 'temperature': [temperature], 'humidity': [humidity], 'ph': [ph]}
        df = pd.DataFrame(data)
        
        # Apply same feature engineering
        for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph']:
            df[f'{col}_sq'] = df[col] ** 2
            df[f'{col}_cube'] = df[col] ** 3
            df[f'{col}_sqrt'] = np.sqrt(df[col])
            df[f'{col}_log'] = np.log1p(df[col])
            df[f'{col}_inv'] = 1 / (df[col] + 0.01)
        
        df['N_P_ratio'] = df['N'] / (df['P'] + 0.01)
        df['N_K_ratio'] = df['N'] / (df['K'] + 0.01)
        df['P_K_ratio'] = df['P'] / (df['K'] + 0.01)
        df['P_N_ratio'] = df['P'] / (df['N'] + 0.01)
        df['K_N_ratio'] = df['K'] / (df['N'] + 0.01)
        df['K_P_ratio'] = df['K'] / (df['P'] + 0.01)
        
        df['NPK_total'] = df['N'] + df['P'] + df['K']
        df['NPK_mean'] = df['NPK_total'] / 3
        df['NPK_std'] = df[['N', 'P', 'K']].std(axis=1)
        df['NPK_var'] = df[['N', 'P', 'K']].var(axis=1)
        df['NPK_max'] = df[['N', 'P', 'K']].max(axis=1)
        df['NPK_min'] = df[['N', 'P', 'K']].min(axis=1)
        df['NPK_range'] = df['NPK_max'] - df['NPK_min']
        df['NPK_median'] = df[['N', 'P', 'K']].median(axis=1)
        
        df['N_prop'] = df['N'] / df['NPK_total']
        df['P_prop'] = df['P'] / df['NPK_total']
        df['K_prop'] = df['K'] / df['NPK_total']
        
        df['NP_product'] = df['N'] * df['P']
        df['NK_product'] = df['N'] * df['K']
        df['PK_product'] = df['P'] * df['K']
        df['NPK_product'] = df['N'] * df['P'] * df['K']
        
        df['NP_geomean'] = np.sqrt(df['N'] * df['P'])
        df['NK_geomean'] = np.sqrt(df['N'] * df['K'])
        df['PK_geomean'] = np.sqrt(df['P'] * df['K'])
        df['NPK_geomean'] = (df['N'] * df['P'] * df['K']) ** (1/3)
        
        df['NP_harmean'] = 2 / (1/(df['N']+0.01) + 1/(df['P']+0.01))
        df['NK_harmean'] = 2 / (1/(df['N']+0.01) + 1/(df['K']+0.01))
        df['PK_harmean'] = 2 / (1/(df['P']+0.01) + 1/(df['K']+0.01))
        
        df['npk_balance'] = 1 - (abs(df['N_prop'] - 0.33) + 
                                abs(df['P_prop'] - 0.33) + 
                                abs(df['K_prop'] - 0.33))
        
        dominant = df[['N', 'P', 'K']].idxmax(axis=1)
        df['N_dominant'] = (dominant == 'N').astype(int)
        df['P_dominant'] = (dominant == 'P').astype(int)
        df['K_dominant'] = (dominant == 'K').astype(int)
        
        df['N2_P'] = df['N']**2 * df['P']
        df['N_P2'] = df['N'] * df['P']**2
        df['N2_K'] = df['N']**2 * df['K']
        df['N_K2'] = df['N'] * df['K']**2
        df['P2_K'] = df['P']**2 * df['K']
        df['P_K2'] = df['P'] * df['K']**2
        
        df['euclidean_norm'] = np.sqrt(df['N']**2 + df['P']**2 + df['K']**2)
        df['manhattan_norm'] = df['N'] + df['P'] + df['K']
        
        df['N_range'] = pd.cut(df['N'], bins=[0, 50, 100, 150, 300], labels=[0, 1, 2, 3]).fillna(1).astype(int)
        df['P_range'] = pd.cut(df['P'], bins=[0, 25, 50, 75, 150], labels=[0, 1, 2, 3]).fillna(1).astype(int)
        df['K_range'] = pd.cut(df['K'], bins=[0, 50, 100, 150, 300], labels=[0, 1, 2, 3]).fillna(1).astype(int)
        
        # Environmental interactions
        df['temp_humidity'] = df['temperature'] * df['humidity']
        df['temp_ph'] = df['temperature'] * df['ph']
        df['humidity_ph'] = df['humidity'] * df['ph']
        df['temp_humidity_ph'] = df['temperature'] * df['humidity'] * df['ph']
        
        # NPK-Environmental interactions
        df['N_temp'] = df['N'] * df['temperature']
        df['P_temp'] = df['P'] * df['temperature']
        df['K_temp'] = df['K'] * df['temperature']
        df['N_humidity'] = df['N'] * df['humidity']
        df['P_humidity'] = df['P'] * df['humidity']
        df['K_humidity'] = df['K'] * df['humidity']
        df['N_ph'] = df['N'] * df['ph']
        df['P_ph'] = df['P'] * df['ph']
        df['K_ph'] = df['K'] * df['ph']
        
        # Environmental ratios
        df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 0.01)
        df['humidity_temp_ratio'] = df['humidity'] / (df['temperature'] + 0.01)
        df['ph_temp_ratio'] = df['ph'] / (df['temperature'] + 0.01)
        df['temp_ph_ratio'] = df['temperature'] / (df['ph'] + 0.01)
        
        # Environmental ranges/categories
        df['temp_range'] = pd.cut(df['temperature'], bins=[0, 15, 25, 35, 50], labels=[0, 1, 2, 3]).fillna(1).astype(int)
        df['humidity_range'] = pd.cut(df['humidity'], bins=[0, 40, 60, 80, 100], labels=[0, 1, 2, 3]).fillna(1).astype(int)
        df['ph_range'] = pd.cut(df['ph'], bins=[0, 5.5, 6.5, 7.5, 14], labels=[0, 1, 2, 3]).fillna(1).astype(int)
        
        return df
    
    def analyze_soil(self, n, p, k, temperature, humidity, ph):
        """Analyze soil with NPK + Environmental sensor data"""
        if self.model is None or self.scaler is None:
            return {'error': 'Crop Recommendation model is not loaded properly.'}
        
        try:
            # Create features for prediction
            feature_df = self._create_features_for_prediction(n, p, k, temperature, humidity, ph)
            
            # Select the same features used in training
            X_all = feature_df[self.feature_names]
            X_selected = self.selector.transform(X_all)
            
            # Scale features
            X_scaled = self.scaler.transform(X_selected)
            
            # Make prediction
            y_pred_encoded = self.model.predict(X_scaled)[0]
            crop_prediction = self.label_encoder.inverse_transform([y_pred_encoded])[0]
            
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Get top recommendations
            top_indices = np.argsort(probabilities)[-5:][::-1]
            recommended_crops = []
            
            for i in top_indices:
                crop_name = self.label_encoder.inverse_transform([i])[0]
                confidence = probabilities[i] * 100
                if confidence > 1.0:
                    recommended_crops.append({
                        "crop": crop_name, 
                        "confidence": f"{confidence:.1f}%"
                    })
            
            soil_quality = self._determine_soil_quality(n, p, k, temperature, humidity, ph)
            
            return {
                'soil_quality': soil_quality,
                'recommended_crops': recommended_crops[:5],
                'top_crop': crop_prediction,
                'model_type': f'Crop Recommendation Ensemble ({", ".join(self.ensemble_components)})',
                'test_accuracy': f"{self.test_accuracy*100:.2f}%",
                'cv_accuracy': f"{self.cv_mean*100:.2f}%",
                'input_features': 'N, P, K, Temperature, Humidity, pH (NPK + Environmental Sensors)'
            }
            
        except Exception as e:
            print(f"Error during crop recommendation analysis: {e}")
            return {'error': 'Crop recommendation analysis failed.'}
    
    def _determine_soil_quality(self, n, p, k, temperature, humidity, ph):
        """Soil quality assessment based on NPK + Environmental levels"""
        # NPK scores based on typical ranges
        n_score = max(0, min(n / 150.0, 1))
        p_score = max(0, min(p / 75.0, 1))
        k_score = max(0, min(k / 200.0, 1))
        
        # Environmental scores
        temp_score = 1 - abs(temperature - 25) / 25  # Optimal around 25°C
        temp_score = max(0, min(temp_score, 1))
        
        humidity_score = 1 - abs(humidity - 70) / 30  # Optimal around 70%
        humidity_score = max(0, min(humidity_score, 1))
        
        ph_score = 1 - abs(ph - 6.5) / 2  # Optimal around 6.5
        ph_score = max(0, min(ph_score, 1))
        
        # Balance score
        total_npk = n + p + k
        if total_npk > 0:
            n_prop = n / total_npk
            p_prop = p / total_npk
            k_prop = k / total_npk
            balance_score = 1 - (abs(n_prop - 0.33) + abs(p_prop - 0.33) + abs(k_prop - 0.33))
            balance_score = max(0, balance_score)
        else:
            balance_score = 0
        
        # Combined score (weighted)
        nutrient_score = (n_score + p_score + k_score) / 3
        environmental_score = (temp_score + humidity_score + ph_score) / 3
        final_score = nutrient_score * 0.4 + environmental_score * 0.4 + balance_score * 0.2
        
        if final_score >= 0.85: return "Excellent"
        elif final_score >= 0.70: return "Good"
        elif final_score >= 0.50: return "Average"
        else: return "Poor"

if __name__ == "__main__":
    print("Testing Crop Recommendation Model...")
    analyzer = CropRecommendationModel()
    
    # Test with sample NPK + Environmental values
    test_cases = [
        (90, 42, 43, 20.9, 82.0, 6.5),    # Rice-like conditions
        (150, 60, 70, 25.0, 75.0, 7.0),   # Wheat-like conditions  
        (180, 100, 220, 28.0, 65.0, 6.8)  # Tomato-like conditions
    ]
    
    for i, (n, p, k, temp, hum, ph) in enumerate(test_cases, 1):
        result = analyzer.analyze_soil(n, p, k, temp, hum, ph)
        print(f"\\nTest case {i} (N={n}, P={p}, K={k}, T={temp}°C, H={hum}%, pH={ph}):")
        print(f"  Top crop: {result.get('top_crop', 'N/A')}")
        print(f"  Test accuracy: {result.get('test_accuracy', 'N/A')}")
        print(f"  CV accuracy: {result.get('cv_accuracy', 'N/A')}")
        print(f"  Soil quality: {result.get('soil_quality', 'N/A')}")
        print(f"  Top 3 recommendations: {result.get('recommended_crops', [])[:3]}")