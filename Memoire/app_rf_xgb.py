import streamlit as st
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import ARIMA
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
import json

class TimeSeriesPredictor:
    def __init__(self):
        try:
            # Chargement du dataset ARIMA
            self.df = pd.read_csv(
                './../Dataset/df_colab_travail_final_engineered.csv',
                parse_dates=['DATE']
            )
            # Chargement des modèles RF et XGBoost
            self.rf_model = joblib.load('./modeles/best_rf.pkl')
            
            # Chargement du modèle XGBoost
            self.xgb_model = XGBRegressor()
            self.xgb_model.load_model('./modeles/best_xgb.json')
            
            # Chargement du dataset pour RF et XGBoost
            self.rf_xgb_data = pd.read_csv('./../Dataset/df_travail_final_random_search.csv_engineered.csv')
            self.rf_xgb_data.drop('Unnamed: 0', axis=1, inplace=True)
            
        except FileNotFoundError as e:
            st.error(f"❌ Erreur de chargement des fichiers: {str(e)}")
            st.stop()
            
        self.target_column = 'RECETTES_BUDGETAIRES'
        self.df[self.target_column] = self.df[self.target_column].astype(float)

    def predict_rf(self, X):
        """Prédiction avec Random Forest"""
        return self.rf_model.predict(X)
    
    def predict_xgb(self, X):
        """Prédiction avec XGBoost"""
        return self.xgb_model.predict(X)
        
    def prepare_series(self):
        # Ajout de la validation des données
        if self.df[self.target_column].isnull().any():
            self.df[self.target_column].fillna(method='ffill', inplace=True)
            st.warning("⚠️ Des valeurs manquantes ont été détectées et comblées.")
            
        return TimeSeries.from_dataframe(
            self.df, 
            time_col='DATE', 
            value_cols=self.target_column
        )
    
    def train_model(self, p=2, d=1, q=2, seasonal_p=2, seasonal_d=1, seasonal_q=2, m=6):
        series = self.prepare_series()
        train_series = series[:-31]
        
        model = ARIMA(
            p=p, d=d, q=q, 
            seasonal_order=(seasonal_p, seasonal_d, seasonal_q, m), 
            random_state=42
        )
        
        with st.spinner('🔄 Entraînement du modèle en cours...'):
            model.fit(train_series)
        
        return model, train_series, series
    
    def evaluate_model(self, model, train_series, full_series):
        test_series = full_series[-31:]
        forecast = model.predict(len(test_series))
        
        mae = mean_absolute_error(test_series.values(), forecast.values())
        mse = mean_squared_error(test_series.values(), forecast.values())
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((test_series.values() - forecast.values()) / test_series.values())) * 100
        
        return {
            'MAE': {'value': mae, 'format': f"{mae:,.2f}", 'description': "Erreur absolue moyenne"},
            'MSE': {'value': mse, 'format': f"{mse:,.2f}", 'description': "Erreur quadratique moyenne"},
            'RMSE': {'value': rmse, 'format': f"{rmse:,.2f}", 'description': "Racine de l'erreur quadratique moyenne"},
            'MAPE': {'value': mape, 'format': f"{mape:.2f}%", 'description': "Pourcentage moyen d'erreur absolue"}
        }

def create_forecast_plot(historical_series, forecast_series, test_series=None, confidence_interval=None):
    fig = go.Figure()
    
    # Données historiques
    fig.add_trace(go.Scatter(
        x=historical_series.time_index, 
        y=historical_series.values().flatten(),
        mode='lines',
        name='Données historiques',
        line=dict(color='#2E86C1', width=2)
    ))
    
    # Intervalle de confiance
    if confidence_interval is not None:
        lower_bound, upper_bound = confidence_interval
        fig.add_trace(go.Scatter(
            x=forecast_series.time_index,
            y=upper_bound,
            mode='lines',
            name='Intervalle de confiance (95%)',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_series.time_index,
            y=lower_bound,
            mode='lines',
            name='Intervalle de confiance (95%)',
            fill='tonexty',
            fillcolor='rgba(231, 76, 60, 0.2)',
            line=dict(width=0),
            showlegend=True
        ))
    
    # Données de test
    if test_series is not None:
        fig.add_trace(go.Scatter(
            x=test_series.time_index,
            y=test_series.values().flatten(),
            mode='lines',
            name='Données réelles',
            line=dict(color='#27AE60', width=2)
        ))
    
    # Prédictions
    fig.add_trace(go.Scatter(
        x=forecast_series.time_index,
        y=forecast_series.values().flatten(),
        mode='lines',
        name='Prédictions',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title={
            'text': 'Prédiction des Recettes Budgétaires',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis_title='Date',
        yaxis_title='Recettes (en milliard FCFA)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600
    )
    
    # Personnalisation du hover
    fig.update_traces(
        hovertemplate="<b>Date</b>: %{x}<br><b>Valeur</b>: %{y:,.2f}<br>"
    )
    
    return fig

def calculate_confidence_interval(forecast_samples, confidence=0.95):
    lower_quantile = (1 - confidence) / 2
    upper_quantile = 1 - lower_quantile
    
    lower_bound = np.quantile(forecast_samples.values(), lower_quantile, axis=1)
    upper_bound = np.quantile(forecast_samples.values(), upper_quantile, axis=1)
    
    return lower_bound, upper_bound

def create_seasonal_plot(series):
    # Création d'un DataFrame avec les données mensuelles
    df = pd.DataFrame({
        'date': series.time_index,
        'value': series.values().flatten()
    })
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    seasonal_fig = go.Figure()
    
    for year in df['year'].unique():
        year_data = df[df['year'] == year]
        seasonal_fig.add_trace(go.Scatter(
            x=year_data['month'],
            y=year_data['value'],
            mode='lines+markers',
            name=str(year),
            hovertemplate="<b>Année</b>: " + str(year) +
                         "<br><b>Mois</b>: %{x}" +
                         "<br><b>Valeur</b>: %{y:,.2f}<br>"
        ))
    
    seasonal_fig.update_layout(
        title="Saisonnalité des Recettes Budgétaires",
        xaxis_title="Mois",
        yaxis_title="Recettes (en milliard FCFA)",
        template="plotly_white",
        height=400,
        showlegend=True,
        legend_title="Année"
    )
    
    return seasonal_fig

def create_forecast_plot_rf(historical_series, forecast_series, test_series=None, confidence_interval=None):
    fig = go.Figure()

    # Données historiques
    fig.add_trace(go.Scatter(
        x=historical_series.index,  # Assurez-vous que historical_series a un index datetime
        y=historical_series.values,
        mode='lines',
        name='Données historiques',
        line=dict(color='#2E86C1', width=2)
    ))

    # Intervalle de confiance (si disponible)
    if confidence_interval is not None:
        lower_bound, upper_bound = confidence_interval
        fig.add_trace(go.Scatter(
            x=forecast_series.index,
            y=upper_bound,
            mode='lines',
            name='Intervalle de confiance (95%)',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_series.index,
            y=lower_bound,
            mode='lines',
            name='Intervalle de confiance (95%)',
            fill='tonexty',
            fillcolor='rgba(231, 76, 60, 0.2)',
            line=dict(width=0),
            showlegend=True
        ))

    # Données de test
    if test_series is not None:
        fig.add_trace(go.Scatter(
            x=test_series.index,
            y=test_series.values,
            mode='lines',
            name='Données réelles',
            line=dict(color='#27AE60', width=2)
        ))

    # Prédictions
    fig.add_trace(go.Scatter(
        x=forecast_series.index,
        y=forecast_series.values,
        mode='lines',
        name='Prédictions',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))

    fig.update_layout(
        title='Prédiction avec Random Forest',
        xaxis_title='Date',
        yaxis_title='Recettes (en milliard FCFA)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600
    )

    fig.update_traces(hovertemplate="<b>Date</b>: %{x}<br><b>Valeur</b>: %{y:,.2f}<br>")

    return fig

def create_forecast_plot_xgb(historical_series, forecast_series, test_series=None, confidence_interval=None):
    fig = go.Figure()

    # Données historiques
    fig.add_trace(go.Scatter(
        x=historical_series.index,
        y=historical_series.values,
        mode='lines',
        name='Données historiques',
        line=dict(color='#2E86C1', width=2)
    ))

    # Intervalle de confiance (si disponible)
    if confidence_interval is not None:
        lower_bound, upper_bound = confidence_interval
        fig.add_trace(go.Scatter(
            x=forecast_series.index,
            y=upper_bound,
            mode='lines',
            name='Intervalle de confiance (95%)',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_series.index,
            y=lower_bound,
            mode='lines',
            name='Intervalle de confiance (95%)',
            fill='tonexty',
            fillcolor='rgba(231, 76, 60, 0.2)',
            line=dict(width=0),
            showlegend=True
        ))

    # Données de test
    if test_series is not None:
        fig.add_trace(go.Scatter(
            x=test_series.index,
            y=test_series.values,
            mode='lines',
            name='Données réelles',
            line=dict(color='#27AE60', width=2)
        ))

    # Prédictions
    fig.add_trace(go.Scatter(
        x=forecast_series.index,
        y=forecast_series.values,
        mode='lines',
        name='Prédictions',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))

    fig.update_layout(
        title='Prédiction avec XGBoost',
        xaxis_title='Date',
        yaxis_title='Recettes (en milliard FCFA)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600
    )

    fig.update_traces(hovertemplate="<b>Date</b>: %{x}<br><b>Valeur</b>: %{y:,.2f}<br>")

    return fig

def main():
    # Configuration de la page
    st.set_page_config(
        page_title="Prédiction des Recettes Budgétaires",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Style CSS personnalisé
    st.markdown("""
        <style>
        .big-font {
            font-size: 36px !important;
            font-weight: bold;
            margin-bottom: 30px;
            color: #1E88E5;
        }
        .metric-card {
            /*background-color: #0E1117;*/
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.2s ease-in-out;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #1E88E5;
        }
        .metric-description {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        .stTabs {
            /*background-color: #0E1117;*/
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # En-tête
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown('<p class="big-font">📈 Prédiction des Recettes Budgétaires</p>', unsafe_allow_html=True)
    with col2:
        st.button('❓ Aide', help="Guide d'utilisation de l'application")
    
    # Sidebar avec paramètres avancés
    with st.sidebar:
        st.markdown("### 🤖 Sélection du Modèle")
        model_choice = st.selectbox(
            'Choisir le modèle',
            ['SARIMA', 'Random Forest', 'XGBoost']
        )
        st.markdown("### ⚙️ Paramètres de Prédiction")
        
        if model_choice == 'SARIMA':
            forecast_horizon = st.slider(
                'Horizon de Prédiction (mois)', 
                min_value=1, 
                max_value=43, 
                value=31
            )
        if model_choice == 'SARIMA':                        
            confidence_interval = st.checkbox(
                'Afficher l\'intervalle de confiance', 
                value=False
            )
            
            with st.expander("🔧 Paramètres Avancés du Modèle"):
                col1, col2 = st.columns(2)
                with col1:
                    p = st.number_input('AR (p)', min_value=0, max_value=5, value=2)
                    d = st.number_input('Différenciation (d)', min_value=0, max_value=2, value=1)
                    q = st.number_input('MA (q)', min_value=0, max_value=5, value=2)
                with col2:
                    seasonal_p = st.number_input('AR Saisonnier', min_value=0, max_value=5, value=2)
                    seasonal_d = st.number_input('Différenciation Saisonnière', min_value=0, max_value=2, value=1)
                    seasonal_q = st.number_input('MA Saisonnier', min_value=0, max_value=5, value=2)
                
                m = st.number_input('Période Saisonnière', min_value=1, max_value=12, value=6)
            
        st.markdown("---")
        st.markdown("### 📊 Informations")
        st.info(f"""
        Cette application utilise un modèle {model_choice} pour prédire 
        les recettes budgétaires futures. Elle permet de:
        - Visualiser les tendances historiques
        - Générer des prédictions
        - Analyser les performances du modèle
        - Exporter les résultats
        """)
    
    try:
        predictor = TimeSeriesPredictor()
        
        if model_choice == 'SARIMA':
            model, train_series, full_series = predictor.train_model(
                p=p, d=d, q=q,
                seasonal_p=seasonal_p,
                seasonal_d=seasonal_d,
                seasonal_q=seasonal_q,
                m=m
            )
            
            # Métriques de performance
            metrics = predictor.evaluate_model(model, train_series, full_series)
            cols = st.columns(4)
            for col, (metric_name, metric_data) in zip(cols, metrics.items()):
                with col:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>{metric_name}</h3>
                            <div class="metric-value">{metric_data['format']}</div>
                            <div class="metric-description">{metric_data['description']}</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Onglets principaux
            tab1, tab2, tab3 = st.tabs([
                '📊 Prédictions', 
                '📈 Analyse des Performances',
                '🔄 Analyse Saisonnière'
            ])
            
            with tab1:
                # Génération des prédictions
                forecast_samples = model.predict(
                    forecast_horizon,
                    num_samples=1000 if confidence_interval else 1
                )
                
                # Calcul de l'intervalle de confiance
                ci_data = None
                if confidence_interval:
                    ci_data = calculate_confidence_interval(forecast_samples)
                
                # Création et affichage du graphique
                fig = create_forecast_plot(
                    train_series, 
                    forecast_samples, 
                    full_series[-31:],
                    ci_data
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Tableau des prédictions
                st.markdown("### 📋 Valeurs Prédites")
                
                # Préparation des données pour l'affichage
                forecast_df = forecast_samples.pd_dataframe()
                forecast_df.rename(columns={'RECETTES_BUDGETAIRES': 'Valeur Prédite'}, inplace=True)
                # Supposons que full_series est un objet TimeSeries de DARTS
                real_values = full_series[-forecast_horizon:].values()  # Extraire les dernières valeurs réelles (de la taille de forecast_horizon)

                # Assurez-vous que forecast_df a un index de type datetime (si ce n'est pas déjà le cas)
                forecast_df.index = pd.to_datetime(forecast_df.index)  # Convertir l'index en DateTime si nécessaire

                # Ajouter la colonne 'Valeur Réelle' au DataFrame des prévisions
                forecast_df['Valeur Réelle'] = real_values.flatten()  # flatten() pour obtenir un vecteur 1D si nécessaire
                forecast_df.index = forecast_df.index.strftime('%Y-%m-%d')
                
                if confidence_interval:
                    forecast_df['Borne Inférieure'] = ci_data[0]
                    forecast_df['Borne Supérieure'] = ci_data[1]
                
                # Formater les valeurs pour l'affichage
                styled_df = forecast_df.style.format("{:,.2f}")
                
                # Affichage avec pagination
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=400
                )
                
                # Options d'export
                col1, col2 = st.columns([1, 6])
                with col1:
                    export_format = st.selectbox(
                        "Format d'export",
                        ['CSV', 'Excel']
                    )
                
                if export_format == 'CSV':
                    st.download_button(
                        label="📥 Télécharger les prédictions (CSV)",
                        data=forecast_df.to_csv(),
                        file_name=f"predictions_recettes_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    # Créer un buffer en mémoire
                    buffer = io.BytesIO()
                    
                    # Sauvegarder le DataFrame dans le buffer
                    forecast_df.to_excel(buffer, index=False)
                    
                    # Réinitialiser le pointeur du buffer
                    buffer.seek(0)
                    
                    # Créer le bouton de téléchargement
                    st.download_button(
                        label="📥 Télécharger les prédictions (Excel)",
                        data=buffer,
                        file_name=f"predictions_recettes_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.ms-excel"
                    )
            
            with tab2:
                # Analyse des performances du modèle
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Distribution des Erreurs")
                    residuals = (full_series[-31:].values() - 
                                model.predict(31).values()).flatten()
                    
                    fig_residuals = px.histogram(
                        residuals,
                        title='Distribution des Erreurs de Prédiction',
                        labels={'value': 'Erreur', 'count': 'Fréquence'},
                        template='plotly_white',
                        color_discrete_sequence=['#1E88E5']
                    )
                    fig_residuals.update_layout(
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig_residuals, use_container_width=True)
                
                with col2:
                    st.markdown("### 📈 Erreurs dans le Temps")
                    fig_error_time = go.Figure()
                    
                    test_dates = full_series[-31:].time_index
                    fig_error_time.add_trace(go.Scatter(
                        x=test_dates,
                        y=residuals,
                        mode='lines+markers',
                        name='Erreurs',
                        line=dict(color='#E74C3C')
                    ))
                    
                    fig_error_time.add_hline(
                        y=0, 
                        line_dash="dash", 
                        line_color="gray",
                        annotation_text="Erreur nulle"
                    )
                    
                    fig_error_time.update_layout(
                        title="Évolution des Erreurs dans le Temps",
                        xaxis_title="Date",
                        yaxis_title="Erreur",
                        height=400,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_error_time, use_container_width=True)
                
                # Statistiques détaillées des erreurs
                st.markdown("### 📊 Statistiques Détaillées des Erreurs")
                error_stats = pd.DataFrame({
                    'Statistique': [
                        'Moyenne', 'Médiane', 'Écart-type', 
                        'Minimum', 'Maximum', 'Skewness', 'Kurtosis'
                    ],
                    'Valeur': [
                        f"{np.mean(residuals):,.2f}",
                        f"{np.median(residuals):,.2f}",
                        f"{np.std(residuals):,.2f}",
                        f"{np.min(residuals):,.2f}",
                        f"{np.max(residuals):,.2f}",
                        f"{pd.Series(residuals).skew():,.2f}",
                        f"{pd.Series(residuals).kurtosis():,.2f}"
                    ],
                    'Description': [
                        'Tendance centrale des erreurs',
                        'Valeur centrale des erreurs',
                        'Dispersion des erreurs',
                        'Plus petite erreur observée',
                        'Plus grande erreur observée',
                        'Asymétrie de la distribution',
                        'Forme de la distribution'
                    ]
                })
                
                st.dataframe(
                    error_stats.style.set_properties(**{
                        #'background-color': '#0E1117',
                        'font-size': '14px'
                    }),
                    use_container_width=True,
                    height=300
                )
            
            with tab3:
                st.markdown("### 🔄 Analyse de la Saisonnalité")
                
                # Graphique de saisonnalité
                seasonal_fig = create_seasonal_plot(full_series)
                st.plotly_chart(seasonal_fig, use_container_width=True)
                
                # Statistiques mensuelles
                st.markdown("### 📊 Statistiques Mensuelles")
                
                monthly_stats = pd.DataFrame({
                    'date': full_series.time_index,
                    'value': full_series.values().flatten()
                })
                monthly_stats['month'] = monthly_stats['date'].dt.month
                monthly_stats['month_name'] = monthly_stats['date'].dt.strftime('%B')
                
                stats_by_month = monthly_stats.groupby('month').agg({
                    'value': ['mean', 'std', 'min', 'max', 'count']
                }).round(2)
                
                stats_by_month.columns = [
                    'Moyenne', 'Écart-type', 'Minimum', 'Maximum', 'Nombre d\'observations'
                ]
                stats_by_month.index = [datetime(2000, m, 1).strftime('%B') for m in range(1, 13)]
                
                st.dataframe(
                    stats_by_month.style.format({
                        'Moyenne': '{:,.2f}',
                        'Écart-type': '{:,.2f}',
                        'Minimum': '{:,.2f}',
                        'Maximum': '{:,.2f}'
                    }),
                    use_container_width=True,
                    height=400
                )
        else:
            # Préparation des données pour RF et XGBoost
            X = predictor.rf_xgb_data.drop(columns=['RECETTES_BUDGETAIRES']).set_index("DATE")
            y = predictor.rf_xgb_data['RECETTES_BUDGETAIRES']
            y.index = predictor.rf_xgb_data['DATE']  # Assigner l'index 'DATE' à y
            forecast_horizon = 31
            
            # Séparation des données : tout sauf les 31 dernières valeurs pour l'entraînement
            X_train = X[:-forecast_horizon]
            y_train = y[:-forecast_horizon]
            
            # 31 dernières valeurs pour le test/évaluation
            X_test = X[-forecast_horizon:]
            y_test = y[-forecast_horizon:]
            
            # Prédictions selon le modèle choisi
            if model_choice == 'Random Forest':
                predictions = predictor.predict_rf(X_test)
                predictions = predictions[-forecast_horizon:]
                model_name = 'Random Forest'
            else:  # XGBoost
                predictions = predictor.predict_xgb(X_test)
                predictions = predictions[-forecast_horizon:]
                model_name = 'XGBoost'

            # Calcul des métriques
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_test - predictions) / y)) * 100
            
            metrics = {
                'MAE': {'value': mae, 'format': f"{mae:,.2f}", 'description': "Erreur absolue moyenne"},
                'MSE': {'value': mse, 'format': f"{mse:,.2f}", 'description': "Erreur quadratique moyenne"},
                'RMSE': {'value': rmse, 'format': f"{rmse:,.2f}", 'description': "Racine de l'erreur quadratique moyenne"},
                'MAPE': {'value': mape, 'format': f"{mape:.2f}%", 'description': "Pourcentage moyen d'erreur absolue"}
            }
            
            # Affichage des métriques
            cols = st.columns(4)
            for col, (metric_name, metric_data) in zip(cols, metrics.items()):
                with col:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>{metric_name}</h3>
                            <div class="metric-value">{metric_data['format']}</div>
                            <div class="metric-description">{metric_data['description']}</div>
                        </div>
                    """, unsafe_allow_html=True)

            # Convertir les séries de test et prédictions en séries pandas pour le graphique
            forecast_series = pd.Series(predictions, index=y_test.index)  # Prédictions
            historical_series = y[:-forecast_horizon]  # Données historiques
            test_series = y_test  # Données réelles du test
            
            # Si un intervalle de confiance est nécessaire (par exemple pour XGBoost), l'ajouter
            confidence_interval = None  # Remplissez si vous avez l'intervalle de confiance (ex. pour XGBoost)
            # Exemple d'intervalle de confiance : confidence_interval = (lower_bound, upper_bound)

            # Créer et afficher le graphique pour Random Forest ou XGBoost
            if model_choice == 'Random Forest':
                forecast_fig_rf = create_forecast_plot_rf(historical_series, forecast_series, test_series, confidence_interval)
                st.plotly_chart(forecast_fig_rf)

            elif model_choice == 'XGBoost':
                forecast_fig_xgb = create_forecast_plot_xgb(historical_series, forecast_series, test_series, confidence_interval)
                st.plotly_chart(forecast_fig_xgb)

            # Création du DataFrame des prédictions
            results_df = pd.DataFrame({
                'Date': y_test.index,
                'Valeurs Réelles': y_test,
                'Prédictions': predictions
            })
            
            results_df.drop('Date', axis=1, inplace=True)
            
            # # Création du graphique
            # fig = go.Figure()
            
            # # Données réelles
            # fig.add_trace(go.Scatter(
            #     x=results_df['Date'],
            #     y=results_df['Valeurs Réelles'],
            #     mode='lines',
            #     name='Valeurs Réelles',
            #     line=dict(color='#2E86C1', width=2)
            # ))
            
            # # Prédictions
            # fig.add_trace(go.Scatter(
            #     x=results_df['Date'],
            #     y=results_df['Prédictions'],
            #     mode='lines',
            #     name='Prédictions',
            #     line=dict(color='#E74C3C', width=2, dash='dash')
            # ))
            
            # fig.update_layout(
            #     title=f"Prédictions avec {model_name}",
            #     xaxis_title='Date',
            #     yaxis_title='Recettes (en milliard FCFA)',
            #     template='plotly_white',
            #     height=600
            # )
            
            # st.plotly_chart(fig, use_container_width=True)
            
            # Affichage des résultats détaillés
            st.markdown(f"### 📋 Résultats Détaillés - {model_name}")
            st.dataframe(
                results_df.style.format({
                    'Valeurs Réelles': '{:,.2f}',
                    'Prédictions': '{:,.2f}'
                }),
                use_container_width=True,
                height=400
            )
            
            # Option d'export
            buffer = io.BytesIO()
            results_df.to_excel(buffer, index=False)
            buffer.seek(0)
            
            st.download_button(
                label=f"📥 Télécharger les résultats ({model_name})",
                data=buffer,
                file_name=f"predictions_{model_name.lower()}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.ms-excel"
            )

    except Exception as e:
        st.error(f"❌ Une erreur s'est produite : {str(e)}")
        st.markdown("""
            ### 🔍 Suggestions de résolution :
            1. Vérifiez le chemin et le format du fichier de données
            2. Assurez-vous que les colonnes requises sont présentes
            3. Vérifiez que les données sont au bon format
            4. Si l'erreur persiste, contactez le support technique
        """)

if __name__ == "__main__":
    main()