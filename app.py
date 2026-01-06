import gradio as gr
import pandas as pd
import numpy as np
import json
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go

# ============================================
# LOAD MODEL AND CONFIG (Using JSON - Compatible!)
# ============================================

# Load XGBoost model (native format)
model = xgb.XGBRegressor()
model.load_model('model.json')

# Load scaler parameters
with open('scaler_params.json', 'r') as f:
    scaler_params = json.load(f)

# Load feature names
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

# Load feature importance
with open('feature_importance.json', 'r') as f:
    feature_importance = json.load(f)

# Load app config
with open('app_config.json', 'r') as f:
    config = json.load(f)

# Load model info
with open('model_info.json', 'r') as f:
    model_info = json.load(f)

# Load data for visualizations
try:
    df_ames = pd.read_csv('AmesHousing_clean.csv')
    HAS_DATA = True
except:
    HAS_DATA = False

# Get dropdown options
neighborhoods = config['dropdown_options']['neighborhoods']
house_styles = config['dropdown_options']['house_styles']
exteriors = config['dropdown_options']['exterior']
NUMERIC_FEATURES = config['numeric_features']


# ============================================
# CUSTOM SCALER (to avoid joblib compatibility issues)
# ============================================

def scale_features(data, scaler_params):
    """Scale features using saved scaler parameters"""
    scaled_data = data.copy()
    mean = np.array(scaler_params['mean'])
    scale = np.array(scaler_params['scale'])
    feature_names = scaler_params['feature_names']
    
    for i, col in enumerate(feature_names):
        if col in scaled_data.columns:
            scaled_data[col] = (scaled_data[col] - mean[i]) / scale[i]
    
    return scaled_data


# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def create_price_gauge(predicted_price, min_price=50000, max_price=500000):
    """Create a gauge chart showing the predicted price"""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_price,
        number={'prefix': "$", 'valueformat': ',.0f'},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Predicted Price", 'font': {'size': 24}},
        delta={'reference': 180000, 'prefix': "$", 'valueformat': ',.0f'},
        gauge={
            'axis': {'range': [min_price, max_price], 'tickwidth': 1, 'tickprefix': '$', 'tickformat': ',.0f'},
            'bar': {'color': "#2563eb"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_price, 150000], 'color': '#dcfce7'},
                {'range': [150000, 250000], 'color': '#fef9c3'},
                {'range': [250000, 350000], 'color': '#fed7aa'},
                {'range': [350000, max_price], 'color': '#fecaca'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': predicted_price
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_feature_importance_chart():
    """Create horizontal bar chart of feature importance"""
    
    # Sort by importance
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=False))
    
    # Get top 15
    top_features = list(sorted_importance.keys())[-15:]
    top_values = [sorted_importance[f] for f in top_features]
    
    # Clean names
    clean_names = [f.replace('_', ' ').replace('Neighborhood ', '📍 ').replace('House Style ', '🏠 ') for f in top_features]
    
    fig = go.Figure(go.Bar(
        x=top_values,
        y=clean_names,
        orientation='h',
        marker_color='steelblue'
    ))
    
    fig.update_layout(
        title='🎯 Top 15 Features Affecting Price',
        height=500,
        xaxis_title='Importance Score',
        yaxis_title='',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_neighborhood_price_chart():
    """Create bar chart showing average price by neighborhood"""
    
    if not HAS_DATA:
        return None
    
    neighborhood_avg = df_ames.groupby('Neighborhood')['SalePrice'].mean().sort_values()
    
    fig = go.Figure(go.Bar(
        x=neighborhood_avg.values,
        y=neighborhood_avg.index,
        orientation='h',
        marker_color='viridis',
        marker=dict(color=neighborhood_avg.values, colorscale='Viridis')
    ))
    
    fig.update_layout(
        title='🏘️ Average Price by Neighborhood',
        height=600,
        xaxis_title='Average Price ($)',
        xaxis=dict(tickformat='$,.0f'),
        yaxis_title='',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_price_vs_area_scatter(highlight_area=None, highlight_price=None):
    """Create scatter plot of price vs living area"""
    
    if not HAS_DATA:
        return None
    
    fig = px.scatter(
        df_ames,
        x='Gr Liv Area',
        y='SalePrice',
        color='Overall Qual',
        title='📈 Price vs Living Area (colored by Quality)',
        opacity=0.6,
        color_continuous_scale='RdYlGn'
    )
    
    if highlight_area and highlight_price:
        fig.add_trace(go.Scatter(
            x=[highlight_area],
            y=[highlight_price],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
            name='Your Property'
        ))
    
    fig.update_layout(
        height=450,
        xaxis_title='Living Area (sq ft)',
        yaxis_title='Sale Price ($)',
        yaxis=dict(tickformat='$,.0f'),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_price_distribution(highlight_price=None):
    """Create histogram of price distribution"""
    
    if not HAS_DATA:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df_ames['SalePrice'],
        nbinsx=50,
        name='All Houses',
        marker_color='rgba(37, 99, 235, 0.6)'
    ))
    
    if highlight_price:
        fig.add_vline(
            x=highlight_price,
            line_dash="dash",
            line_color="red",
            line_width=3,
            annotation_text=f"Your Price: ${highlight_price:,.0f}",
            annotation_position="top"
        )
    
    mean_price = df_ames['SalePrice'].mean()
    fig.add_vline(
        x=mean_price,
        line_dash="dot",
        line_color="green",
        line_width=2,
        annotation_text=f"Average: ${mean_price:,.0f}",
        annotation_position="bottom"
    )
    
    fig.update_layout(
        title='📊 Distribution of House Prices',
        xaxis_title='Sale Price ($)',
        yaxis_title='Count',
        xaxis=dict(tickformat='$,.0f'),
        height=350,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_quality_boxplot():
    """Create box plot of prices by quality rating"""
    
    if not HAS_DATA:
        return None
    
    fig = px.box(
        df_ames,
        x='Overall Qual',
        y='SalePrice',
        title='📦 Price Range by Quality Rating',
        color='Overall Qual',
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_layout(
        height=400,
        xaxis_title='Overall Quality (1-10)',
        yaxis_title='Sale Price ($)',
        yaxis=dict(tickformat='$,.0f'),
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_year_vs_price_chart():
    """Create chart showing price vs year built"""
    
    if not HAS_DATA:
        return None
    
    year_avg = df_ames.groupby('Year Built')['SalePrice'].mean().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_ames['Year Built'],
        y=df_ames['SalePrice'],
        mode='markers',
        marker=dict(size=5, color='rgba(37, 99, 235, 0.3)'),
        name='Individual Houses'
    ))
    
    fig.add_trace(go.Scatter(
        x=year_avg['Year Built'],
        y=year_avg['SalePrice'],
        mode='lines',
        line=dict(color='red', width=3),
        name='Average Trend'
    ))
    
    fig.update_layout(
        title='🗓️ Price vs Year Built',
        xaxis_title='Year Built',
        yaxis_title='Sale Price ($)',
        yaxis=dict(tickformat='$,.0f'),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_price(overall_qual, gr_liv_area, garage_cars, garage_area, 
                  total_bsmt_sf, first_flr_sf, year_built, full_bath,
                  year_remod, totrms_abvgrd, lot_area, neighborhood, 
                  house_style, exterior):
    """Predict house price and return visualizations"""
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Overall Qual': [overall_qual],
        'Gr Liv Area': [gr_liv_area],
        'Garage Cars': [garage_cars],
        'Garage Area': [garage_area],
        'Total Bsmt SF': [total_bsmt_sf],
        '1st Flr SF': [first_flr_sf],
        'Year Built': [year_built],
        'Full Bath': [full_bath],
        'Year Remod/Add': [year_remod],
        'TotRms AbvGrd': [totrms_abvgrd],
        'Lot Area': [lot_area],
    })
    
    # Scale numeric features using our custom function
    input_scaled = scale_features(input_data, scaler_params)
    
    # Add all categorical columns with 0s
    for feat in feature_names:
        if feat not in input_scaled.columns:
            input_scaled[feat] = 0
    
    # Set selected categorical values to 1
    neighborhood_col = f'Neighborhood_{neighborhood}'
    if neighborhood_col in feature_names:
        input_scaled[neighborhood_col] = 1
        
    style_col = f'House Style_{house_style}'
    if style_col in feature_names:
        input_scaled[style_col] = 1
        
    exterior_col = f'Exterior 1st_{exterior}'
    if exterior_col in feature_names:
        input_scaled[exterior_col] = 1
    
    # Ensure correct column order
    input_final = input_scaled[feature_names]
    
    # Make prediction
    prediction = float(model.predict(input_final)[0])
    
    # Calculate ranges
    low_price = prediction * 0.9
    high_price = prediction * 1.1
    
    # Create summary
    summary = f"""
## 💰 Predicted Price: ${prediction:,.0f}

| Estimate | Price |
|----------|-------|
| 🔻 Low | ${low_price:,.0f} |
| 🎯 **Best Estimate** | **${prediction:,.0f}** |
| 🔺 High | ${high_price:,.0f} |

### 🏠 Property Summary
- **Quality:** {overall_qual}/10
- **Living Area:** {gr_liv_area:,} sq ft
- **Year Built:** {year_built}
- **Neighborhood:** {neighborhood}

### 📈 Model: {model_info['model_name']} | Accuracy: {model_info['r2_score']:.1%}
"""
    
    # Create visualizations
    gauge = create_price_gauge(prediction)
    scatter = create_price_vs_area_scatter(gr_liv_area, prediction)
    dist = create_price_distribution(prediction)
    
    return summary, gauge, scatter, dist


# ============================================
# BUILD GRADIO APP
# ============================================

with gr.Blocks(
    title="🏠 Real Estate Price Predictor",
    theme=gr.themes.Soft(primary_hue="blue")
) as demo:
    
    gr.Markdown("""
    # 🏠 Real Estate Price Intelligence
    ### AI-powered home valuation with interactive visualizations
    """)
    
    with gr.Tabs():
        
        # TAB 1: Price Predictor
        with gr.TabItem("💰 Price Predictor"):
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Property Features")
                    
                    overall_qual = gr.Slider(1, 10, value=5, step=1, label="Overall Quality (1-10)")
                    gr_liv_area = gr.Slider(300, 5000, value=1500, step=50, label="Living Area (sq ft)")
                    total_bsmt_sf = gr.Slider(0, 3000, value=1000, step=50, label="Basement (sq ft)")
                    first_flr_sf = gr.Slider(300, 4000, value=1200, step=50, label="1st Floor (sq ft)")
                    lot_area = gr.Slider(1000, 50000, value=10000, step=500, label="Lot Area (sq ft)")
                    year_built = gr.Slider(1900, 2025, value=2000, step=1, label="Year Built")
                    year_remod = gr.Slider(1950, 2025, value=2010, step=1, label="Year Remodeled")
                    full_bath = gr.Slider(0, 4, value=2, step=1, label="Full Bathrooms")
                    totrms_abvgrd = gr.Slider(2, 15, value=6, step=1, label="Total Rooms")
                    garage_cars = gr.Slider(0, 4, value=2, step=1, label="Garage (cars)")
                    garage_area = gr.Slider(0, 1500, value=500, step=50, label="Garage (sq ft)")
                    
                    neighborhood = gr.Dropdown(neighborhoods, value=neighborhoods[0], label="Neighborhood")
                    house_style = gr.Dropdown(house_styles, value=house_styles[0], label="House Style")
                    exterior = gr.Dropdown(exteriors, value=exteriors[0], label="Exterior")
                    
                    predict_btn = gr.Button("🔍 Predict Price", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Results")
                    summary_output = gr.Markdown()
                    gauge_plot = gr.Plot(label="Price Gauge")
            
            gr.Markdown("### 📈 Market Comparison")
            with gr.Row():
                scatter_plot = gr.Plot(label="Your Property vs Market")
                dist_plot = gr.Plot(label="Price Distribution")
            
            predict_btn.click(
                fn=predict_price,
                inputs=[overall_qual, gr_liv_area, garage_cars, garage_area,
                       total_bsmt_sf, first_flr_sf, year_built, full_bath,
                       year_remod, totrms_abvgrd, lot_area, neighborhood,
                       house_style, exterior],
                outputs=[summary_output, gauge_plot, scatter_plot, dist_plot]
            )
        
        # TAB 2: Market Analysis
        with gr.TabItem("📊 Market Analysis"):
            gr.Markdown("### Explore Housing Market Data")
            
            with gr.Row():
                with gr.Column():
                    neigh_btn = gr.Button("🏘️ Neighborhood Prices", variant="primary")
                    neigh_plot = gr.Plot()
                    neigh_btn.click(fn=create_neighborhood_price_chart, outputs=neigh_plot)
                
                with gr.Column():
                    qual_btn = gr.Button("📦 Quality vs Price", variant="primary")
                    qual_plot = gr.Plot()
                    qual_btn.click(fn=create_quality_boxplot, outputs=qual_plot)
            
            year_btn = gr.Button("🗓️ Year Built vs Price", variant="primary")
            year_plot = gr.Plot()
            year_btn.click(fn=create_year_vs_price_chart, outputs=year_plot)
        
        # TAB 3: Feature Importance
        with gr.TabItem("🎯 Feature Importance"):
            gr.Markdown("### Which Features Matter Most?")
            imp_btn = gr.Button("📊 Show Feature Importance", variant="primary")
            imp_plot = gr.Plot()
            imp_btn.click(fn=create_feature_importance_chart, outputs=imp_plot)
        
        # TAB 4: About
        with gr.TabItem("ℹ️ About"):
            gr.Markdown(f"""
## 📊 Model Information

| Metric | Value |
|--------|-------|
| **Model** | {model_info['model_name']} |
| **Accuracy (R²)** | {model_info['r2_score']:.1%} |
| **RMSE** | ${model_info['rmse']:,.0f} |
| **MAE** | ${model_info['mae']:,.0f} |
| **Features** | {model_info['n_features']} |

### Dataset
- **Source:** Ames Housing Dataset
- **Location:** Ames, Iowa (2006-2010)
- **Records:** ~2,900 home sales

### Tech Stack
XGBoost + Gradio + Plotly
            """)
    
    gr.Markdown("---\n**Built with** ❤️ using XGBoost + Gradio + Plotly")

if __name__ == "__main__":
    demo.launch()
