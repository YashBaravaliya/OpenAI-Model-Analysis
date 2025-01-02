import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Set page config
st.set_page_config(
    page_title="OpenAI Model Analysis Dashboard",
    # page_layout="wide",
    initial_sidebar_state="expanded"
)

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('model_info.csv')
    df['Date_Added'] = pd.to_datetime(df['Date_Added'])
    return df

filtered_df = load_data()





# Main dashboard
st.title('OpenAI Model Analysis Dashboard')

# Key metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Models", len(filtered_df))
with col2:
    st.metric("Use Cases", len(filtered_df['Use_Case'].unique()))
with col3:
    avg_token_price = filtered_df[filtered_df['Input_Token_Price'].notna()]['Input_Token_Price'].mean()
    st.metric("Avg Input Token Price", f"${avg_token_price:.6f}")

# Model Cost Comparison
st.header('Model Cost Comparison')

# Create a combined cost metric for different use cases
cost_df = filtered_df.copy()

# Function to get the primary cost metric for each use case
def get_cost_metric(row):
    if pd.notna(row['Input_Token_Price']):
        return row['Input_Token_Price']
    elif pd.notna(row['Price_Per_Image']):
        return row['Price_Per_Image']
    elif pd.notna(row['Price_Per_Minute']):
        return row['Price_Per_Minute']
    elif pd.notna(row['Price_Per_1K_Chars']):
        return row['Price_Per_1K_Chars']
    else:
        return 0

cost_df['Cost_Metric'] = cost_df.apply(get_cost_metric, axis=1)

# Create the plot
fig_cost = go.Figure()

# Add traces for each use case
for use_case in cost_df['Use_Case'].unique():
    use_case_df = cost_df[cost_df['Use_Case'] == use_case]
    
    # Add bar for each model in this use case
    fig_cost.add_trace(go.Bar(
        name=use_case,
        x=use_case_df['Model'],
        y=use_case_df['Cost_Metric'],
        text=use_case_df['Cost_Metric'].apply(lambda x: f'${x:.6f}' if x < 0.01 else f'${x:.3f}'),
        textposition='auto',
    ))

# Update layout
fig_cost.update_layout(
    title='Model Cost Comparison Across Use Cases',
    xaxis_title="Model",
    yaxis_title="Cost ($)",
    barmode='group',
    height=600,
    showlegend=True,
    xaxis_tickangle=-45,
    yaxis_type='log',  # Using log scale for better visualization of different price ranges
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

# Add hover template
fig_cost.update_traces(
    hovertemplate="<br>".join([
        "Model: %{x}",
        "Cost: %{text}",
        "Use Case: %{data.name}"
    ])
)

st.plotly_chart(fig_cost, use_container_width=True)

## Detailed Cost Analysis Table
st.header('Detailed Cost Breakdown')

# Prepare comprehensive cost table
def prepare_detailed_cost_table(df):
    # Create a copy of the dataframe with all relevant columns
    detailed_df = df[['Model', 'Use_Case', 'Date_Added']].copy()
    
    # Add text/vision related costs
    detailed_df['Input_Token_Price'] = df['Input_Token_Price']
    detailed_df['Output_Token_Price'] = df['Output_Token_Price']
    detailed_df['Cached_Input_Token_Price'] = df['Cached_Input_Token_Price']
    
    # Add batch processing costs
    detailed_df['Batch_Input_Price'] = df['batch_input_token_price']
    detailed_df['Batch_Output_Price'] = df['batch_output_token_price']
    
    # Add audio-specific costs
    detailed_df['Input_Audio_Token_Price'] = df['Input_Audio_Token_Price']
    detailed_df['Output_Audio_Token_Price'] = df['Output_Audio_Token_Price']
    detailed_df['Price_Per_Minute'] = df['Price_Per_Minute']
    detailed_df['Price_Per_1K_Chars'] = df['Price_Per_1K_Chars']
    
    # Add image-specific costs
    detailed_df['Price_Per_Image'] = df['Price_Per_Image']
    detailed_df['Quality'] = df['Quality']
    detailed_df['Resolution'] = df['Resolution']
    
    # Calculate cost ratios where applicable
    conditions = [
        (df['Input_Token_Price'].notna() & df['Output_Token_Price'].notna()),
        (df['Input_Audio_Token_Price'].notna() & df['Output_Audio_Token_Price'].notna())
    ]
    choices = [
        df['Output_Token_Price'] / df['Input_Token_Price'],
        df['Output_Audio_Token_Price'] / df['Input_Audio_Token_Price']
    ]
    detailed_df['Output/Input_Ratio'] = np.select(conditions, choices, default=np.nan)
    
    # Calculate batch savings where applicable
    condition_batch = (df['Input_Token_Price'].notna() & df['batch_input_token_price'].notna())
    detailed_df['Batch_Savings'] = np.where(
        condition_batch,
        (df['Input_Token_Price'] - df['batch_input_token_price']) / df['Input_Token_Price'] * 100,
        np.nan
    )
    
    return detailed_df

detailed_cost_df = prepare_detailed_cost_table(filtered_df)

# Create tabs for different views
cost_tabs = st.tabs([
    "Text/Vision Models", 
    "Audio Models", 
    "Image Models",
    "All Models"
])

with cost_tabs[0]:
    text_vision_costs = detailed_cost_df[detailed_cost_df['Use_Case'] == 'text+vision']
    if not text_vision_costs.empty:
        st.subheader('Text/Vision Model Costs')
        columns_to_show = [
            'Model', 'Input_Token_Price', 'Output_Token_Price', 
            'Cached_Input_Token_Price', 'Batch_Input_Price', 
            'Batch_Output_Price', 'Output/Input_Ratio', 'Batch_Savings'
        ]
        st.dataframe(
            text_vision_costs[columns_to_show].style.format({
                'Input_Token_Price': '${:.6f}',
                'Output_Token_Price': '${:.6f}',
                'Cached_Input_Token_Price': '${:.6f}',
                'Batch_Input_Price': '${:.6f}',
                'Batch_Output_Price': '${:.6f}',
                'Output/Input_Ratio': '{:.2f}x',
                'Batch_Savings': '{:.1f}%'
            })
        )

with cost_tabs[1]:
    audio_costs = detailed_cost_df[
        (detailed_cost_df['Use_Case'] == 'text+audio') | 
        (detailed_cost_df['Price_Per_Minute'].notna()) |
        (detailed_cost_df['Price_Per_1K_Chars'].notna())
    ]
    if not audio_costs.empty:
        st.subheader('Audio Model Costs')
        columns_to_show = [
            'Model', 'Use_Case', 'Input_Audio_Token_Price', 
            'Output_Audio_Token_Price', 'Price_Per_Minute', 
            'Price_Per_1K_Chars', 'Output/Input_Ratio'
        ]
        st.dataframe(
            audio_costs[columns_to_show].style.format({
                'Input_Audio_Token_Price': '${:.3f}',
                'Output_Audio_Token_Price': '${:.3f}',
                'Price_Per_Minute': '${:.3f}',
                'Price_Per_1K_Chars': '${:.3f}',
                'Output/Input_Ratio': '{:.2f}x'
            })
        )

with cost_tabs[2]:
    image_costs = detailed_cost_df[detailed_cost_df['Use_Case'] == 'image']
    if not image_costs.empty:
        st.subheader('Image Model Costs')
        columns_to_show = ['Model', 'Quality', 'Resolution', 'Price_Per_Image']
        st.dataframe(
            image_costs[columns_to_show].style.format({
                'Price_Per_Image': '${:.3f}'
            })
        )

with cost_tabs[3]:
    st.subheader('All Models - Complete Cost Breakdown')
    
    # Allow user to select columns to display
    all_columns = detailed_cost_df.columns.tolist()
    selected_columns = st.multiselect(
        'Select columns to display',
        all_columns,
        default=['Model', 'Use_Case', 'Input_Token_Price', 'Output_Token_Price', 'Price_Per_Image', 'Price_Per_Minute']
    )
    
    if selected_columns:
        # Sort by the first numeric column by default
        numeric_columns = detailed_cost_df[selected_columns].select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            sort_column = numeric_columns[0]
            sorted_df = detailed_cost_df[selected_columns].sort_values(sort_column, ascending=False)
        else:
            sorted_df = detailed_cost_df[selected_columns]
            
        st.dataframe(
            sorted_df.style.format({
                'Input_Token_Price': '${:.6f}',
                'Output_Token_Price': '${:.6f}',
                'Cached_Input_Token_Price': '${:.6f}',
                'Batch_Input_Price': '${:.6f}',
                'Batch_Output_Price': '${:.6f}',
                'Input_Audio_Token_Price': '${:.3f}',
                'Output_Audio_Token_Price': '${:.3f}',
                'Price_Per_Minute': '${:.3f}',
                'Price_Per_1K_Chars': '${:.3f}',
                'Price_Per_Image': '${:.3f}',
                'Output/Input_Ratio': '{:.2f}x',
                'Batch_Savings': '{:.1f}%'
            })
        )

# Add download button for the complete dataset
csv = detailed_cost_df.to_csv(index=False)
st.download_button(
    label="Download Complete Cost Analysis",
    data=csv,
    file_name="model_cost_analysis.csv",
    mime="text/csv"
)

# Price Correlation Analysis
st.header('Model Price Correlation Analysis')

# Prepare data for visualization
def prepare_price_data(df):
    price_df = df.copy()
    
    # Create primary and secondary price columns based on model type
    def get_primary_price(row):
        if pd.notna(row['Input_Token_Price']):
            return row['Input_Token_Price']
        elif pd.notna(row['Price_Per_Image']):
            return row['Price_Per_Image']
        elif pd.notna(row['Price_Per_Minute']):
            return row['Price_Per_Minute']
        elif pd.notna(row['Price_Per_1K_Chars']):
            return row['Price_Per_1K_Chars']
        elif pd.notna(row['Input_Audio_Token_Price']):
            return row['Input_Audio_Token_Price']
        return None

    def get_secondary_price(row):
        if pd.notna(row['Output_Token_Price']):
            return row['Output_Token_Price']
        elif pd.notna(row['Output_Audio_Token_Price']):
            return row['Output_Audio_Token_Price']
        return None

    price_df['Primary_Price'] = price_df.apply(get_primary_price, axis=1)
    price_df['Secondary_Price'] = price_df.apply(get_secondary_price, axis=1)
    
    # Calculate price ratio where applicable
    price_df['Price_Ratio'] = price_df.apply(
        lambda x: x['Secondary_Price'] / x['Primary_Price'] 
        if pd.notna(x['Secondary_Price']) and pd.notna(x['Primary_Price']) and x['Primary_Price'] != 0 
        else 1, 
        axis=1
    )
    
    return price_df[['Model', 'Use_Case', 'Primary_Price', 'Secondary_Price', 'Price_Ratio']]

price_analysis_df = prepare_price_data(filtered_df)
price_analysis_df = price_analysis_df.dropna(subset=['Primary_Price'])


fig_distribution = go.Figure()

# Add box plots for primary prices
fig_distribution.add_trace(go.Box(
    y=price_analysis_df['Primary_Price'],
    x=price_analysis_df['Use_Case'],
    name='Primary Price',
    boxpoints='all',
    text=price_analysis_df['Model'],
    hovertemplate="<br>".join([
        "Model: %{text}",
        "Price: $%{y:.6f}",
        "Use Case: %{x}"
    ])
))

# Add box plots for secondary prices where applicable
secondary_price_df = price_analysis_df[price_analysis_df['Secondary_Price'].notna()]
if not secondary_price_df.empty:
    fig_distribution.add_trace(go.Box(
        y=secondary_price_df['Secondary_Price'],
        x=secondary_price_df['Use_Case'],
        name='Secondary Price',
        boxpoints='all',
        text=secondary_price_df['Model'],
        hovertemplate="<br>".join([
            "Model: %{text}",
            "Price: $%{y:.6f}",
            "Use Case: %{x}"
        ])
    ))

fig_distribution.update_layout(
    title='Price Distribution by Model Type',
    yaxis_title="Price ($)",
    xaxis_title="Use Case",
    yaxis_type="log",
    height=600,
    showlegend=True
)

st.plotly_chart(fig_distribution, use_container_width=True)

# Add detailed price analysis table
st.subheader('Detailed Price Analysis')
st.dataframe(
    price_analysis_df.style.format({
        'Primary_Price': '${:.6f}',
        'Secondary_Price': '${:.6f}',
        'Price_Ratio': '{:.2f}x'
    })
)

# Text and Vision Models Analysis
st.header('Text + Vision Models Analysis')
text_vision_df = filtered_df[filtered_df['Use_Case'] == 'text+vision']

if not text_vision_df.empty:
    # Create tabs for different visualizations
    tv_tab1, tv_tab2 = st.tabs(["Token Price Comparison", "Batch Processing"])
    
    with tv_tab1:
        fig_tv = go.Figure()
        fig_tv.add_trace(go.Bar(
            name='Input Token Price',
            x=text_vision_df['Model'],
            y=text_vision_df['Input_Token_Price'],
            text=text_vision_df['Input_Token_Price'].apply(lambda x: f'${x:.6f}'),
            textposition='auto',
        ))
        fig_tv.add_trace(go.Bar(
            name='Output Token Price',
            x=text_vision_df['Model'],
            y=text_vision_df['Output_Token_Price'],
            text=text_vision_df['Output_Token_Price'].apply(lambda x: f'${x:.6f}'),
            textposition='auto',
        ))
        fig_tv.update_layout(
            title='Text + Vision Models - Token Pricing',
            barmode='group',
            xaxis_tickangle=-45,
            height=500,
            yaxis_title="Price per Token ($)",
            xaxis_title="Model"
        )
        st.plotly_chart(fig_tv, use_container_width=True)

    with tv_tab2:
        # Batch processing comparison
        batch_df = text_vision_df[text_vision_df['Has_Batch_API'] == 1.0]
        if not batch_df.empty:
            fig_batch = go.Figure()
            fig_batch.add_trace(go.Bar(
                name='Regular Input Price',
                x=batch_df['Model'],
                y=batch_df['Input_Token_Price'],
                text=batch_df['Input_Token_Price'].apply(lambda x: f'${x:.6f}'),
                textposition='auto',
            ))
            fig_batch.add_trace(go.Bar(
                name='Batch Input Price',
                x=batch_df['Model'],
                y=batch_df['batch_input_token_price'],
                text=batch_df['batch_input_token_price'].apply(lambda x: f'${x:.6f}'),
                textposition='auto',
            ))
            fig_batch.update_layout(
                title='Batch vs Regular Processing Prices',
                barmode='group',
                xaxis_tickangle=-45,
                height=500,
                yaxis_title="Price per Token ($)",
                xaxis_title="Model"
            )
            st.plotly_chart(fig_batch, use_container_width=True)
        else:
            st.info("No batch processing data available for the selected models.")

# Audio Models Detailed Analysis
st.header('Text + Audio Models Analysis')
audio_df = filtered_df[filtered_df['Use_Case'] == 'text+audio']

if not audio_df.empty:
    # Create tabs for different visualizations
    audio_tab1, audio_tab2 = st.tabs(["Token Prices", "Audio Token Prices"])
    
    with audio_tab1:
        fig_audio1 = go.Figure()
        fig_audio1.add_trace(go.Bar(
            name='Input Token Price',
            x=audio_df['Model'],
            y=audio_df['Input_Token_Price'],
            text=audio_df['Input_Token_Price'].apply(lambda x: f'${x:.6f}'),
            textposition='auto',
        ))
        fig_audio1.add_trace(go.Bar(
            name='Output Token Price',
            x=audio_df['Model'],
            y=audio_df['Output_Token_Price'],
            text=audio_df['Output_Token_Price'].apply(lambda x: f'${x:.6f}'),
            textposition='auto',
        ))
        fig_audio1.update_layout(
            title='Text + Audio Models - Text Token Pricing',
            barmode='group',
            xaxis_tickangle=-45,
            height=500,
            yaxis_title="Price per Token ($)",
            xaxis_title="Model"
        )
        st.plotly_chart(fig_audio1, use_container_width=True)
    
    with audio_tab2:
        fig_audio2 = go.Figure()
        fig_audio2.add_trace(go.Bar(
            name='Input Audio Token Price',
            x=audio_df['Model'],
            y=audio_df['Input_Audio_Token_Price'],
            text=audio_df['Input_Audio_Token_Price'].apply(lambda x: f'${x:.3f}'),
            textposition='auto',
        ))
        fig_audio2.add_trace(go.Bar(
            name='Output Audio Token Price',
            x=audio_df['Model'],
            y=audio_df['Output_Audio_Token_Price'],
            text=audio_df['Output_Audio_Token_Price'].apply(lambda x: f'${x:.3f}'),
            textposition='auto',
        ))
        fig_audio2.update_layout(
            title='Text + Audio Models - Audio Token Pricing',
            barmode='group',
            xaxis_tickangle=-45,
            height=500,
            yaxis_title="Price per Audio Token ($)",
            xaxis_title="Model"
        )
        st.plotly_chart(fig_audio2, use_container_width=True)

# Pure Audio Models Analysis
st.header('Pure Audio Models Analysis')
pure_audio_df = filtered_df[
    (filtered_df['Price_Per_Minute'].notna()) | 
    (filtered_df['Price_Per_1K_Chars'].notna())
]

if not pure_audio_df.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        # Price per minute models
        per_minute_df = pure_audio_df[pure_audio_df['Price_Per_Minute'].notna()]
        if not per_minute_df.empty:
            fig_per_minute = px.bar(
                per_minute_df,
                x='Model',
                y='Price_Per_Minute',
                title='Price per Minute - Audio Models',
                text_auto='.3f',
                height=400
            )
            fig_per_minute.update_layout(
                xaxis_tickangle=-45,
                yaxis_title="Price per Minute ($)"
            )
            st.plotly_chart(fig_per_minute, use_container_width=True)
    
    with col2:
        # Price per 1K chars models
        per_char_df = pure_audio_df[pure_audio_df['Price_Per_1K_Chars'].notna()]
        if not per_char_df.empty:
            fig_per_char = px.bar(
                per_char_df,
                x='Model',
                y='Price_Per_1K_Chars',
                title='Price per 1K Characters - Audio Models',
                text_auto='.3f',
                height=400
            )
            fig_per_char.update_layout(
                xaxis_tickangle=-45,
                yaxis_title="Price per 1K Characters ($)"
            )
            st.plotly_chart(fig_per_char, use_container_width=True)

# Image Models Analysis remains the same
st.header('Image Models Analysis')
image_df = filtered_df[filtered_df['Use_Case'] == 'image']

if not image_df.empty:
    fig_image = px.bar(
        image_df,
        x='Model',
        y='Price_Per_Image',
        color='Quality',
        facet_col='Resolution',
        title='Image Model Pricing by Resolution and Quality'
    )
    fig_image.update_layout(height=500)
    st.plotly_chart(fig_image, use_container_width=True)

# Add price comparison tables
st.header('Detailed Price Comparison Tables')

tabs = st.tabs([
    "Text + Vision Models", 
    "Text + Audio Models", 
    "Pure Audio Models", 
    "Image Models"
])

with tabs[0]:
    if not text_vision_df.empty:
        st.dataframe(
            text_vision_df[['Model', 'Input_Token_Price', 'Output_Token_Price', 'batch_input_token_price', 'batch_output_token_price']]
            .style.format({
                'Input_Token_Price': '${:.6f}',
                'Output_Token_Price': '${:.6f}',
                'batch_input_token_price': '${:.6f}',
                'batch_output_token_price': '${:.6f}'
            })
        )

with tabs[1]:
    if not audio_df.empty:
        st.dataframe(
            audio_df[['Model', 'Input_Token_Price', 'Output_Token_Price', 'Input_Audio_Token_Price', 'Output_Audio_Token_Price']]
            .style.format({
                'Input_Token_Price': '${:.6f}',
                'Output_Token_Price': '${:.6f}',
                'Input_Audio_Token_Price': '${:.3f}',
                'Output_Audio_Token_Price': '${:.3f}'
            })
        )

with tabs[2]:
    if not pure_audio_df.empty:
        st.dataframe(
            pure_audio_df[['Model', 'Price_Per_Minute', 'Price_Per_1K_Chars']]
            .style.format({
                'Price_Per_Minute': '${:.3f}',
                'Price_Per_1K_Chars': '${:.3f}'
            })
        )

with tabs[3]:
    if not image_df.empty:
        st.dataframe(
            image_df[['Model', 'Quality', 'Resolution', 'Price_Per_Image']]
            .style.format({
                'Price_Per_Image': '${:.3f}'
            })
        )

# Download filtered data
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download Filtered Data",
    data=csv,
    file_name="filtered_model_data.csv",
    mime="text/csv"
)
