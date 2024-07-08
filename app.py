import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans

# Set the page configuration
st.set_page_config(
    page_title="Diagnostic Center Agent Scheduling",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com',
        'Report a bug': "https://www.example.com",
        'About': "# This is a Streamlit app for scheduling agents at a diagnostic center."
    }
)

# Load the data
@st.cache_data()
def clustering(x, y):
    df = pd.read_csv("https://raw.githubusercontent.com/asaikiran1999/diagnostic-center-agents-sheduling/main/final_data.csv")
    df_filtered = df[df['Sample Collection Date'] == x].copy()

    if df_filtered.empty:
        raise ValueError(f"No data found for date: {x}")

    le = LabelEncoder()
    df_filtered['patient location'] = le.fit_transform(df_filtered['patient location'])
    df_filtered['Diagnostic Centers'] = le.fit_transform(df_filtered['Diagnostic Centers'])
    df_filtered['Availabilty time (Patient)'] = le.fit_transform(df_filtered['Availabilty time (Patient)'])
    X = df_filtered[['patient location', 'Diagnostic Centers', 'shortest distance Patient-Pathlab(m)', 'Availabilty time (Patient)']].copy()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    kmeans = KMeans(n_clusters=y, random_state=1234)
    kmeans.fit(X_scaled_df)

    labels = kmeans.labels_
    df_filtered["Agent id"] = labels
    
    return df_filtered.copy()


def main():
    # Custom CSS styles
    st.markdown("""
        <style>
            body {
                font-family: 'Montserrat', sans-serif;
                background-color: #f5f5f5;
            }
            
            .stButton button {
                background-color: #007bff;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 18px;
                font-weight: bold;
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
                transition: background-color 0.3s ease;
            }
            
            .stButton button:hover {
                background-color: #0056b3;
            }
            
            .stSelectbox select {
                background-color: #f1f1f1;
                color: #333;
                padding: 12px;
                border: none;
                border-radius: 6px;
                font-size: 18px;
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
            }
            
            .stNumberInput input {
                background-color: #f1f1f1;
                color: #333;
                padding: 12px;
                border: none;
                border-radius: 6px;
                font-size: 18px;
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
            }
            
            .stDataFrame {
                background-color: white;
                color: #333;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
                font-size: 16px;
            }
            
            .stDataFrame th {
                background-color: #007bff;
                color: white;
                font-size: 20px;
                font-weight: bold;
                padding: 14px;
                border-radius: 6px;
            }
            
            .stDataFrame td {
                font-size: 18px;
                padding: 12px;
            }
            
            @media (max-width: 767px) {
                .stButton button {
                    padding: 10px 20px;
                    font-size: 16px;
                }
                
                .stSelectbox select,
                .stNumberInput input {
                    padding: 10px;
                    font-size: 16px;
                }
                
                .stDataFrame th {
                    font-size: 18px;
                    padding: 12px;
                }
                
                .stDataFrame td {
                    font-size: 16px;
                    padding: 10px;
                }
            }
        </style>
    """, unsafe_allow_html=True)

    st.title('Diagnostic Center Agent Scheduling')

    # Sidebar inputs
    st.sidebar.title('Filters')
    date = st.sidebar.selectbox('Date for schedule generation (YYYY-MM-DD)', ['2022-01-02 00:00:00', '2022-01-03 00:00:00', '2022-01-04 00:00:00', '2022-01-05 00:00:00', '2022-01-06 00:00:00', '2022-01-07 00:00:00', '2022-01-08 00:00:00', '2022-01-09 00:00:00', '2022-01-10 00:00:00', '2022-01-11 00:00:00', '2022-01-12 00:00:00', '2022-01-13 00:00:00', '2022-01-14 00:00:00', '2022-01-15 00:00:00', '2022-01-16 00:00:00', '2022-01-17 00:00:00', '2022-01-18 00:00:00', '2022-01-19 00:00:00', '2022-01-20 00:00:00', '2022-01-21 00:00:00', '2022-01-22 00:00:00', '2022-01-23 00:00:00', '2022-01-24 00:00:00', '2022-01-25 00:00:00', '2022-01-26 00:00:00', '2022-01-27 00:00:00', '2022-01-28 00:00:00', '2022-01-29 00:00:00', '2022-01-30 00:00:00', '2022-01-31 00:00:00'])
    num_agents = st.sidebar.number_input('Enter number of agents:', min_value=2, step=1)

    if st.sidebar.button('Generate Clusters'):
        if date and num_agents:
            try:
                df_result = clustering(date, num_agents)
                st.session_state.df_result = df_result  # Store result in session state
                st.success('Clusters generated successfully!')
            except ValueError as e:
                st.error(f'Error generating clusters: {e}')

    # Display agent schedule
    if 'df_result' in st.session_state:
        df_result = st.session_state.df_result  # Retrieve result from session state
        agent_id = st.number_input('Enter agent ID to show schedule:', min_value=0, max_value=num_agents-1, step=1, value=0)

        if st.button('Show Agent Schedule'):
            try:
                df_result = df_result.reset_index()
                df_agent = df_result[df_result["Agent id"] == agent_id]

                # Ensure 'Availabilty time (Patient)' column is processed correctly
                df_agent['avail'] = df_agent['Availabilty time (Patient)'].apply(
                    lambda x: int(x.split('to')[0].strip().split(':')[0]) if isinstance(x, str) and 'to' in x else None
                )

                # Sort and filter columns
                df_agent_sorted = df_agent.sort_values(['avail', 'shortest distance Patient-Pathlab(m)'], ascending=[True, True])
                df_agent_filtered = df_agent_sorted.drop(['Test Booking Time HH:MM', 'Test Booking Date', 'shortest distance Patient-Pathlab(m)', 'Sample Collection Date'], axis=1)
                df_agent_filtered = df_agent_filtered.drop(df_agent_filtered.columns[10], axis=1)

                # Reorder columns
                first_column = df_agent_filtered.pop('Availabilty time (Patient)')
                df_agent_filtered.insert(0, 'Availabilty time (Patient)', first_column)
                st.write(df_agent_filtered)
            except KeyError as e:
                st.error(f'Error accessing DataFrame: {e}')
            except Exception as e:
                st.error(f'Error processing schedule data: {e}')


if __name__ == '__main__':  
    main()
