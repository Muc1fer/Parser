import streamlit as st
import pandas as pd
import numpy as np
import io
import altair as alt

def parse_earnings_csv(file):
    # Read the CSV without header
    df = pd.read_csv(file, header=None)
    
    # Extract dates from first row, columns 1 onwards
    dates = df.iloc[0, 1:].values.astype(str)
    
    # Find the row indices for implied and actual
    implied_row = df[df[0] == 'Implied Earn Move'].index[0]
    actual_row = df[df[0] == 'Actual Earn Move'].index[0]
    
    # Extract implied and actual values (columns 1 onwards)
    implied_values = df.iloc[implied_row, 1:].values
    actual_values = df.iloc[actual_row, 1:].values
    
    # Convert to float, handling empty strings or NaN
    implied = []
    for val in implied_values:
        try:
            implied.append(float(val) if val != '' else np.nan)
        except (ValueError, TypeError):
            implied.append(np.nan)
    
    actual = []
    for val in actual_values:
        try:
            actual.append(float(val) if val != '' else np.nan)
        except (ValueError, TypeError):
            actual.append(np.nan)
    
    # Determine number of past quarters: count non-NaN actuals from the start
    num_past = 0
    for val in actual:
        if not np.isnan(val):
            num_past += 1
        else:
            break  # Assume past are consecutive
    
    # Past dates, implied, and actual
    past_dates = dates[:num_past]
    past_implied = implied[:num_past]
    past_actual = actual[:num_past]
    
    # Upcoming implied: first non-NaN after past, or last if no future
    future_implied = implied[num_past:]
    upcoming = next((x for x in future_implied if not np.isnan(x)), np.nan)
    
    # Absolute actual
    abs_actual = [abs(x) for x in past_actual if not np.isnan(x)]
    
    # 1. Counts of RV > IV
    def count_surpasses(last_n):
        if num_past < last_n:
            last_implied = past_implied
            last_abs = abs_actual
        else:
            last_implied = past_implied[-last_n:]
            last_abs = abs_actual[-last_n:]
        
        count = 0
        for i in range(len(last_abs)):
            imp = last_implied[i]
            if not np.isnan(imp) and imp > 0 and last_abs[i] > imp:
                count += 1
        return count
    
    count_4 = count_surpasses(4)
    count_8 = count_surpasses(8)
    
    # 2. Largest multiple: max |actual|/implied where implied > 0
    multiples = []
    for i in range(len(past_implied)):
        imp = past_implied[i]
        if not np.isnan(imp) and imp > 0:
            multiples.append(abs_actual[i] / imp)
    
    largest_multiple = max(multiples) if multiples else np.nan
    
    # 3. Avg Implied for last 8 past, avg only non-zero/non-nan
    if num_past < 8:
        last_8_imp = [x for x in past_implied if not np.isnan(x) and x > 0]
    else:
        last_8_imp = [x for x in past_implied[-8:] if not np.isnan(x) and x > 0]
    
    aim = sum(last_8_imp) / len(last_8_imp) if last_8_imp else np.nan  # AIM
    
    # 4. Avg Actual (RV) for last 8 past, absolute
    if num_past < 8:
        last_8_abs = abs_actual
    else:
        last_8_abs = abs_actual[-8:]
    
    aam = sum(last_8_abs) / len(last_8_abs) if last_8_abs else np.nan  # AAM
    
    cim = upcoming  # CIM
    
   # Suggestion 1: Scenario Classification and Validity Flag with emojis
    validity = "Valid ✅"
    scenario = ""
    approx_threshold = 0.5  # Percentage points for ≈
    
    if count_4 > 1:
        validity = "Invalid (Jump Risk) ❗"
        scenario = "Setup invalid due to excessive RV>IV events."
    else:
        if count_8 > 2:
            validity = "Valid (but do not use full size) ✅❗"
        if not np.isnan(aim) and not np.isnan(aam):
            if aim > aam:
                scenario = "Good Candidate (Market Overprices Vol) ✅"
            elif aim < aam:
                scenario = "Bad Candidate (Market Underprices Vol) ❗"
            else:  # aim ≈ aam
                scenario = "Possible Candidate (Stable Earnings Behavior - Evaluate CIM) ✅❗"
                if not np.isnan(cim):
                    if cim > aim:
                        scenario = "Good Candidate (Vol is overpriced) ✅"
                    elif cim < aim:
                        scenario = "Possible Candidate (Shrinking Expected Move Band - Check Conditions) ✅❗"
    
    # Suggestion 2: Metrics for Realized Move Behavior and Clustering
    def percent_centered(last_n):
        if num_past < last_n:
            last_implied = past_implied
            last_abs = abs_actual
        else:
            last_implied = past_implied[-last_n:]
            last_abs = abs_actual[-last_n:]
        
        centered_count = 0
        total_valid = 0
        for i in range(len(last_abs)):
            imp = last_implied[i]
            if not np.isnan(imp) and imp > 0:
                total_valid += 1
                if last_abs[i] < 0.5 * imp:
                    centered_count += 1
        return (centered_count / total_valid * 100) if total_valid > 0 else np.nan
    
    centered_4 = percent_centered(4)
    centered_8 = percent_centered(8)
    
    # Clustering of exceedances
    exceed_indices = [i for i in range(len(past_implied)) if not np.isnan(past_implied[i]) and past_implied[i] > 0 and abs_actual[i] > past_implied[i]]
    cluster_pattern = "None"
    if exceed_indices:
        if len(exceed_indices) > 1:
            consecutive = all(exceed_indices[j] == exceed_indices[j-1] + 1 for j in range(1, len(exceed_indices)))
            if consecutive:
                cluster_pattern = f"Clustered ({len(exceed_indices)} consecutive)"
            else:
                cluster_pattern = "Sporadic"
        else:
            cluster_pattern = "Sporadic (Single)"
    
    # Suggestion 3: Trend Analysis for Implied Moves
    if num_past >= 8:
        recent_4_imp = [x for x in past_implied[-4:] if not np.isnan(x) and x > 0]
        prior_4_imp = [x for x in past_implied[-8:-4] if not np.isnan(x) and x > 0]
    elif num_past >= 4:
        recent_4_imp = [x for x in past_implied[-4:] if not np.isnan(x) and x > 0]
        prior_4_imp = [x for x in past_implied[:-4] if not np.isnan(x) and x > 0]
    else:
        recent_4_imp = [x for x in past_implied if not np.isnan(x) and x > 0]
        prior_4_imp = []
    
    avg_recent_4 = sum(recent_4_imp) / len(recent_4_imp) if recent_4_imp else np.nan
    avg_prior_4 = sum(prior_4_imp) / len(prior_4_imp) if prior_4_imp else np.nan
    
    trend_pct = ((avg_recent_4 - avg_prior_4) / avg_prior_4 * 100) if not np.isnan(avg_prior_4) and avg_prior_4 != 0 else 0
    if abs(trend_pct) < 5:
        trend_desc = "Stable"
    elif trend_pct > 0:
        trend_desc = f"Expanding by {round(trend_pct, 2)}%"
    else:
        trend_desc = f"Contracting by {round(abs(trend_pct), 2)}%"
    
    # Clean past_implied for chart
    clean_past_implied = [x if not np.isnan(x) else 0 for x in past_implied]
    
    # Suggestion 4: Tabular data
    past_multiples = [abs_actual[i] / past_implied[i] if not np.isnan(past_implied[i]) and past_implied[i] > 0 else np.nan for i in range(len(past_actual))]
    df_past = pd.DataFrame({
        'Date': past_dates,
        'Implied Move (%)': [round(x, 2) if not np.isnan(x) else 'N/A' for x in past_implied],
        'Actual Move (%)': [round(x, 2) if not np.isnan(x) else 'N/A' for x in past_actual],
        'Abs Actual (%)': [round(abs(x), 2) if not np.isnan(x) else 'N/A' for x in past_actual],
        'Multiple': [round(m, 2) if not np.isnan(m) else 'N/A' for m in past_multiples]
    })
    
    # Chart data for improved implied vs actual chart (last 8 or less)
    chart_data = []
    for i in range(max(0, num_past - 8), num_past):
        date = past_dates[i]
        imp = past_implied[i] if not np.isnan(past_implied[i]) else 0
        act = past_actual[i] if not np.isnan(past_actual[i]) else 0
        direction = 'Positive' if act >= 0 else 'Negative'
        chart_data.append({'Date': date, 'Implied': imp, 'Actual': act, 'Direction': direction})
    
    df_chart = pd.DataFrame(chart_data)
    df_chart['Neg Implied'] = -df_chart['Implied']
    
    # Return results
    results = {
        'RV > IV past 4 ER': count_4,
        'RV > IV past 8 ER': count_8,
        'Largest Multiple': round(largest_multiple, 2) if not np.isnan(largest_multiple) else 'N/A',
        'Avg Implied Move % (AIM)': round(aim, 2) if not np.isnan(aim) else 'N/A',
        'Avg RV % (AAM)': round(aam, 2) if not np.isnan(aam) else 'N/A',
        'Expected Move % (Upcoming, CIM)': round(cim, 2) if not np.isnan(cim) else 'N/A',
        'Validity Flag': validity,
        'Scenario Classification': scenario,
        '% Centered (Last 4)': round(centered_4, 2) if not np.isnan(centered_4) else 'N/A',
        '% Centered (Last 8)': round(centered_8, 2) if not np.isnan(centered_8) else 'N/A',
        'Exceedance Pattern': cluster_pattern,
        'Implied Move Trend': trend_desc
    }
    
    return results, df_past, clean_past_implied, df_chart

# Streamlit app
st.title("Earnings CSV Parser")

uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    for file in reversed(uploaded_files):
        st.subheader(f"Results for {file.name}")
        results, df_past, clean_past_implied, df_chart = parse_earnings_csv(file)
        
        # Display validity and scenario prominently
        st.write(f"**Validity Flag:** {results['Validity Flag']}")
        st.write(f"**Scenario Classification:** {results['Scenario Classification']}")
        
        # Display key metrics using a DataFrame for perfect alignment (no index column)
        st.subheader("Key Metrics")
        
        metrics_data = {
            "Metric": [
                "RV > IV past 4 ER",
                "RV > IV past 8 ER",
                "Avg Implied Move % (AIM)",
                "Avg RV % (AAM)",
                "Current Expected Move % (CIM)",
                "Largest Multiple"
            ],
            "Value": [
                results['RV > IV past 4 ER'],
                results['RV > IV past 8 ER'],
                f"{results['Avg Implied Move % (AIM)']}%",
                f"{results['Avg RV % (AAM)']}%",
                f"{results['Expected Move % (Upcoming, CIM)']}%",
                results['Largest Multiple']
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        styled_metrics = metrics_df.style.set_properties(**{'text-align': 'right'}, subset=['Value']) \
                                         .set_properties(**{'font-weight': 'bold'}, subset=['Metric']) \
                                         .hide(axis="index")
        st.table(styled_metrics)
        
        # Additional Metrics in the same table format
        st.subheader("Additional Metrics")
        
        additional_data = {
            "Metric": [
                "% Centered (Last 4)",
                "% Centered (Last 8)",
                "Exceedance Pattern",
                "Implied Move Trend"
            ],
            "Value": [
                f"{results['% Centered (Last 4)']}%",
                f"{results['% Centered (Last 8)']}%",
                results['Exceedance Pattern'],
                results['Implied Move Trend']
            ]
        }
        
        additional_df = pd.DataFrame(additional_data)
        styled_additional = additional_df.style.set_properties(**{'text-align': 'right'}, subset=['Value']) \
                                                .set_properties(**{'font-weight': 'bold'}, subset=['Metric']) \
                                                .hide(axis="index")
        st.table(styled_additional)
        
        # Copyable data for Google Sheets
        st.subheader("Paste into Google Sheets")
        
        # Format percentage values with % sign
        aim_val = results['Avg Implied Move % (AIM)']
        aam_val = results['Avg RV % (AAM)']
        cim_val = results['Expected Move % (Upcoming, CIM)']
        
        aim_str = f"{aim_val}%" if aim_val != 'N/A' else 'N/A'
        aam_str = f"{aam_val}%" if aam_val != 'N/A' else 'N/A'
        cim_str = f"{cim_val}%" if cim_val != 'N/A' else 'N/A'
        
        copyable = f"{results['RV > IV past 4 ER']}\t{results['RV > IV past 8 ER']}\t{aim_str}\t{aam_str}\t{cim_str}\t{results['Largest Multiple']}"
        st.code(copyable, language='text')
        
        # Tabular view
        st.subheader("Past Earnings Data")
        st.dataframe(df_past)
        
        # Line chart for implied trend
        st.subheader("Implied Moves Trend")
        st.line_chart(clean_past_implied)
        
        # Improved bar chart for comparison
        st.subheader("Implied vs Actual Moves (Last 8 Quarters)")
        
        background_band = alt.Chart(df_chart).mark_rect(opacity=0.2, color='yellow').encode(
            x='Date:O',
            y='Neg Implied:Q',
            y2='Implied:Q'
        )
        
        actual_bars = alt.Chart(df_chart).mark_bar(width=20).encode(
            x='Date:O',
            y='Actual:Q',
            color=alt.Color('Direction:N', scale=alt.Scale(domain=['Positive', 'Negative'], range=['green', 'red'])),
            tooltip=['Date', 'Actual', 'Implied']
        )
        
        chart = background_band + actual_bars
        chart = chart.properties(
            width=600,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)