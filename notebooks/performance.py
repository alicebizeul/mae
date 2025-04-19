import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Given data dictionary
scores = {
    100: {'PMAE_VitB': 0.294, 'MAE_VitB_75': 0.141, 'MAE_VitB': 0.217, 'PMAE_tiny_VIT': 0.185, 'MAE_tiny_VIT': 0.146},
    120: { 'MAE_tiny_VIT': 0.147 },
    200: {'PMAE_VitB': 0.311, 'MAE_VitB_75': 0.168, 'MAE_VitB': 0.238, 'PMAE_tiny_VIT': 0.212, 'MAE_tiny_VIT': 0.153},
    240: { 'MAE_tiny_VIT': 0.155 },
    300: {'PMAE_VitB': 0.311, 'MAE_VitB_75': 0.207, 'MAE_VitB': 0.255, 'PMAE_tiny_VIT': 0.222, 'MAE_tiny_VIT': 0.156},
    360: { 'MAE_tiny_VIT': 0.159 },
    400: {'PMAE_VitB': 0.318, 'MAE_VitB_75': 0.214, 'MAE_VitB': 0.277, 'PMAE_tiny_VIT': 0.225, 'MAE_tiny_VIT': 0.158},
    480: { 'MAE_tiny_VIT': 0.159 },
    500: {'PMAE_VitB': 0.320, 'MAE_VitB_75': 0.226, 'MAE_VitB': 0.292, 'PMAE_tiny_VIT': 0.224, 'MAE_tiny_VIT': 0.161},
    600: {'PMAE_VitB': 0.319, 'MAE_VitB_75': 0.230, 'MAE_VitB': 0.303, 'PMAE_tiny_VIT': 0.226, 'MAE_tiny_VIT': 0.159},
    700: {'PMAE_VitB': 0.320, 'MAE_VitB_75': 0.249, 'MAE_VitB': 0.309, 'PMAE_tiny_VIT': 0.222, 'MAE_tiny_VIT': 0.158},
    720: { 'MAE_tiny_VIT': 0.158 },
    800: {'PMAE_VitB': 0.321, 'MAE_VitB_75': 0.258, 'MAE_VitB': 0.308, 'PMAE_tiny_VIT': 0.224, 'MAE_tiny_VIT': 0.155},
    840: { 'MAE_tiny_VIT': 0.157 },
    960: { 'MAE_tiny_VIT': 0.155 },
    1000: {'MAE_VitB_75': 0.265, 'MAE_VitB': 0.316},
    1200: {'MAE_VitB_75': 0.281, 'MAE_VitB': 0.334},
    1400: {'MAE_VitB_75': 0.291, 'MAE_VitB': 0.349},
    1600: {'MAE_VitB_75': 0.297, 'MAE_VitB': 0.354},
}

# Left column: Original epochs for curves
epochs = [100, 200, 300, 400, 500, 600, 700, 800]

# Extract and convert values for Vit-T (tiny) and Vit-B (base) for left column plots.
# Multiplying each ratio by 100 to convert to percentages.
pmae_tiny = [scores[e]['PMAE_tiny_VIT'] * 100 for e in epochs]
mae_tiny   = [scores[e]['MAE_tiny_VIT'] * 100 for e in epochs]
pmae_vitb  = [scores[e]['PMAE_VitB'] * 100 for e in epochs]
mae_vitb_75   = [scores[e]['MAE_VitB_75'] * 100 for e in epochs]
mae_vitb   = [scores[e]['MAE_VitB'] * 100 for e in epochs]

# Right column: Equivalent FLOPs.
# For Vit-T, using epochs for which the corresponding MAE measurement exists.
epochs_eq_vit_t = [100, 200, 300, 400, 500, 600, 700, 800]
pmae_tiny_eq = [scores[e]['PMAE_tiny_VIT'] * 100 for e in epochs_eq_vit_t]
# Use different keys for MAE (already adjusted for FLOPs): these keys are doubled compared to the original.
epochs_eq_vit_t = [120, 240, 360, 480, 600, 720, 840, 960]
mae_tiny_eq = [scores[e]['MAE_tiny_VIT'] * 100 for e in epochs_eq_vit_t]

# For Vit-B, using all epochs where the MAE measurement is available at 2*e.
epochs_eq_vit_b = [100, 200, 300, 400, 500, 600, 700, 800]
pmae_vitb_eq = [scores[e]['PMAE_VitB'] * 100 for e in epochs_eq_vit_b]
mae_vitb_eq = [scores[e*2]['MAE_VitB'] * 100 for e in epochs_eq_vit_b]
mae_vitb_75_eq = [scores[e*2]['MAE_VitB_75'] * 100 for e in epochs_eq_vit_b]

# Create a 2x2 subplot figure.
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Vit-T (Equivalent Epochs)", "Vit-T (Equivalent FLOPs)",
        "Vit-B (Equivalent Epochs)", "Vit-B (Equivalent FLOPs)"
    )
)

# --- Left Column: Equivalent Epochs (no legend) ---
# Top left: Vit-T curves
fig.add_trace(
    go.Scatter(x=epochs, y=pmae_tiny, mode='lines+markers',
               line=dict(color='#4169E1'), showlegend=False),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=epochs, y=mae_tiny, mode='lines+markers',
               line=dict(color='#F28C28',dash="dash"), showlegend=False),
    row=1, col=1
)

# Bottom left: Vit-B curves
fig.add_trace(
    go.Scatter(x=epochs, y=pmae_vitb, mode='lines+markers',
               line=dict(color='#4169E1'), showlegend=False),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=epochs, y=mae_vitb, mode='lines+markers',
               line=dict(color='#F28C28',dash="dash"), showlegend=False),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=epochs, y=mae_vitb_75, mode='lines+markers',
               line=dict(color='#F28C28'), showlegend=False),
    row=2, col=1
)

# --- Right Column: Equivalent FLOPs ---
# Top right: Vit-T curves (explicit legend labels provided)
fig.add_trace(
    go.Scatter(x=epochs_eq_vit_t, y=pmae_tiny_eq, mode='lines+markers',
               name=r'$\text{PMAE}$', line=dict(color='#4169E1')),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(x=epochs_eq_vit_t, y=mae_tiny_eq, mode='lines+markers',
               name=r'$\text{MAE}_{\text{ocl}}$', line=dict(color='#F28C28',dash="dash")),
    row=1, col=2
)

# Bottom right: Vit-B curves (no explicit legend, so hide it)
fig.add_trace(
    go.Scatter(x=epochs_eq_vit_b, y=pmae_vitb_eq, mode='lines+markers',
               line=dict(color='#4169E1'), showlegend=False),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(x=epochs_eq_vit_b, y=mae_vitb_eq, mode='lines+markers',
               line=dict(color='#F28C28',dash="dash"),showlegend=False),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(x=epochs_eq_vit_b, y=mae_vitb_75_eq, mode='lines+markers',name=r'$\text{MAE}_{\text{std}}$',
               line=dict(color='#F28C28')),
    row=2, col=2
)

# --- Update Layout ---

# Set y-axis labels: keep for left column, remove for right column.
fig.update_yaxes(title_text="Lin. Probe Acc (%)", row=1, col=1, title_font=dict(size=20))
fig.update_yaxes(title_text="", row=1, col=2)
fig.update_yaxes(title_text="Lin. Probe Acc (%)", row=2, col=1, title_font=dict(size=20))
fig.update_yaxes(title_text="", row=2, col=2)

# Ensure the y-axis scales are the same for plots on the same row.
fig.update_yaxes(matches="y1", row=1, col=2)
fig.update_yaxes(matches="y3", row=2, col=2)

# Update x-axis for left column uniformly.
fig.update_xaxes(title_text="Epochs", title_font=dict(size=20), tickfont=dict(size=16), row=1, col=1)
fig.update_xaxes(title_text="Epochs", title_font=dict(size=20), tickfont=dict(size=16), row=2, col=1)

# Update x-axis for right column:
# Set title to "FLOPs" and remove tick labels.
fig.update_xaxes(title_text="FLOPs", title_font=dict(size=20), row=1, col=2, showticklabels=False)
fig.update_xaxes(title_text="FLOPs", title_font=dict(size=20), row=2, col=2, showticklabels=False)

# Update overall y-axis tick font.
fig.update_yaxes(tickfont=dict(size=16))

# Update overall layout.
fig.update_layout(
    template='plotly_white',
    height=800,
    width=1000,
    legend=dict(
        font=dict(size=20),
    )
)

# Save the figure to a file and display it.
fig.write_image("performance_curves.png")
