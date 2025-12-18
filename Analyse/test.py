import pandas as pd
import plotly.express as px

# ==============================
# 1️⃣ Charger les données
# ==============================
df = pd.read_csv("Data/SP500_features.csv", parse_dates=['Date'])
df = df.sort_values("Date")
df.set_index("Date", inplace=True)

# Créer un return futur pour colorisation
df['Return_next'] = df['Return_1'].shift(-1)

# Créer une colonne couleur : vert si return futur > 0, rouge sinon
df['Color'] = df['Return_next'].apply(lambda x: 'Up' if x>0 else 'Down')

# ==============================
# 2️⃣ Graphique interactif
# ==============================
fig = px.scatter(
    df.reset_index(),
    x='Date',
    y='Close',
    color='Color',
    color_discrete_map={'Up':'green','Down':'red'},
    hover_data=['Close','Return_next'],
    title='Prix de clôture coloré selon le retour du jour suivant'
)

# Ajouter ligne pour suivre la courbe
fig.add_scatter(x=df.index, y=df['Close'], mode='lines', line=dict(color='blue', width=1), name='Close')

# Afficher avec zoom interactif
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Close',
    hovermode='x',
    legend_title_text='Retour futur'
)

fig.show()
