import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from datetime import datetime
import os
import base64

# -------------------------
# Funzione helper per valori validi
# -------------------------
def is_valid_value(val):
    return pd.notna(val) and str(val).strip() != ""

# -------------------------
# File CSV
# -------------------------
users_file = "users_db.csv"
reports_file = "reports_db.csv"
notifications_file = "notifications.csv"
stazioni_file = "stazioni_toscana.csv"

# -------------------------
# Caricamento dati
# -------------------------
users_db = pd.read_csv(users_file) if os.path.exists(users_file) else pd.DataFrame(
    columns=["user_id","email","password","role"]
)

if os.path.exists(reports_file):
    reports_db = pd.read_csv(reports_file)
    expected_cols = ["report_id","user_id","title","description","category",
                     "status","assigned_to","stazione","latitude","longitude","photo_path"]
    for col in expected_cols:
        if col not in reports_db.columns:
            reports_db[col] = pd.NA
    reports_db = reports_db[expected_cols]
else:
    reports_db = pd.DataFrame(columns=["report_id","user_id","title","description","category",
                                       "status","assigned_to","stazione","latitude","longitude","photo_path"])
reports_db.columns = reports_db.columns.str.strip()

notifications_db = pd.read_csv(notifications_file) if os.path.exists(notifications_file) else pd.DataFrame(
    columns=["notification_id","operator_email","message","timestamp"]
)

# -------------------------
# Caricamento stazioni FS Toscana
# -------------------------
if os.path.exists(stazioni_file):
    stazioni_toscana = pd.read_csv(stazioni_file)
    stazioni_toscana.columns = stazioni_toscana.columns.str.strip()
else:
    stazioni_toscana = pd.DataFrame(columns=["nome_stazione","indirizzo","comune","provincia","regione","latitudine","longitudine"])
    st.warning(f"⚠️ File {stazioni_file} non trovato! Mettilo nella stessa cartella di app.py")

# -------------------------
# Funzioni base
# -------------------------
def save_users(): users_db.to_csv(users_file,index=False)
def save_reports(): reports_db.to_csv(reports_file,index=False)
def save_notifications(): notifications_db.to_csv(notifications_file,index=False)

def create_user(email,password,role="cittadino"):
    global users_db
    user_id = len(users_db)+1
    users_db = pd.concat([users_db,pd.DataFrame([[user_id,email,password,role]],columns=users_db.columns)],ignore_index=True)
    save_users()
    st.success(f"Utente creato: {email} ({role})")
    return user_id

def login(email,password):
    user = users_db[(users_db['email']==email) & (users_db['password']==password)]
    if not user.empty:
        return int(user.iloc[0]['user_id']), user.iloc[0]['role']
    return None, None

# -------------------------
# Classificatore automatico categoria
# -------------------------
training_data = [
    ("Il lampione in via Roma è spento","Illuminazione"),
    ("Buca davanti al civico 10","Strade"),
    ("Raccolta rifiuti non effettuata","Rifiuti"),
    ("Gioco rotto nel parco giochi","Parco/Verde"),
    ("Troppi rifiuti abbandonati in strada","Rifiuti"),
    ("Lampione lampeggia di continuo","Illuminazione"),
    ("Panchina danneggiata nel parco","Parco/Verde"),
    ("Strada dissestata davanti al supermercato","Strade")
]
desc, cat = zip(*training_data)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(desc)
clf = MultinomialNB()
clf.fit(X_train, cat)

def suggest_category(description):
    X_test = vectorizer.transform([description])
    return clf.predict(X_test)[0]

# -------------------------
# Notifiche
# -------------------------
def notify_operators(report_id,title):
    global notifications_db
    operators = users_db[users_db['role']=="operatore"]
    for _,op in operators.iterrows():
        message = f"Nuova segnalazione ID {report_id} - {title}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        notifications_db = pd.concat([notifications_db,pd.DataFrame([[len(notifications_db)+1,op['email'],message,timestamp]],columns=notifications_db.columns)],ignore_index=True)
    save_notifications()

# -------------------------
# Converti immagine in base64 per popup mappa
# -------------------------
def get_image_base64(photo_path):
    if os.path.exists(photo_path):
        with open(photo_path, "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data).decode()
            return f"data:image/png;base64,{encoded}"
    return ""

# -------------------------
# Session state
# -------------------------
for key in ['user_id','role','email']:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------
# Interfaccia Streamlit
# -------------------------
st.title("Mini-App Segnalazioni")
menu = ["Registrazione","Login"]
choice = st.sidebar.selectbox("Menu",menu)

# -------------------------
# Registrazione
# -------------------------
if choice=="Registrazione":
    st.subheader("Crea nuovo utente")
    email = st.text_input("Email")
    password = st.text_input("Password",type="password")
    role = st.selectbox("Ruolo",["cittadino","operatore"])
    if st.button("Crea Utente"):
        create_user(email,password,role)

# -------------------------
# Login e area protetta
# -------------------------
elif choice=="Login":
    if st.session_state['user_id'] is None:
        st.subheader("Login")
        email = st.text_input("Email Login")
        password = st.text_input("Password Login",type="password")
        if st.button("Login"):
            uid, r = login(email,password)
            if uid:
                st.session_state['user_id'] = uid
                st.session_state['role'] = r
                st.session_state['email'] = email
                st.success(f"Login riuscito! Ruolo: {r}")
            else:
                st.error("Login fallito")
    else:
        current_user_id = st.session_state['user_id']
        current_role = st.session_state['role']
        current_email = st.session_state['email']
        st.success(f"Utente loggato: {current_email} ({current_role})")

        # -------------------------
        # Cittadino
        # -------------------------
        if current_role=="cittadino":
            st.subheader("Crea segnalazione")
            title = st.text_input("Titolo")
            description = st.text_area("Descrizione")
            suggested = ""
            if description:
                suggested = suggest_category(description)
                st.info(f"Categoria suggerita: {suggested}")
            category = st.text_input("Categoria", value=suggested if description else "")

            # Selezione stazione FS
            stazione_list = stazioni_toscana['nome_stazione'].tolist() if not stazioni_toscana.empty else []
            stazione_selezionata = st.selectbox("Seleziona stazione FS", stazione_list)

            photo = st.file_uploader("Carica foto", type=["png","jpg","jpeg"])
            photo_path = ""
            if photo:
                if not os.path.exists("uploads"):
                    os.makedirs("uploads")
                photo_path = os.path.join("uploads", photo.name)
                with open(photo_path, "wb") as f:
                    f.write(photo.getbuffer())
                st.success("Foto caricata!")

            if st.button("Invia Segnalazione"):
                latitude = longitude = None
                if stazione_selezionata:
                    st_info = stazioni_toscana[stazioni_toscana['nome_stazione']==stazione_selezionata]
                    if not st_info.empty:
                        st_row = st_info.iloc[0]
                        latitude = st_row['latitudine']
                        longitude = st_row['longitudine']
                report_id = len(reports_db)+1
                reports_db = pd.concat([reports_db,pd.DataFrame([[report_id,current_user_id,title,description,category,"inviata","",stazione_selezionata,latitude,longitude,photo_path]],columns=reports_db.columns)],ignore_index=True)
                save_reports()
                notify_operators(report_id,title)
                st.success(f"Segnalazione inviata ID {report_id}")

            df = reports_db[reports_db['user_id']==current_user_id]
            st.subheader("Le tue segnalazioni")
            st.dataframe(df)

            st.subheader("Mappa delle tue segnalazioni")
            m = folium.Map(location=[43.771,11.254], zoom_start=8)
            for _, row in df.iterrows():
                lat = row.get('latitude')
                lon = row.get('longitude')
                if pd.notnull(lat) and pd.notnull(lon):
                    title_row = row.get('title', 'Segnalazione')
                    desc_row = row.get('description', '')
                    popup_text = f"<b>{title_row}</b><br>{desc_row}"
                    stazione_name = row.get('stazione')
                    if is_valid_value(stazione_name) and 'nome_stazione' in stazioni_toscana.columns:
                        st_info = stazioni_toscana[stazioni_toscana['nome_stazione']==stazione_name]
                        if not st_info.empty:
                            st_row = st_info.iloc[0]
                            popup_text += f"<br><b>Stazione:</b> {st_row['nome_stazione']}<br><b>Indirizzo:</b> {st_row['indirizzo']}"
                    photo_path = row.get('photo_path')
                    if is_valid_value(photo_path):
                        img_base64 = get_image_base64(photo_path)
                        if img_base64:
                            popup_text += f"<br><img src='{img_base64}' width='150'>"
                    folium.Marker([lat, lon], popup=popup_text).add_to(m)
            folium_static(m)

        # -------------------------
        # Operatore
        # -------------------------
        elif current_role=="operatore":
            st.subheader("Gestione Segnalazioni")
            categorie = ["Tutte"] + sorted(reports_db['category'].dropna().unique().tolist())
            stati = ["Tutti","inviata","presa in carico","risolta"]
            assegnati = ["Tutti"] + sorted(reports_db['assigned_to'].dropna().unique().tolist())
            filtro_categoria = st.selectbox("Filtra per Categoria", categorie)
            filtro_stato = st.selectbox("Filtra per Stato", stati)
            filtro_assegnato = st.selectbox("Filtra per Assegnato a", assegnati)

            df_filtrato = reports_db.copy()
            if filtro_categoria != "Tutte": df_filtrato = df_filtrato[df_filtrato['category']==filtro_categoria]
            if filtro_stato != "Tutti": df_filtrato = df_filtrato[df_filtrato['status']==filtro_stato]
            if filtro_assegnato != "Tutti": df_filtrato = df_filtrato[df_filtrato['assigned_to']==filtro_assegnato]

            st.dataframe(df_filtrato)

            st.subheader("Assegna e Aggiorna Stato")
            report_options = df_filtrato['report_id'].astype(str).tolist()
            if report_options:
                selected_report = st.selectbox("Seleziona Segnalazione ID", report_options)
                assigned_to = st.text_input("Assegna a (nome utente/team)")
                new_status = st.selectbox("Aggiorna Stato", ["inviata","presa in carico","risolta"])
                if st.button("Applica Modifiche"):
                    idx = reports_db[reports_db['report_id']==int(selected_report)].index[0]
                    reports_db.at[idx,'assigned_to'] = assigned_to
                    reports_db.at[idx,'status'] = new_status
                    save_reports()
                    st.success(f"Segnalazione {selected_report} aggiornata")

            st.subheader("Mappa Segnalazioni Filtrate")
            m = folium.Map(location=[43.771,11.254], zoom_start=8)
            for _, row in df_filtrato.iterrows():
                lat = row.get('latitude')
                lon = row.get('longitude')
                if pd.notnull(lat) and pd.notnull(lon):
                    title_row = row.get('title', 'Segnalazione')
                    status_row = row.get('status','')
                    assigned_to_row = row.get('assigned_to','')
                    desc_row = row.get('description','')
                    popup_text = f"<b>{title_row}</b> ({status_row})<br>Assegnato a: {assigned_to_row}<br>{desc_row}"
                    stazione_name = row.get('stazione')
                    if is_valid_value(stazione_name) and 'nome_stazione' in stazioni_toscana.columns:
                        st_info = stazioni_toscana[stazioni_toscana['nome_stazione']==stazione_name]
                        if not st_info.empty:
                            st_row = st_info.iloc[0]
                            popup_text += f"<br><b>Stazione:</b> {st_row['nome_stazione']}<br><b>Indirizzo:</b> {st_row['indirizzo']}"
                    photo_path = row.get('photo_path')
                    if is_valid_value(photo_path):
                        img_base64 = get_image_base64(photo_path)
                        if img_base64:
                            popup_text += f"<br><img src='{img_base64}' width='150'>"
                    folium.Marker([lat, lon], popup=popup_text).add_to(m)
            folium_static(m)

            st.subheader("Esporta Segnalazioni Filtrate")
            if not df_filtrato.empty:
                csv = df_filtrato.to_csv(index=False).encode('utf-8')
                st.download_button("Scarica CSV", data=csv, file_name='segnalazioni_filtrate.csv', mime='text/csv')
            else:
                st.info("Nessuna segnalazione da esportare")

            st.subheader("Notifiche")
            df_n = notifications_db[notifications_db['operator_email']==current_email]
            st.dataframe(df_n)
