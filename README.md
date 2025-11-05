
# Chatbot Intelligent Multimodal (IA Locale & Générative)

Ce projet met en œuvre un **chatbot intelligent** capable de traiter **le texte, l’audio et l’image**, entièrement déployé en **local** à l’aide du modèle **Ollama** et d’un pipeline **RAG hybride** (BM25 + similarité cosinus).

---

## 🚀 Fonctionnalités principales
- **RAG Hybride** combinant recherche sémantique et lexicale.  
- **Génération d’images** avec *Stability AI – sd-turbo*.  
- **Transcription audio** via *Whisper (base)*.  
- **Base de données PostgreSQL** pour stocker les *embeddings*, *chunks* et l’historique des conversations.  
- **Docker** pour orchestrer les services et simplifier le déploiement.  
- **Exécution locale** avec *Ollama* pour assurer la confidentialité.

---

## 🧩 Installation et exécution

### 1️⃣ Cloner le projet
```bash
git clone https://github.com/Abdellatiftayebi/Chatbot-Intelligent-Multimodal.git
```
### 2️⃣ Créer et activer un environnement virtuel

#### Sous Windows :

```bash
python -m venv venv
venv\Scripts\activate
```
#### Sous Linux / macOS :
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4️⃣ Créer l’image Docker

Assurez-vous d’avoir Docker installé, puis exécutez la commande suivante à partir du dossier contenant le fichier docker-compose.yaml (dans le dossier docker_compose) :

```bash
docker-compose up -d
```

### 5️⃣ Configurer PostgreSQL avec pgAdmin 
- Connectez-vous à pgAdmin.
- Créez un nouveau serveur puis une base de données : 
Une fois connecté dans pgAdmin:
    1. Clique sur Add New Server (ou clic droit sur "Servers" puis "Create" > "Server").
    2. Dans l'onglet General :
    Donne un nom à ta connexion, par exemple: ChatbotDB
    3. Dans l'onglet Connection:
    Host name/address: postgres_db
    Important: Ce n'est pas localhost car pgAdmin est dans un container différent, ils communiquent via le réseau Docker, donc il faut utiliser le nom du service Docker          postgres_db (nom du container).
       - Port: 5432
       - Maintenance database: chatbotdb_2025 (le nom de ta base)
       - Username: admin (Ou postgres si tu préfères)
       - Password: docuBot111
    4. Clique sur Save.

### 6️⃣ Initialiser la base de données
```bash
python Database/init_db.py
```
### 7️⃣ Lancer le serveur FastAPI
```bash
 uvicorn coding:app --reload
```
##### L’API sera disponible à l’adresse :
👉 http://127.0.0.1:8000
   ou bien tu utiliser mon interface ChatBot.html

# Technologies utilisées 
- Python, FastAPI
- Ollama (modèle local)
- Whisper (speech-to-text)
- Stability AI (sd-turbo)
- PostgreSQL / pgAdmin
- Docker
- BM25, cos_similarité

Auteur

#### Abdellatif Tayebi
- 🔗 [LinkedIn](https://www.linkedin.com/in/abdellatif-tayebi-55986b2b3)
- 📧 Contact : abdellatif.tayebi.23@ump.ac.ma
