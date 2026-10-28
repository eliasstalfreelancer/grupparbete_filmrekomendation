# Filmrekommendationssystem

## Inledning

Detta projekt är ett grupparbete inom Data Science där vi har utvecklat ett filmrekommendationssystem.

Syftet med projektet är att rekommendera filmer utifrån filmer som användaren redan tycker om.

Projektet innehåller två modeller:

- **Baseline-modell** – använder filmernas genrer.
- **Huvudmodell** – använder både genrer och keywords, där keywords omvandlas med TF-IDF.

Båda modellerna använder cosine similarity för att jämföra användarens profil med filmerna i datasetet.

Användaren väljer fem favoritfilmer i en Streamlit-applikation och får sedan tio filmrekommendationer.

Projektet använder data från **The Movies Dataset** på Kaggle.

---

## Gruppmedlemmar

- Elias Stålhjärta
- Ida Claesson
- Azar Mohseni
- Tobias Uppenberg

---

## Installation

### 1. Klona repositoryt

```bash
git clone https://github.com/eliasstalfreelancer/grupparbete_filmrekomendation.git
```

Gå sedan in i projektmappen:

```bash
cd grupparbete_filmrekomendation
```

### 2. Skapa en virtuell Python-miljö

På Windows:

```bash
py -m venv .venv
```

Aktivera miljön:

```bash
.\.venv\Scripts\activate
```

### 3. Installera dependencies

```bash
pip install -r requirements.txt
```

### 4. Data

Projektet använder data från:

**The Movies Dataset**  
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

Datafilerna ska ligga i mappen:

```text
Kaggle_Movie_Data/
```

Applikationen använder bland annat:

```text
Kaggle_Movie_Data/movies_clean.csv
Kaggle_Movie_Data/keywords.csv
```

---

## Användning

Starta Streamlit-applikationen från projektmappen:

```bash
streamlit run streamlit_app.py
```

Applikationen öppnas därefter i webbläsaren.

### Så använder du applikationen

1. Sök efter och välj fem olika favoritfilmer.
2. Välj vilken rekommendationsmodell som ska användas:
   - **Baseline (genre)**
   - **Huvudmodell (genre + keywords)**
3. Klicka på **Hämta rekommendationer**.
4. Systemet visar de tio filmer som modellen bedömer passar bäst utifrån de valda filmerna.

### Modeller

**Baseline-modellen** använder endast filmernas genrer för att skapa en användarprofil och jämför sedan denna profil med andra filmer med cosine similarity.

**Huvudmodellen** använder både genrer och keywords. Keywords omvandlas till numeriska features med TF-IDF innan de kombineras med genreinformationen.

Det gör det möjligt för huvudmodellen att använda mer detaljerad information om filmernas innehåll än baselinen.
