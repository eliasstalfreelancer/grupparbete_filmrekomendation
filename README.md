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

---

## Modeller

### Baseline

Baseline-modellen använder endast filmernas genrer.

Genrerna konverteras till binära features och de fem valda filmerna kombineras till en användarprofil. Cosine similarity används sedan för att jämföra användarprofilen med övriga filmer och rangordna rekommendationerna.

### Huvudmodell

Huvudmodellen använder både genrer och keywords.

Keywords omvandlas till numeriska features med **TF-IDF** innan de kombineras med genreinformationen.

Det gör det möjligt för huvudmodellen att använda mer detaljerad information om filmernas innehåll än baselinen.

---

## Utvärdering

Modellerna utvärderades med historiska användarbetyg.

Filmer med ett betyg på minst **4.0** behandlades som filmer användaren tycker om.

För varje testanvändare användes fem gillade filmer som input till rekommendationsmodellen. Resterande gillade filmer gömdes och användes sedan som facit för att kontrollera hur många av dem modellerna lyckades rekommendera.

Totalt utvärderades modellerna på **625 användare**.

Följande mått användes:

- **Hit Rate@10** – om minst en relevant film förekommer bland de tio rekommendationerna.
- **Recall@10** – hur stor andel av de gömda relevanta filmerna som återfinns bland de tio rekommendationerna.
- **NDCG@10** – tar även hänsyn till hur högt de relevanta filmerna placeras i rekommendationslistan.

### Resultat

| Modell | Hit Rate@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Baseline – Genre | 0.182 | 0.004 | 0.025 |
| Huvudmodell – Genre + Keywords | 0.232 | 0.007 | 0.039 |

Huvudmodellen fick högre resultat än baselinen på samtliga tre utvärderingsmått.

Hit Rate@10 ökade från **0.182 till 0.232**, vilket innebär att huvudmodellen oftare lyckades hitta minst en av användarens gömda relevanta filmer.

Recall@10 ökade från **0.004 till 0.007** och NDCG@10 ökade från **0.025 till 0.039**.

Resultaten tyder därför på att keywords tillför användbar information utöver enbart genre i den här utvärderingen.

Resultaten visar däremot inte hur verkliga användare skulle uppleva rekommendationerna på en riktig streamingplattform.

---

## Begränsningar

Alla filmer i datasetet har inte keywords.

Under utvärderingen upptäcktes att filmer utan keywords påverkade huvudmodellens ranking negativt. Huvudmodellen filtrerar därför bort filmer som saknar keywords från sina slutliga rekommendationer.

Det innebär att:

- baseline-modellen kan rekommendera filmer utan keywords
- huvudmodellen endast rekommenderar filmer där keyword-information finns

Detta är en begränsning i den nuvarande lösningen och något som skulle kunna förbättras om projektet utvecklades vidare.
