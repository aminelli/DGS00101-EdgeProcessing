1. Flood (segmentazione di aree allagate da dati Sentinel-1):
     1. Premessa: una particolare area di interesse (AOI) si è allagata in seguito a una alluvione.
     2. Dato in input: immagine SAR acquisita con la missione ESA Copernicus Sentinel-1. Livello di produzione 1B (GRD)
     3. Task edge:
         1. Import del dato, lettura metadati e geocoding su ellissoide -> pre-processing standard (in teoria fattibile anche su GPU)
         2. Segmentazione dell’area alluvionata -> AI
     4. Esempio di prodotto:  https://mapping.emergency.copernicus.eu/activations/EMSR865/    <- attenzione: come prodotto va inteso solo il contenuto vettoriale con i poligoni delle aree alluvionate esposto su mappa.




# Flood Delineation

Agisci come un team tecnico con competenze in telerivelamento SAR, machine learning geospaziale, python ed Edge AI.

Progetta e realizza una demo funzionante per identificare e mappare aree alluvionate.

I dati sorgenti saranno recuperati dal Copernicus Sentinel-1 e in particaolare l'attivazione di tipo EMSR.

L'obiettivo finale è ottenere un modello di AI addestrato centralmente e la generazione di modelli quantizzati/ottimizzati per il deployment su Edge AI.


## Obiettivo e Scenario

Una particolare area di interesse (AOI) si è allagata in seguito a una alluvione.

Come riferimento cartografico considera l'attivazione Copernicus EMSR865 <https://mapping.emergency.copernicus.eu/activations/EMSR865/>





----
----
----
# GEMINI

Agisci come un ingegnere di Machine Learning e remote sensing specializzato in dati geospaziali e SAR/ottici. Voglio creare una pipeline end-to-end in Python per addestrare, validare e valutare un modello di AI basato sui dati del servizio Copernicus Emergency Management Service (`mapping.emergency.copernicus.eu`) con attivazioni **EMSR** (Rapid Mapping).

### 1. Contesto e obiettivo del modello
- **Fenomeno target:** [es. mappatura estensione allagamenti / flood delineation]
- **Tipo di prodotto EMSR usato come target/etichetta:** [es. strati vettoriali/raster di Delineation - DEL, distinguendo area allagata, acqua permanente e area valida non allagata]
- **Sensore di input osservativo:** [es. ESA Copernicus Sentinel-1 Level-1 GRD in polarizzazione VV/VH, pre- e post-evento]
- **Dati ausiliari opzionali:** [es. DEM / pendenza, land cover, maschera acqua permanente]
- **Task ML/DL:** [es. segmentazione semantica su patch georeferenziate]

### 2. Requisiti tecnici e stack
- Linguaggio e librerie principali: Python 3.x, `PyTorch`, `rasterio`, `geopandas`, `shapely`, `numpy`, `scikit-learn` (indica eventuali altre librerie utili e motivane la scelta).
- Gestione esplicita di **CRS**, **allineamento pixel/griglia**, **valori No-Data** (`ignore_index` nella loss) e **sbilanciamento delle classi** (es. combinazione di Dice/IoU loss e Cross-Entropy/Focal loss).
- **Prevenzione del data leakage:** split di training, validation e test **per attivazione EMSR o area geografica non contigua**, mai casuale a livello di pixel o patch adiacenti.

### 3. Cosa voglio che tu produca
1. **Architettura del progetto:** struttura delle cartelle e dei moduli (ingestione/preparazione dati EMSR + input satellitari, rasterizzazione e creazione patch, dataset/dataloader, modello, training loop, valutazione e inferenza con salvataggio GeoTIFF georeferenziato).
2. **Codice Python modulare e ben commentato**, pronto da eseguire o adattare, includendo:
   - caricamento e allineamento spaziale dei dati EMSR (EMSR) e degli input raster;
   - generazione di patch e gestione del class imbalance / no-data;
   - definizione del modello (es. U-Net o architettura equivalente adatta al task) e della funzione di loss;
   - loop di addestramento con validazione, salvataggio del miglior checkpoint e calcolo di **IoU per classe**, **F1-score**, **Precision** e **Recall**;
   - script di inferenza su una nuova scena/patch con ricostruzione georeferenziata.
3. **Spiegazione delle scelte chiave** e delle principali insidie legate alle attivazioni EMSR (eterogeneità dei sensori, qualità/copertura delle etichette, risoluzione spaziale) con relative contromisure pratiche.

Se qualche dettaglio sopra è ambiguo o manca per scrivere codice solido, fai le tue assunzioni in modo esplicito e procedi senza bloccarti.

----
----
----

# Chat GPT:

Ti propongo un prompt strutturato per chiedere a un'AI (ChatGPT, Claude, Gemini o altro coding agent) di progettare e sviluppare un sistema di **Intelligenza Artificiale in Python basato sui dati del Copernicus Emergency Management Service (CEMS), limitatamente alle attivazioni EMSR (Rapid Mapping)**.

L'obiettivo è ottenere un progetto software completo, non un semplice notebook dimostrativo.

## **Prompt – Sviluppo modello AI basato su Copernicus EMSR**

# Ruolo

Agisci come un Senior AI Engineer, Machine Learning Engineer e Geospatial Data Scientist, con competenze approfondite in:

- Python e Deep Learning.
- Machine Learning e Computer Vision.
- Remote Sensing e immagini satellitari SAR.
- Copernicus Emergency Management Service (CEMS).
- Copernicus Sentinel-1 e Sentinel-2.
- Elaborazione di dati raster e vettoriali geospaziali.
- GIS, GDAL, Rasterio, GeoPandas e PostGIS.
- MLOps e deployment di modelli AI.

# 1. Obiettivo del progetto

Realizza un sistema di Intelligenza Artificiale sviluppato interamente in Python, in grado di acquisire, elaborare e utilizzare i dati pubblicati dal servizio:

[https://mapping.emergency.copernicus.eu/](https://mapping.emergency.copernicus.eu/)

Il sistema deve basarsi esclusivamente sulle attivazioni di tipo **EMSR (Emergency Management Service – Rapid Mapping)**.

L'obiettivo è costruire un dataset geospaziale utilizzabile per addestrare uno o più modelli di Machine Learning e Deep Learning, capaci di identificare automaticamente fenomeni naturali e stimarne gli effetti sul territorio attraverso l'analisi delle immagini satellitari.

Non limitarti a descrivere un'architettura teorica: sviluppa un progetto software concretamente implementabile, modulare, documentato e corredato di codice Python eseguibile.

# 2. Acquisizione dei dati Copernicus EMS

Implementa un componente Python per interrogare automaticamente le API di Copernicus EMS Mapping.

Documentazione di riferimento:

[https://mapping.emergency.copernicus.eu/](https://mapping.emergency.copernicus.eu/)

Il sistema deve:

1. Interrogare le API REST per recuperare il catalogo delle attivazioni.
2. Filtrare esclusivamente le attivazioni con identificativo EMSR.
3. Recuperare i metadati delle singole attivazioni.
4. Recuperare l'elenco dei prodotti cartografici disponibili.
5. Identificare i prodotti associati a ciascuna area di interesse (AOI).
6. Recuperare i collegamenti ai file scaricabili.
7. Scaricare automaticamente i prodotti disponibili.
8. Gestire paginazione, timeout, errori HTTP, retry e download interrotti.
9. Registrare i metadati di ogni acquisizione.
10. Evitare duplicazioni attraverso identificativi univoci e checksum.

Considera i diversi prodotti disponibili nelle attivazioni Rapid Mapping:

- First Estimate Product (FEP).
- Delineation (DEL).
- Grading (GRA).
- Reference (REF).
- Situational Reporting (SR).

Non presumere che ogni attivazione contenga tutti i prodotti. Rileva dinamicamente quelli effettivamente disponibili.

# 3. Acquisizione delle immagini satellitari

I prodotti cartografici CEMS devono essere associati, ove possibile, alle immagini satellitari originali utilizzate per l'analisi dell'evento.

Utilizza principalmente:

- Sentinel-1 SAR.
- Sentinel-1 Level-1 GRD.
- Sentinel-1 VV e VH.
- Sentinel-2 MSI, quando disponibile.
- Altre fonti satellitari solo se necessarie e compatibili con il progetto.

Utilizza Copernicus Data Space Ecosystem per il recupero delle immagini originali:

[https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/)

Per ogni attivazione EMSR identifica:

- Area geografica interessata.
- Data e ora dell'evento.
- Data e ora delle acquisizioni satellitari.
- Sensore utilizzato.
- Modalità di acquisizione.
- Polarizzazione.
- Risoluzione spaziale.
- Copertura geografica.

Associa le immagini pre-evento e post-evento ai prodotti cartografici CEMS, verificando la compatibilità temporale e geografica.

Non inventare associazioni tra immagini e prodotti cartografici quando non esistono sufficienti informazioni per verificarle.

# 4. Data Engineering e preprocessing

Realizza una pipeline di preprocessing geospaziale utilizzando:

- GDAL.
- Rasterio.
- GeoPandas.
- Shapely.
- PyProj.
- NumPy.
- Pandas.

La pipeline deve comprendere:

1. Lettura di immagini GeoTIFF.
2. Lettura di GeoJSON, GeoPackage e Shapefile.
3. Verifica e uniformazione dei sistemi di riferimento geografico.
4. Georeferenziazione e allineamento delle immagini.
5. Gestione della risoluzione spaziale.
6. Preprocessing radiometrico delle immagini SAR.
7. Gestione dei valori NoData.
8. Normalizzazione dei dati.
9. Creazione di immagini multicanale.
10. Rasterizzazione dei prodotti cartografici.
11. Creazione delle Ground Truth.
12. Suddivisione delle immagini in patch utilizzabili per il training.

Mantieni separati i dati originali, quelli preprocessati e quelli utilizzati per l'addestramento.

# 5. Costruzione del dataset AI

Realizza un sistema che trasformi automaticamente i prodotti cartografici EMSR in etichette utilizzabili per il Deep Learning.

Il dataset dovrà contenere:

- Immagini satellitari.
- Maschere di segmentazione.
- Classificazioni territoriali.
- Metadati dell'attivazione.
- Metadati delle immagini.
- Identificativi dei prodotti CEMS.
- Informazioni sulla provenienza dei dati.

Implementa una struttura di dataset organizzata per:

- Training.
- Validation.
- Test.

La suddivisione deve avvenire per attivazione o evento geografico, evitando contaminazioni tra immagini appartenenti allo stesso evento.

Gestisci esplicitamente le aree non classificate, le aree NoData e le eventuali incertezze delle etichette.

# 6. Sviluppo dei modelli AI

Progetta un sistema modulare che consenta di addestrare modelli differenti in funzione della tipologia di evento.

Valuta le seguenti applicazioni:

| Fenomeno   | Modello AI                        | Obiettivo                                                |
| ---------- | --------------------------------- | -------------------------------------------------------- |
| Flood      | U-Net / DeepLabV3                 | Identificazione delle aree allagate                      |

Non creare un unico modello generalista se la natura dei dati richiede modelli specializzati.

Separa le architetture in moduli indipendenti.

# 7. Framework Deep Learning

Utilizza principalmente PyTorch.

Valuta:

- U-Net.
- DeepLabV3.
- ResNet.
- EfficientNet.
- Modelli di segmentazione semantica.
- Modelli di change detection.

Per ogni modello implementa:

1. Definizione dell'architettura.
2. Configurazione degli input.
3. DataLoader.
4. Data augmentation.
5. Loss Function.
6. Optimizer.
7. Training loop.
8. Validation loop.
9. Early stopping.
10. Checkpointing.
11. Salvataggio dei pesi.
12. Ripristino del modello.

Prevedi la possibilità di utilizzare GPU NVIDIA tramite CUDA, mantenendo la compatibilità con CPU.

# 8. Valutazione dei modelli

Implementa metriche di valutazione appropriate per ciascuna applicazione.

Per i modelli di segmentazione considera:

- Precision.
- Recall.
- F1-score.
- Intersection over Union (IoU).
- Dice Coefficient.
- Confusion Matrix.

Per i modelli di Object Detection considera anche:

- Mean Average Precision (mAP).
- Precision e Recall per classe.

Genera report di valutazione, grafici e mappe geospaziali delle predizioni.

La valutazione deve essere effettuata anche per singola attivazione EMSR, in modo da verificare la capacità di generalizzazione su eventi non utilizzati nel training.

# 9. Inference e produzione dei risultati

Il sistema deve poter ricevere nuove immagini satellitari e utilizzare i modelli addestrati per produrre automaticamente predizioni.

Gli output devono essere esportabili in:

- GeoTIFF.
- GeoJSON.
- GeoPackage.
- PNG.
- JSON.

Le predizioni devono mantenere le informazioni geografiche originali.

Prevedi inoltre la possibilità di visualizzare i risultati mediante QGIS o un'interfaccia WebGIS.

# 10. Architettura software

Organizza il progetto in moduli Python indipendenti.

Proponi una struttura simile alla seguente:

copernicus-ai/\
│\
├── config/\
│   ├── settings.yaml\
│   └── models.yaml\
│\
├── data/\
│   ├── raw/\
│   ├── processed/\
│   ├── datasets/\
│   └── predictions/\
│\
├── src/\
│   ├── ingestion/\
│   │   ├── cems_api.py\
│   │   ├── activations.py\
│   │   ├── products.py\
│   │   └── sentinel.py\
│   │\
│   ├── preprocessing/\
│   │   ├── raster.py\
│   │   ├── vector.py\
│   │   ├── sar.py\
│   │   └── masks.py\
│   │\
│   ├── dataset/\
│   │   ├── builder.py\
│   │   ├── patches.py\
│   │   └── dataloader.py\
│   │\
│   ├── models/\
│   │   ├── flood/\
│   │   ├── wildfire/\
│   │   ├── earthquake/\
│   │   └── landslide/\
│   │\
│   ├── training/\
│   ├── evaluation/\
│   ├── inference/\
│   └── api/\
│\
├── notebooks/\
├── tests/\
├── requirements.txt\
└── README.md

Puoi modificare la struttura se ritieni che esista un'organizzazione migliore.

# 11. MLOps

Integra, ove opportuno:

- MLflow per experiment tracking e model registry.
- DVC per versionamento dei dataset.
- Apache Airflow per orchestrazione delle pipeline.
- FastAPI per esporre i servizi di predizione.
- Docker per la containerizzazione.

Prevedi la tracciabilità completa di:

- Dati utilizzati.
- Attivazioni EMSR di origine.
- Prodotti cartografici.
- Versioni del dataset.
- Parametri di training.
- Versioni dei modelli.
- Metriche di valutazione.

# 12. Vincoli e requisiti progettuali

Rispetta rigorosamente i seguenti requisiti:

1. Utilizza esclusivamente attivazioni EMSR di Copernicus EMS Mapping come fonte dei prodotti cartografici di riferimento.
2. Non inventare endpoint API, nomi di campi, formati o attributi.
3. Verifica gli endpoint sulla documentazione ufficiale.
4. Distingui i prodotti cartografici dalle immagini satellitari originali.
5. Non considerare automaticamente ogni prodotto DEL o GRA come una Ground Truth perfetta.
6. Gestisci le differenze tra sensori, risoluzioni e date di acquisizione.
7. Non effettuare training su dataset non verificati.
8. Documenta le limitazioni dei dati disponibili.
9. Mantieni il codice modulare, riutilizzabile e configurabile.
10. Non utilizzare dati sintetici per dichiarare prestazioni reali del modello.

# 13. Modalità di sviluppo richieste

Procedi per fasi.

**FASE 1 – Analisi preliminare**

Analizza le API CEMS, i prodotti EMSR, le fonti satellitari e la disponibilità effettiva dei dataset.

Produci un documento tecnico contenente:

- Architettura proposta.
- Fonti dati.
- API da utilizzare.
- Tipologie di prodotti.
- Modelli AI candidati.
- Dipendenze software.
- Criticità tecniche.
- Requisiti hardware.

Non iniziare a scrivere tutto il codice prima di aver completato questa analisi.

**FASE 2 – Progettazione**

Definisci la struttura del progetto, i moduli Python, le interfacce e i flussi di elaborazione.

**FASE 3 – Data Ingestion**

Implementa il recupero delle attivazioni EMSR e dei relativi prodotti.

**FASE 4 – Dataset Engineering**

Implementa preprocessing, associazione delle immagini satellitari, Ground Truth e creazione del dataset.

**FASE 5 – Model Development**

Implementa i modelli AI e le pipeline di training.

**FASE 6 – Evaluation**

Implementa la validazione e la generazione dei report.

**FASE 7 – Deployment**

Implementa il servizio di inference e l'esportazione dei risultati geospaziali.

Per ogni fase:

- Fornisci codice Python completo.
- Indica i file da creare.
- Spiega le dipendenze.
- Fornisci i comandi di installazione.
- Fornisci esempi di esecuzione.
- Gestisci errori ed eccezioni.
- Verifica la coerenza con le fasi precedenti.

Non utilizzare pseudocodice al posto del codice implementativo, salvo quando una funzionalità dipende da informazioni o servizi non ancora disponibili.

Se mancano informazioni, dichiaralo esplicitamente e chiedi i dati necessari. Non inventare dettagli tecnici.

## **Alcune scelte progettuali che ho inserito volutamente**

Ci sono quattro aspetti importanti del prompt.

| Scelta                  | Motivazione                                                                       |
| ----------------------- | --------------------------------------------------------------------------------- |
| Solo EMSR               | Evita di mescolare attivazioni Rapid Mapping con altri servizi CEMS               |
| Ground Truth da DEL/GRA | Consente di utilizzare i prodotti cartografici come etichette, quando compatibili |
| Modelli specializzati   | Le alluvioni e i terremoti richiedono approcci differenti                         |
| Separazione per evento  | Evita data leakage e valutazioni ingannevoli                                      |

## **Una precisazione architetturale**

Non darei per scontato che tutti i fenomeni EMSR possano essere addestrati con lo stesso tipo di input.

Ad esempio:

- **Flood:** Sentinel-1 GRD pre/post evento è un ottimo punto di partenza.
- **Wildfire:** Sentinel-2 e altri dati ottici multispettrali possono essere più appropriati per la mappatura delle superfici bruciate.
- **Earthquake:** per la classificazione dei danni agli edifici sono generalmente necessarie immagini ad alta risoluzione.
- **Landslide:** possono essere utili immagini ottiche, SAR e modelli digitali del terreno.

Il modello deve quindi essere progettato in funzione dei dati realmente disponibili per ogni attivazione.

**Il mio consiglio:** utilizza questo prompt inizialmente per ottenere la FASE 1, cioè l'analisi di fattibilità e la progettazione. Solo dopo aver verificato quali attivazioni EMSR e quali prodotti sono effettivamente utilizzabili, passa alla generazione del codice. Questo evita di costruire una pipeline AI su un dataset che potrebbe non essere sufficientemente completo o omogeneo.

----
----
----

## CLAUDE

# Prompt proposto

Puoi copiarlo così com'è e sostituire i campi tra `[parentesi quadre]`.

````text
Agisci come un ingegnere senior di machine learning per il telerilevamento (Earth Observation) con esperienza in PyTorch, rasterio, geopandas e dati Copernicus.

## OBIETTIVO
Realizza una pipeline Python completa e riproducibile che addestri un modello di deep learning di [segmentazione semantica] per [mappatura delle alluvioni] usando come etichette i dati vettoriali delle attivazioni Copernicus Emergency Management Service (CEMS) Rapid Mapping con codice EMSR (mapping.emergency.copernicus.eu) e come input immagini [Sentinel-1 GRD (VV, VH) + DEM/pendenza].

## CONTESTO TECNICO
- Le attivazioni EMSR forniscono quasi solo vettori (Shapefile/GeoJSON) di tipo delineation/grading, NON le immagini di input: le immagini vanno recuperate separatamente da [Copernicus Data Space Ecosystem (STAC/OData)] e abbinate per area e data.
- I poligoni coprono solo l'AOI mappata dall'analista: fuori dall'AOI le etichette sono sconosciute e vanno trattate con una maschera "ignore" (valore 255), non come classe negativa.
- Ambiente di esecuzione: [GPU locale NVIDIA 12 GB / Azure ML / Colab], Python 3.11, [Linux].
- Lista di attivazioni da usare: [EMSR###, EMSR###, ...] (oppure: "proponi uno script che le selezioni da un file CSV che fornisco io con colonne: codice, data_evento, tipo_evento").

## PIPELINE RICHIESTA (moduli separati)
1. **Config**: file YAML con percorsi, lista attivazioni, dimensione tile, iperparametri, seed.
2. **Download etichette**: lettura dei vettori EMSR da una cartella locale di ZIP già scaricati (non inventare endpoint non documentati per il download dal portale CEMS; se serve un'API, indicami quale e come verificarla).
3. **Download immagini**: ricerca e scarico delle scene Sentinel-1 GRD più vicine temporalmente all'evento, con report dello scarto temporale per ogni attivazione e scarto dei casi oltre [24 ore].
4. **Preprocessing SAR**: orbit file, thermal noise removal, calibrazione σ⁰, terrain correction con DEM, conversione in dB (con pyroSAR o SNAP/GPT; indica chiaramente quale usi e come si installa).
5. **Rasterizzazione etichette** sulla griglia dell'immagine, con maschera ignore fuori AOI.
6. **Tiling** [256x256] con overlap opzionale, scarto delle tile senza dati o fuori AOI, salvataggio di un indice (CSV/Parquet) con attivazione, coordinate e data.
7. **Split** train/val/test **per attivazione/area geografica**, mai casuale per tile, per evitare data leakage. Rendi lo split configurabile e salvato su file.
8. **Dataset e DataLoader** PyTorch con normalizzazione per canale (statistiche calcolate solo sul training set) e augmentation compatibili con il SAR (flip, rotazioni di 90°, crop; niente trasformazioni radiometriche aggressive).
9. **Modello**: [U-Net con encoder resnet34 tramite segmentation_models_pytorch], input a [4] canali, output binario. Rendi l'architettura facilmente sostituibile.
10. **Training**: loss Dice + BCE con supporto a ignore_index, AdamW, scheduler, mixed precision, early stopping, checkpoint del modello migliore, logging su [TensorBoard/MLflow].
11. **Valutazione** sul test set: IoU, F1/Dice, precisione e recall sulla classe acqua, più analisi per tipo di terreno o per attivazione. Salva esempi qualitativi (immagine, etichetta, predizione).
12. **Inferenza** su una nuova scena: sliding window con overlap e blending, soglia configurabile, esportazione in GeoTIFF e vettorizzazione in GeoJSON.

## VINCOLI DI QUALITÀ
- Codice modulare, con type hints, docstring e logging; niente codice monolitico in un unico notebook.
- Struttura del progetto ad albero, `requirements.txt` (o `environment.yml`) con versioni indicate.
- Riproducibilità: seed fissati, configurazione salvata insieme a ogni esperimento.
- Gestione degli errori e dei casi limite (scene mancanti, CRS diversi, tile vuote, classi sbilanciate).
- Non inventare API, nomi di campi degli shapefile o funzioni di librerie: se non sei sicuro di un dettaglio, dichiaralo esplicitamente e indica come verificarlo. Per gli attributi degli shapefile EMSR scrivi codice che ispeziona le colonne disponibili invece di assumerle.

## FORMATO DELLA RISPOSTA
1. Prima un breve piano con l'albero delle cartelle e le assunzioni che stai facendo.
2. Poi il codice file per file, ciascuno in un blocco separato con il percorso come titolo.
3. Infine: comandi per eseguire la pipeline passo passo, limiti noti del metodo e possibili miglioramenti.

Procedi per fasi: consegna prima i moduli 1-6 (dati), poi attendi la mia conferma per proseguire con i moduli 7-12 (modello e valutazione).
````

## Come usarlo bene

- **Procedi a fasi.** Una pipeline così lunga in un'unica risposta tende a produrre codice superficiale. L'ultima riga del prompt serve proprio a spezzarla.
- **Restringi il caso.** Più sei specifico su evento (alluvioni, incendi), sensore e ambiente, migliore è il codice. Per gli incendi, ad esempio, sostituisci Sentinel-1 con Sentinel-2 e aggiungi gli indici NBR/dNBR.
- **Fornisci un esempio reale.** Se incolli l'elenco delle colonne di uno shapefile EMSR scaricato (`gdf.columns`), l'AI scrive codice molto più preciso.
- **Verifica sempre i dettagli.** Chiedi di testare il codice su una sola attivazione prima di scalare. Le API e le versioni delle librerie cambiano, e anche un buon modello può sbagliare un parametro.
- **Alternativa più rapida.** Se il tuo obiettivo è soprattutto imparare, puoi sostituire i punti 2-6 con "usa il dataset MMFlood già preparato" e concentrarti sul modello.

Se vuoi, posso adattarlo a un caso specifico (alluvioni, incendi o danni agli edifici) e al tuo ambiente, per esempio Azure ML o una GPU locale.