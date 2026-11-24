# Progetto Flood Delineation


## Ruolo

Agisci come un Senior AI Engineer, Machine Learning Engineer e Geospatial Data Scientist, con competenze approfondite in:

- Python e Deep Learning.
- Machine Learning e Computer Vision.
- Remote Sensing e immagini satellitari SAR.
- Copernicus Emergency Management Service (CEMS).
- Copernicus Sentinel-1 e Sentinel-2.
- Elaborazione di dati raster e vettoriali geospaziali.
- GIS, GDAL, Rasterio, GeoPandas e PostGIS.
- MLOps e deployment di modelli AI.

Voglio creare una pipeline end-to-end in Python per addestrare, validare e valutare un modello di AI basato sui dati del servizio Copernicus Emergency Management Service (`mapping.emergency.copernicus.eu`) con attivazioni **EMSR** (Rapid Mapping).


## Obiettivo del progetto

Realizza un sistema di Intelligenza Artificiale sviluppato interamente in Python, in grado di acquisire, elaborare e utilizzare i dati pubblicati dal servizio:

[https://mapping.emergency.copernicus.eu/](https://mapping.emergency.copernicus.eu/)

Il sistema deve basarsi esclusivamente sulle attivazioni di tipo **EMSR (Emergency Management Service – Rapid Mapping)**.

L'obiettivo è costruire un dataset geospaziale utilizzabile per addestrare uno o più modelli di Machine Learning e Deep Learning, capaci di identificare automaticamente fenomeni naturali e stimarne gli effetti sul territorio attraverso l'analisi delle immagini satellitari.

Non limitarti a descrivere un'architettura teorica: sviluppa un progetto software concretamente implementabile, modulare, documentato e corredato di codice Python eseguibile.

Realizza una pipeline Python completa e riproducibile che addestri un modello di deep learning di [segmentazione semantica] per [mappatura delle alluvioni] usando come etichette i dati vettoriali delle attivazioni Copernicus Emergency Management Service (CEMS) Rapid Mapping con codice EMSR (mapping.emergency.copernicus.eu) e come input immagini [Sentinel-1 GRD (VV, VH) + DEM/pendenza].

## Contesto e obiettivo del modello di AI da generare
- **Fenomeno target:** [es. mappatura estensione allagamenti / flood delineation]
- **Tipo di prodotto EMSR usato come target/etichetta:** [es. strati vettoriali/raster di Delineation - DEL, distinguendo area allagata, acqua permanente e area valida non allagata]
- **Sensore di input osservativo:** [es. ESA Copernicus Sentinel-1 Level-1 GRD in polarizzazione VV/VH, pre- e post-evento]
- **Dati ausiliari opzionali:** [es. DEM / pendenza, land cover, maschera acqua permanente]
- **Task ML/DL:** [es. segmentazione semantica su patch georeferenziate]

## Acquisizione dei dati Copernicus EMS

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

- Delineation (DEL).
- Grading (GRA).

Non presumere che ogni attivazione contenga tutti i prodotti. 
Rileva dinamicamente quelli effettivamente disponibili.
In particolare usa l'attivazione EMSR865 per limitare i dati da richiedere per questo demo.

Tieni conto del fatto che:

- Le attivazioni EMSR forniscono quasi solo vettori (Shapefile/GeoJSON) di tipo delineation/grading, NON le immagini di input: le immagini vanno recuperate separatamente da [Copernicus Data Space Ecosystem (STAC/OData)] e abbinate per area e data.
- I poligoni coprono solo l'AOI mappata dall'analista: fuori dall'AOI le etichette sono sconosciute e vanno trattate con una maschera "ignore" (valore 255), non come classe negativa.

## Acquisizione delle immagini satellitari

I prodotti cartografici CEMS devono essere associati, ove possibile, alle immagini satellitari originali utilizzate per l'analisi dell'evento.

Utilizza principalmente:

- Sentinel-1 Level-1 GRD.
- Sentinel-1 VV e VH.
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

## Data Engineering e preprocessing

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

## Costruzione del dataset AI

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

## Sviluppo dei modelli AI

Progetta un sistema modulare legato ai fenomeni alluvionali (Flood).

Valuta le seguenti indicazioni:

| Fenomeno   | Modello AI                        | Obiettivo                                                |
| ---------- | --------------------------------- | -------------------------------------------------------- |
| Flood      | U-Net / DeepLabV3                 | Identificazione delle aree allagate                      |

## Framework Deep Learning

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

## Valutazione dei modelli

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


## Inference e produzione dei risultati

Il sistema deve poter ricevere nuove immagini satellitari e utilizzare i modelli addestrati per produrre automaticamente predizioni.

Gli output devono essere esportabili in:

- GeoTIFF.
- GeoJSON.
- GeoPackage.
- PNG.
- JSON.

Le predizioni devono mantenere le informazioni geografiche originali.

Prevedi inoltre la possibilità di visualizzare i risultati mediante QGIS o un'interfaccia WebGIS.

## Architettura software

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


## MLOps

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

## Vincoli e requisiti progettuali

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
11. Tieni presente che si tratta di una demo per cui limitati al minimo indispensabile per scopi didattici.


## Modalità di sviluppo richieste

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

Se mancano informazioni, dichiaralo esplicitamente e chiedi i dati necessari. 

Non inventare dettagli tecnici.


## Note generali

Usa una nomenclatura inglese per il codice, script etc. 
Commenta in italiano il codice con un approccio documentale chiaro ed esplicativo sullo scope.

Genera un dictionary e documentazione in markdown di tutte le sigle che hai adottato con relativa spiegazione; 
fai lo stesso per i topics toccati di maggior interesse e sui dettagli tecnologici; questo file deve spiegare gli aspetti tecnologici (non è quindi la documentazione del progetto python)

Genera in markdown una documentazione di porogetto in italiano che sia dettagliata e contenente tutto il necessario (comandi, info utili etc.); questo deve fungere da documentazione di progetto.

Sii preciso, tecnico, e spiega bene i passaggi.

Se hai bisogno di altro per aspetti che non sono riuscito a descrivere o che ho descritto male, ponimi le domande prima di iniziare a generare il tutto.
