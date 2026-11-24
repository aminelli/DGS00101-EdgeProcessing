# Fase 2 — Progettazione

**Perimetro approvato:** solo EMSR865; demo didattica; modello di segmentazione Flood in PyTorch.  
**Decisione etichette:** poligoni `observedEventA` come positivi; background dentro l'AOI e fuori dai poligoni come negativo per la demo, marcato come assunzione non verificata. Fuori dall'AOI, NoData, pixel senza input valido e aree coperte solo da layer modellati restano ignore (255).  
**Limite valutazione:** un solo evento e una sola AOI; nessuna affermazione di generalizzazione a eventi distinti.

Questa fase definisce struttura, componenti, flussi, metadati e test. Non implementa ancora il codice.

## 1. Struttura proposta

```text
copernicus-flood-ai/
├── config/
│   └── settings.yaml
├── data/
│   ├── raw/
│   │   ├── cems/
│   │   └── sentinel1/
│   ├── processed/
│   ├── datasets/
│   ├── predictions/
│   └── manifests/
├── src/
│   └── copernicus_flood/
│       ├── __init__.py
│       ├── cli.py
│       ├── ingestion/
│       │   ├── cems_client.py
│       │   ├── cdse_stac_client.py
│       │   └── download_registry.py
│       ├── preprocessing/
│       │   ├── sar.py
│       │   ├── alignment.py
│       │   └── labels.py
│       ├── dataset/
│       │   ├── builder.py
│       │   ├── patches.py
│       │   └── torch_dataset.py
│       ├── models/
│       │   └── unet.py
│       ├── training/
│       │   └── train.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   └── report.py
│       └── inference/
│           └── predict.py
├── tests/
│   ├── fixtures/
│   │   └── emsr865_response.json
│   ├── test_cems_client.py
│   ├── test_labels.py
│   ├── test_split.py
│   └── test_metrics.py
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

I dati scaricati e i pesi dei modelli non vanno inseriti nel controllo versione. La fixture di test contiene solo la risposta pubblica EMSR865 strettamente necessaria ai test; non contiene credenziali o dati personali. I dati grezzi scaricati restano separati dai raster preprocessati e dalle patch.

## 2. Componenti e interfacce

### Ingestion

- **`cems_client.py`**
  - Interroga esclusivamente la route pubblica verificata, con `code=EMSR865`.
  - Valida la forma generale della risposta, seleziona l'unica attivazione attesa e identifica dinamicamente prodotti, layer, versioni e `downloadPath`.
  - Non codifica a priori la presenza di GRA e non usa link ricostruiti.
  - Interfaccia logica: `get_activation(code)`, `get_products(activation)`, `get_product_layers(product)`.
- **`cdse_stac_client.py`**
  - Cerca nella collezione STAC verificata `sentinel-1-grd` per bbox/tempi e poi applica verifica geometrica usando il poligono AOI effettivo.
  - Filtra in modo esplicito polarizzazioni, modalità, date e footprint; restituisce gli item e gli asset VV/VH effettivamente disponibili.
  - Non dichiara una corrispondenza CEMS-CDSE solo perché l'item cade nella stessa giornata.
- **`download_registry.py`**
  - Mantiene un registro SQLite locale (libreria standard Python) di richieste, risposte, download, hash, URL, timestamp e stato.
  - Salta un download già completato solo quando il file locale supera i controlli disponibili; riprende o ripete una richiesta in base alle capacità HTTP verificate.
  - I test usano fixture e trasporto HTTP simulato, senza interrogare o riscaricare servizi reali.

### Preprocessing e dataset

- **`sar.py`**: legge gli asset COG selezionati, controlla valori NoData e applica una trasformazione radiometrica/configurata dopo aver verificato il contenuto e la procedura applicabile. Registra parametri e unità; nessuna normalizzazione nascosta. Baseline iniziale: VV/VH della scena post-evento associata; le scene pre-evento non entrano nello stack del modello finché co-registrazione e compatibilità orbitale non siano validate. DEM/pendenza restano opzionali e fuori dal baseline iniziale.
- **`alignment.py`**: imposta una griglia di lavoro comune, CRS proiettato adatto alla AOI, trasformazione e risoluzione esplicite; verifica le intersezioni e le porzioni mancanti.
- **`labels.py`**: usa `observedEventA` per la maschera primaria. I poligoni positivi diventano classe 1; background interno all'AOI e fuori dai poligoni diventa classe 0, esplicitamente etichettato come assunzione didattica non verificata; esterno AOI e NoData diventano 255. Le geometrie coperte esclusivamente da `modelledEventA` o `maximumFloodExtentA` diventano 255, evitando di trattarle per errore come non allagate. Questi layer restano separati e non sono fusi implicitamente con le etichette osservate.
- **`builder.py`**: crea raster di input e maschere allineate, un manifesto di provenienza per immagine/prodotto e controlli di validità per ogni output.
- **`patches.py`**: genera patch georeferenziate solo da pixel coperti da input valido. Registra coordinate/trasformazione, quota di pixel ignorati e distribuzione delle classi.
- **`torch_dataset.py`**: carica tensor input e target; converte 255 in `ignore_index` coerente con la loss e con le metriche.

### Modello, training e output

- **`unet.py`**: U-Net compatta con numero di canali in ingresso derivato dalla configurazione, una classe di output Flood e supporto CPU/CUDA.
- **`train.py`**: training e validation riproducibili, seed, configurazione salvata, loss compatibile con ignore 255, checkpoint miglior modello e ripristino.
- **`metrics.py`**: confusion matrix e metriche binarie calcolate solo sui pixel validi; gestione esplicita di divisioni per zero.
- **`report.py`**: report per split e mappe di confronto; tutte le metriche vengono etichettate come esplorative sul singolo evento EMSR865.
- **`predict.py`**: inferenza su input compatibile, output GeoTIFF con CRS/trasformazione e manifesto dei parametri del modello.
- **`cli.py`**: comandi separati per acquisire, costruire dataset, addestrare, valutare e inferire; ogni fase fallisce in modo esplicito se prerequisiti o file mancano.

MLflow, DVC, Airflow, FastAPI e Docker sono esclusi dal disegno iniziale: non sono necessari per la demo locale riproducibile.

## 3. Flusso di elaborazione

1. Leggere la risposta EMSR865 dalla cache locale, se presente e valida; altrimenti richiederla e registrarla.
2. Selezionare le versioni DEL effettivamente completate e salvare i relativi archivi/livelli senza sovrascrivere versioni differenti.
3. Eseguire ricerche CDSE STAC per periodi pre/post; salvare la risposta e validare le scene per footprint e metadati. Il baseline del modello usa VV/VH post-evento; l'input pre-evento rimane una possibile estensione, non un requisito implicito.
4. Produrre un rapporto di associazione che distingua `verified`, `candidate` e `rejected`; richiedere verifica manuale in caso di ambiguità.
5. Leggere VV/VH, applicare preprocessing documentato e riallineare all'area effettivamente coperta.
6. Rasterizzare l'AOI, `observedEventA` e le geometrie modellate da escludere; assegnare classi 1/0/255 secondo l'assunzione approvata.
7. Generare patch e split spaziali deterministici, escludendo sovrapposizioni e aree di guardia tra split.
8. Addestrare U-Net solo su patch con quantità configurabile di pixel validi e scene in stato `verified`; registrare configurazione, assunzione sui negativi e checkpoint.
9. Valutare sui blocchi spaziali tenuti da parte, riportando che tutti gli split appartengono allo stesso evento.
10. Eseguire inferenza e salvare GeoTIFF, report e manifesti.

## 4. Contratto dati e tracciabilità

Ogni record del manifesto del dataset deve contenere almeno:

- `activation_code`, `aoi_name`, `aoi_number`;
- `product_id`, `product_type`, `product_version`, `product_status`, `layer_name`;
- URL di origine, data di acquisizione, data di download, checksum locale;
- STAC item ID, collection, datetime, platform, orbit, polarizzazioni, asset href e checksum quando presenti;
- CRS, trasformazione, dimensioni, risoluzione, NoData e parametri SAR;
- classe positiva e provenienza della maschera;
- dichiarazione `negative_label_assumption: aoi_background_is_negative`;
- dimensione patch, coordinate/indice del blocco e split assegnato;
- versione della configurazione e seed.

La risposta CEMS/STAC originale e il manifesto vanno conservati per consentire audit e riproduzione. Valori mancanti restano null/assenti con motivazione; non si completano con valori dedotti.

## 5. Split e valutazione

Poiché EMSR865 contiene una sola AOI/evento, non è possibile assegnare eventi distinti ai tre split. Il disegno adotta quindi **blocchi spaziali disgiunti con fascia di esclusione**, assegnati deterministicamente, per evitare che patch sovrapposte o contigue finiscano in training e test. Tutti gli split restano dello stesso evento; la metrica di test misura soltanto trasferimento spaziale interno a EMSR865 e non generalizzazione temporale o geografica a nuovi eventi.

Le proporzioni e la dimensione dei blocchi non vengono fissate prima di conoscere area valida, patch utilizzabili e distribuzione delle classi. La fase di costruzione dataset deve segnalare se uno split non contiene esempi validi di entrambe le classi; in quel caso non produrre metriche fuorvianti né dichiarare un test valido.

## 6. Dipendenze previste

**Runtime:** `requests`, `rasterio`, `geopandas`, `shapely`, `pyproj`, `numpy`, `pandas`, `torch`, `PyYAML`.  
**Sviluppo/test:** `pytest`; `matplotlib` per grafici statici, se richiesti.

Le versioni e i comandi d'installazione saranno fissati nella fase d'implementazione dopo una verifica di compatibilità con l'ambiente. Non si installano dipendenze in questa fase di progettazione.

## 7. Strategia di test

- Client CEMS: parsing della fixture EMSR865, selezione prodotti/versioni e comportamento esplicito su risposta incompleta o campi inattesi.
- Registro: richiesta già completata non ripetuta; download incompleto non marcato come riuscito; checksum verificato quando disponibile.
- Etichette: poligono positivo, background approvato come assunzione, esterno AOI, NoData e ignore.
- Split: nessuna sovrapposizione di coordinate/patch tra train/validation/test e determinismo a parità di seed.
- Metriche: pixel 255 esclusi; casi senza predizioni o positivi gestiti senza NaN/fallback ingannevoli.
- I test possono usare piccoli raster sintetici esclusivamente per verificare il software; non entrano nel dataset né nei risultati prestazionali del modello.

## 8. Vincoli per la fase 3

Prima di implementare l'ingestion, confermare i campi live rispetto al file OpenAPI locale e registrare fixture versionate; il servizio live ha già mostrato differenze di schema. Non iniziare alcun download massivo: limitare le richieste a EMSR865 e alle scene Sentinel-1 che intersecano davvero la AOI. Mantenere la distinzione tra dato osservato, dato modellato e ipotesi sulle etichette.

