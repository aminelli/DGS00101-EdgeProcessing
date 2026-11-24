# Fase 4 — Dataset Engineering

## Scopo e stato

Questa fase implementa i passaggi riutilizzabili per associare le scene, allineare VV/VH, rasterizzare le etichette CEMS e creare un manifesto di patch. Il confronto tra tabella CEMS `source`, `or_src_id` nei footprint `imageFootprintA`, orario UTC e geometrie STAC ha verificato cinque item Sentinel-1.

I dieci COG VV/VH verificati (circa 6,2 GB) sono stati acquisiti via S3 CDSE e controllati per dimensione e checksum. I TIFF Sentinel-1 espongono 210 GCP in EPSG:4326, non un CRS/affine standard; il preprocessing li riproietta nel CRS UTM stimato dall'AOI prima di mosaico, maschera ed estrazione delle patch. Sono stati generati dataset per i tre prodotti CEMS 2435, 2436 e 2437. I manifesti risultano `ready_for_training` perché ogni split spaziale contiene pixel positivi e negativi; lo sfondo negativo resta l'assunzione didattica dichiarata, non un'etichetta osservata verificata.

## File creati o modificati

- `src/copernicus_flood/preprocessing/raster.py`: usa CRS o GCP georeferenziati, riproietta VV/VH nel CRS di destinazione, mosaica sulla griglia VV, crea uno stack GeoTIFF a due bande `float32`, mantiene l'ampiezza grezza e propaga NoData.
- `src/copernicus_flood/preprocessing/labels.py`: legge i GeoJSON direttamente dagli ZIP CEMS; assegna `1` ai poligoni `Flooded area` di `observedEventA`, `0` al background interno all'AOI secondo l'assunzione didattica approvata, e `255` fuori dall'AOI, su NoData e dove i layer modellati sono l'unica evidenza. I layer modellati facoltativi non sono fusi con le etichette osservate.
- `src/copernicus_flood/dataset/patches.py`: genera patch a blocchi spaziali con fascia di separazione e registra conteggi per split, classi mancanti e split vuoti.
- `src/copernicus_flood/ingestion/cdse_client.py`: restringe i candidati STAC a un massimo di 90 secondi dall'orario CEMS e conserva gli href HTTPS OIDC e S3 quando disponibili.
- `src/copernicus_flood/ingestion/matching.py`: verifica gli item rispetto alla fonte Sentinel-1 e al relativo footprint CEMS. Un item deve coprire almeno il 90% della propria impronta nel footprint CEMS e l'unione degli item deve coprire almeno il 98% del footprint CEMS.
- `src/copernicus_flood/ingestion/cdse_auth.py` e `assets.py`: autenticazione client-credentials OIDC e download riprendibili dei soli asset VV/VH verificati; il token resta in memoria e le credenziali sono lette da `CDSE_CLIENT_ID` e `CDSE_CLIENT_SECRET`.
- `src/copernicus_flood/ingestion/s3_assets.py`: download S3 CDSE degli asset VV/VH verificati, con controllo delle dimensioni, trasferimenti multipart, SHA-256 e registrazione nel registro esistente. Le chiavi sono lette dall'ambiente o richieste in modo mascherato nel terminale.
- `src/copernicus_flood/ingestion/registry.py`: aggiunge gli header di autenticazione non persistenti e controlla la dimensione attesa dichiarata da STAC prima di registrare il file.
- `src/copernicus_flood/dataset/build.py`: CLI che verifica l'associazione e la registrazione dei COG, quindi produce stack, maschera, manifesto patch e riepilogo.
- `tests/`: test offline di parsing, matching, autenticazione, download, raster e maschere; i dati sintetici verificano il software e non sono input o risultati di training.

## Dipendenze

Dipendenze runtime geospaziali aggiunte in `requirements.txt`: GeoPandas, Rasterio e PyProj. Installazione (dalla cartella `copernicus-flood-ai`):

```powershell
python -m pip install -r requirements-dev.txt
```

Il comando usa anche le dipendenze di test già definite nel file dev.

## Esecuzione

I dataset reali sono stati creati dai tre prodotti DEL e dai soli item STAC associati e registrati:

```powershell
$env:PYTHONPATH = "src"
python -m copernicus_flood.dataset.build `
  --product-zip data/raw/cems/EMSR865/AOI01/EMSR865_AOI01_DEL_v2_id2435.zip `
  --vv data/raw/sentinel1/EMSR865/product_2435/<item-1>/VV.tif data/raw/sentinel1/EMSR865/product_2435/<item-2>/VV.tif `
  --vh data/raw/sentinel1/EMSR865/product_2435/<item-1>/VH.tif data/raw/sentinel1/EMSR865/product_2435/<item-2>/VH.tif `
  --stac-item-ids <item-1> <item-2> `
  --output-dir data/processed/EMSR865/DEL_v2_id2435
```

Parametri disponibili: `--patch-size` (default 256) e `--min-valid-fraction` (default 0.8). Il manifesto `patches.jsonl` riporta offset e dimensione delle patch, split assegnato, pixel validi/positivi/negativi e percorsi raster. `dataset_manifest.json` include le classi, l'associazione verificata, la provenienza e l'assunzione sui negativi.

## Demo web di inferenza

Dalla cartella del progetto, installare le dipendenze in `requirements.txt` e avviare:

```powershell
python -m streamlit run app.py --server.address 127.0.0.1
```

La pagina consente di scegliere un dataset elaborato, un checkpoint `checkpoint_best.pt` nella cartella `data/models/EMSR865`, uno split, una patch e una soglia. L'inferenza usa la CPU e riguarda **solo la patch selezionata**, non l'intera AOI. Mostra l'anteprima VV e la maschera prevista; permette di scaricare maschera e probabilità GeoTIFF, poligoni GeoJSON, GeoPackage se presenti e manifesto. I risultati sono salvati in sottocartelle univoche di `data/inference/EMSR865/demo`.

La demo richiede i dataset già elaborati, gli archivi CEMS e un checkpoint locale. Segnala se il checkpoint proviene da un dataset diverso. Il checkpoint `acceptance-smoke` è stato addestrato per una sola epoca: risultati esplorativi, non operativi; lo sfondo negativo resta un'assunzione didattica.

Il manifesto `data/manifests/emsr865_stac_candidates.json` riporta le associazioni verificate, i relativi source ID e le coperture dei footprint. Per gli asset verificati, gli URL HTTPS richiedono OIDC; gli URI S3 richiedono chiavi S3 CDSE. Il progetto non salva le credenziali in file o nel registro. Il builder individua nel registro l'URI di download utilizzato (S3 o HTTPS) e controlla dimensioni e SHA-256 prima del preprocessing.

Per acquisire i COG completi si può usare OIDC oppure S3. OIDC richiede un client OAuth creato secondo la [documentazione CDSE](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html) e le variabili locali `CDSE_CLIENT_ID` e `CDSE_CLIENT_SECRET`.

L'alternativa S3 usa l'endpoint ufficiale `https://eodata.dataspace.copernicus.eu` e chiavi generate dal [portale S3 CDSE](https://eodata-s3keysmanager.dataspace.copernicus.eu/), accessibile con un account CDSE registrato. Le chiavi S3 non sono le credenziali OIDC. Eseguire il comando in un terminale interattivo per inserirle tramite prompt mascherati, oppure configurare `CDSE_S3_ACCESS_KEY` e `CDSE_S3_SECRET_KEY` solo nell'ambiente locale; non inserirle nei file di configurazione o nel repository:

```powershell
python -m copernicus_flood.cli --config config/settings.yaml `
  --download-verified-assets --asset-transport s3
```

Il trasporto OIDC esistente rimane disponibile esplicitamente:

```powershell
python -m copernicus_flood.cli --config config/settings.yaml --download-verified-assets
```

Entrambi i trasporti scaricano VV e VH per i cinque item verificati (10 COG, circa 6,2 GB secondo le dimensioni STAC attuali), verificano le dimensioni, calcolano SHA-256 e registrano i download per evitare duplicazioni. S3 usa trasferimenti multipart concorrenti con retry gestiti da boto3. `--refresh` forza nuovi trasferimenti ed è sconsigliato per file di queste dimensioni.

## Verifiche svolte

- `python -m pytest -q`: **13 test superati**, inclusa la riproiezione GCP.
- `python -m compileall -q src tests`: completato.
- CLI di dataset, training, valutazione e inferenza caricate correttamente (`--help`).
- Riesecuzione dell'ingestion senza refresh: i tre ZIP CEMS sono stati riusati (`transferred: false`); il filtro temporale ha ridotto gli item STAC da 11 a 5. Tutti e cinque hanno superato la verifica per source ID, tempo e footprint: copertura unione CEMS 99,8–99,9%.
- Lettura dei tre ZIP CEMS reali: ogni archivio contiene un'AOI e poligoni `observedEventA`; `maximumFloodExtentA` è assente dal DEL iniziale ed è gestito come layer opzionale.
- Download S3 verificato per tutti i dieci asset: dimensioni STAC corrispondenti e SHA-256 registrati.
- Build reale dei prodotti 2435, 2436 e 2437: tutti e tre i manifesti `ready_for_training`, con split train/validation/test non vuoti e contenenti pixel positivi e negativi.
- Smoke test end-to-end su DEL v2: training CPU per 1 epoca, valutazione sul test spaziale e inferenza geospaziale su una finestra di 512x512 pixel dell'AOI; generati report, GeoTIFF, GeoJSON, GeoPackage e PNG.
- Metriche smoke-test, non indicative di prestazioni di produzione: precision 0,097, recall 0,642 e F1 0,169 sul test spaziale. Il training è durato una sola epoca e usa background negativo assunto.

## Limiti e prerequisiti mancanti

- I manifesti sono pronti per un esperimento di training, non attestano qualità operativa: l'etichetta negativa è un'assunzione e gli split sono spaziali all'interno di EMSR865.
- Il preprocessing è stato eseguito sui tre prodotti reali; verificare i rispettivi `dataset_manifest.json` e `patches.jsonl` prima di training o valutazione.
- L'input SAR resta ampiezza grezza: calibrazione radiometrica e normalizzazione non sono implementate. Non sono inclusi DEM, pendenza, pre-evento o change detection.
- La maschera negativa è un'assunzione didattica esplicita, non una verità osservata da CEMS.
- La suddivisione spaziale è interna a EMSR865; non è possibile dichiarare generalizzazione a eventi distinti con questa singola attivazione.

La pipeline completa è stata esercitata sui dati EMSR865. Un training significativo e una valutazione operativa richiedono più epoche, verifica delle etichette negative e dati di eventi distinti.
