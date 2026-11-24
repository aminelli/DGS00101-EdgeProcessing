# Fase 3 — Data Ingestion

## Scopo e perimetro

Il codice acquisisce esclusivamente l'attivazione EMSR865, scarica i prodotti DEL/GRA completati e fattibili che l'API dichiara disponibili, e cerca nel catalogo STAC Sentinel-1 GRD candidati VV/VH per data e intersezione con l'AOI. Gli item Sentinel-1 sono registrati come **candidati**, non come associazioni verificate, e in questa fase non vengono scaricati.

## File introdotti

- `src/copernicus_flood/ingestion/cems_client.py`: accesso all'API CEMS, filtro EMSR865 e download degli ZIP dei prodotti.
- `src/copernicus_flood/ingestion/cdse_client.py`: ricerca STAC per giorno dell'acquisizione CEMS, selezione di scene GRDH VV/VH che intersecano l'AOI e produzione di un rapporto candidati.
- `src/copernicus_flood/ingestion/registry.py`: cache HTTP in SQLite, registro chiamate, hash SHA-256, retry HTTP e download riprendibili.
- `src/copernicus_flood/cli.py`: comando d'ingestion.
- `config/settings.yaml`: perimetro e parametri di esecuzione.
- `requirements.txt`, `requirements-dev.txt`: dipendenze runtime e test.
- `tests/`: test offline del filtro prodotti, della cache e dell'idempotenza dei download.

La route CEMS corrisponde a quella descritta nel file OpenAPI locale `..\..\cems_rapidmapping_openapi_specs.yaml`; la risposta runtime viene salvata come fixture operativa e non si assume che lo schema live coincida perfettamente con lo schema dichiarato.

## Installazione ed esecuzione

Da PowerShell, nella cartella `copernicus-flood-ai`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
$env:PYTHONPATH = "src"
python -m copernicus_flood.cli --config config/settings.yaml
```

Il comando scarica i prodotti CEMS e interroga STAC. Le risposte API e i download completati vengono riusati nelle esecuzioni successive. Per richiedere un aggiornamento esplicito:

```powershell
python -m copernicus_flood.cli --config config/settings.yaml --refresh
```

`--refresh` forza l'aggiornamento delle risposte API e il nuovo download degli archivi CEMS. Non eseguire il refresh senza necessità, perché trasferisce nuovamente i dati.

## Output attesi

- `data/raw/cems/EMSR865/activation.json`: metadati dell'attivazione.
- `data/raw/cems/EMSR865/AOI01/*.zip`: archivi dei prodotti CEMS completati.
- `data/manifests/emsr865_stac_candidates.json`: associazioni candidate, sempre con `verified: false`.
- `data/manifests/ingestion.sqlite3`: cache delle risposte, registro delle richieste e checksum dei download.

I percorsi sotto `data/` sono esclusi dal controllo versione tramite `.gitignore`.

## Verifica eseguita

- `python -m pytest -q`: **3 test superati**.
- Esecuzione live del comando: ricevuta EMSR865, scaricati tre prodotti DEL (`2435`, `2436`, `2437`) e trovati 11 candidati STAC; nessuno è marcato come verificato.
- Riesecuzione senza `--refresh`: le tre voci prodotto risultano `transferred: false`, quindi non vengono riscaricate.
- Verifica degli archivi: i tre ZIP sono leggibili e `ZipFile.testzip()` non ha rilevato errori CRC.

Dimensioni e SHA-256 dei file scaricati in questa esecuzione:

| Prodotto | Dimensione (byte) | SHA-256 |
|---|---:|---|
| DEL v2, id 2435 | 24 444 004 | `179b7aff9326cc7b6c58e0ae1c7f9aca5f1bac1745802b6da29c64361b2934ec` |
| DEL monitoring 01 v1, id 2436 | 23 428 446 | `bf0f699fd7356d3786d989cc32d6064c8fb144e8065e8f44bcaf1dbcdd195345` |
| DEL monitoring 02 v1, id 2437 | 33 382 442 | `67ec57896183ef919ae4716c642652fb16910d47732fccb8afcb297ffbe32dd2` |

## Errori e limiti

- La cache evita richieste ripetute per URL identici; `--refresh` è l'azione esplicita per aggiornare i dati.
- I download usano file temporanei `.part`, verificano la lunghezza HTTP quando disponibile e supportano il ripristino tramite `Range` se il server lo consente. Gli archivi completati sono registrati con hash locale.
- Le immagini Sentinel-1 non sono scaricate: i candidati richiedono un controllo ulteriore dei metadati e della copertura prima di poter essere usati per training.
- Il conteggio dei candidati dipende dal catalogo al momento della richiesta. Nessun candidato è una corrispondenza CEMS-CDSE certificata solo in virtù della stessa data o dell'intersezione geografica.
- La cache conserva risposte e hash localmente; cancellare `data/manifests/ingestion.sqlite3` elimina la memoria delle richieste e dei download e può causare nuovi trasferimenti.

