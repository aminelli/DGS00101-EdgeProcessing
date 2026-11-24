# Fase 1 — Analisi preliminare

**Progetto:** mappatura delle alluvioni con dati EMSR865 e Sentinel-1  
**Verifica delle fonti:** 24 novembre 2026  
**Perimetro:** esclusivamente l'attivazione EMSR865. Questa fase analizza dati e fattibilità; non implementa codice applicativo e non avvia training.

## 1. Esito in breve

EMSR865 è un'attivazione Rapid Mapping per l'alluvione in Córdoba, Colombia. L'API pubblica restituisce una sola AOI, `Monterìa`, tre prodotti Delineation (DEL) — il prodotto iniziale e due monitoraggi — e nessun prodotto Grading (GRA). Sono pubblicati livelli GeoJSON e archivi ZIP dei prodotti.

La ricerca del catalogo CDSE conferma la disponibilità di scene Sentinel-1 GRD VV/VH che intersecano l'estensione geografica in date compatibili con le immagini elencate nei prodotti CEMS. È inoltre presente una scena del 21 gennaio, prima dell'evento registrato il 26 gennaio. Questo è sufficiente per proseguire con la progettazione dell'acquisizione e della verifica spaziale; non prova ancora la copertura completa delle AOI né l'idoneità delle scene come coppie di input.

**La demo non è ancora pronta per un training verificabile.** Le geometrie campione di `observedEventA` rappresentano poligoni positivi di area allagata; l'API e i livelli consultati non forniscono una maschera esplicita di tutti i pixel validi non allagati o di acqua permanente. Inoltre EMSR865 contiene un solo evento e una sola AOI: non consente una suddivisione train/validation/test per evento né una misura di generalizzazione verso eventi distinti. Non assumere il resto dell'AOI come classe negativa senza verificarne la semantica.

## 2. Fonti e metodi verificati

| Fonte                                                                                                                      | Uso in questa analisi                                                                                                  |
| -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| [CEMS Mapping](https://mapping.emergency.copernicus.eu/)                                                                   | Servizio e portale ufficiale Rapid Mapping.                                                                            |
| [API pubblica per EMSR865](https://mapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/?code=EMSR865) | Metadati dell'attivazione, AOI, prodotti, versioni, livelli e link di download. Risposta JSON consultata direttamente. |
| [Specifiche OpenAPI locali](./cems_rapidmapping_openapi_specs.yaml)                                                        | File indicato come riferimento per mappare l'API; descrive la route e lo schema attesi.                                |
| [Risorse CEMS](https://mapping.emergency.copernicus.eu/resources/)                                                         | Materiale informativo ufficiale sul servizio.                                                                          |
| [Documentazione CDSE — STAC](https://documentation.dataspace.copernicus.eu/APIs/STAC.html)                                 | Endpoint STAC, collezioni e modalità di ricerca del catalogo.                                                          |
| [Collezione STAC Sentinel-1 GRD](https://stac.dataspace.copernicus.eu/v1/collections/sentinel-1-grd)                       | Metadati ufficiali della collezione Sentinel-1 GRD.                                                                    |
| [Documentazione CDSE — OData](https://documentation.dataspace.copernicus.eu/APIs/OData.html)                               | Alternativa documentata per interrogazione catalogo e download dei prodotti.                                           |
| [API CDSE](https://dataspace.copernicus.eu/analyse/apis)                                                                   | Panoramica ufficiale delle interfacce di accesso ai dati.                                                              |

Le interrogazioni STAC eseguite hanno filtrato la collezione `sentinel-1-grd` per bbox dell'AOI e intervalli temporali brevi. Una ricerca per bbox non sostituisce la verifica d'intersezione con il poligono AOI effettivo: questa dovrà essere effettuata durante il data engineering.

## 3. CEMS: attivazione, API e prodotti

### Attivazione e AOI

- **Codice:** EMSR865.
- **Evento:** Flood in Córdoba, Colombia; categoria Flood, sottocategoria Riverine flood.
- **Data evento registrata dall'API:** 26 gennaio 2026, 05:30 UTC.
- **AOI:** una, denominata `Monterìa`, numero 1.
- **Stato:** attivazione chiusa.

La route documentata nel file OpenAPI e verificata sulla risposta live è:

`GET https://mapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/?code=EMSR865`

Il parametro `code` consente di limitare la richiesta all'attivazione richiesta. La risposta corrente usa una struttura paginata (`count`, `next`, `previous`, `results`) anche se questa forma non è descritta completamente nel file OpenAPI locale.

### Prodotti effettivamente restituiti

Per l'unica AOI l'API restituisce tre record DEL, tutti fattibili e con versione completata:

| Prodotto            | Versione | Data immagine indicata da CEMS (UTC) | Livelli principali                                                       |
| ------------------- | -------: | ------------------------------------ | ------------------------------------------------------------------------ |
| DEL iniziale        |        2 | 2026-02-02 23:14                     | `observedEventA`, `modelledEventA`, `floodDepthA`                        |
| DEL monitoraggio 01 |        1 | 2026-02-08 23:14                     | `observedEventA`, `modelledEventA`, `maximumFloodExtentA`, `floodDepthA` |
| DEL monitoraggio 02 |        1 | 2026-02-14 10:50                     | `observedEventA`, `modelledEventA`, `maximumFloodExtentA`, `floodDepthA` |

I record contengono `downloadPath` per gli ZIP dei singoli prodotti. L'API espone anche i livelli in JSON e i relativi stili; per esempio, il livello `observedEventA` del DEL iniziale è disponibile come [GeoJSON](https://rapidmapping-viewer.s3.eu-west-1.amazonaws.com/EMSR865/AOI01/DEL_PRODUCT/EMSR865_AOI01_DEL_PRODUCT_observedEventA_v2.json). L'API riporta inoltre `productsPath` per l'archivio complessivo e `blpPath` per il Base Layers Package.

Non è stato restituito alcun GRA per EMSR865. I livelli esaminati non espongono un livello esplicitamente denominato acqua permanente.

### Semantica osservata e limiti delle etichette

Nel GeoJSON `observedEventA` consultato, gli attributi includono `event_type: 5-Flood`, `obj_desc: Riverine flood`, `det_method: Semi-automatic extraction` e `notation: Flooded area`. I poligoni sono pertanto candidati positivi coerenti con il target, ma rimangono un prodotto cartografico derivato e non una verità a terra indipendente.

Il livello `modelledEventA` indica invece `det_method: Modelling`; il livello di monitoraggio `maximumFloodExtentA` è anch'esso modellato. Raccomandazione preliminare: non mescolare questi livelli in un'unica classe di verità senza una decisione esplicita e una valutazione separata della loro semantica.

Nei livelli consultati non compare un poligono o attributo che certifichi ogni area non allagata come osservata e valida. Il confine AOI dell'API è utile per localizzare il prodotto, ma non dimostra da solo che ogni pixel interno sia una misura negativa. L'utente ha tuttavia autorizzato esplicitamente, per la finalità didattica, l'uso del background interno all'AOI come classe negativa non verificata. Le aree esterne all'AOI, i NoData e le zone dall'etichetta incerta restano ignore (255).

## 4. Sentinel-1 e associazione temporale

La documentazione CDSE espone la collezione STAC `sentinel-1-grd` e i relativi endpoint di ricerca e item. La collezione descrive prodotti Sentinel-1 GRD Level-1; gli asset VV e VH interrogati sono GeoTIFF/COG `uint16`, riportati come ampiezza, con pixel spacing di 10 m. Questo valore di spaziatura non va confuso con la risoluzione effettiva del sensore. Il preprocessing radiometrico e geometrico dovrà essere scelto e documentato prima di usare i valori come input.

Le ricerche CDSE per l'estensione EMSR865 hanno restituito scene che intersecano il bbox e hanno polarizzazioni VV/VH:

| Ruolo candidato            | Item STAC                                                                 | Riscontro con il metadato CEMS                                                                                                                                                                               |
| -------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Pre-evento                 | `S1A_IW_GRDH_1SDV_20260121T231523_20260121T231548_062872_07E2FE_9C36_COG` | Acquisizione 21 gennaio, prima della data evento del 26 gennaio; la geometria va ancora intersecata con l'AOI completa.                                                                                      |
| Post-evento / DEL iniziale | `S1A_IW_GRDH_1SDV_20260202T231523_20260202T231548_063047_07E979_15F2_COG` | CEMS riporta il 2 febbraio alle 23:14; il candidato STAC acquisisce alle 23:15:23 e riporta VV/VH. Compatibile temporalmente, ma l'identità precisa della sorgente va verificata sui metadati/prodotto CEMS. |
| Monitoraggio 01            | `S1C_IW_GRDH_1SDV_20260208T231438_20260208T231511_006271_00C99F_D74A_COG` | CEMS riporta l'8 febbraio alle 23:14; STAC riporta 23:14:38, con VV/VH.                                                                                                                                      |
| Monitoraggio 02            | `S1A_IW_GRDH_1SDV_20260214T105030_20260214T105055_063214_07EFAC_E4E9_COG` | CEMS riporta il 14 febbraio alle 10:50; STAC riporta 10:50:30, con VV/VH.                                                                                                                                    |

Questi riscontri dimostrano la disponibilità di candidati catalogati, non una corrispondenza certificata fra ogni item STAC e il raster d'immagine del prodotto CEMS. In particolare, la risposta EMSR indica nomi di file ortorettificati e metadati dell'immagine, ma non espone in quei record un link CDSE o un identificativo STAC. L'associazione va registrata come verificata solo dopo confronto di footprint, tempo, sensore, orbita e metadati degli asset. Occorre anche verificare che la copertura delle scene combinate includa l'intera area d'interesse usata per l'etichetta.

Le scene STAC riportano anche asset COG accessibili tramite link HTTPS/OIDC e percorsi S3 autenticati. La modalità concreta di download o lettura parziale dovrà essere scelta in fase di progettazione in base alle credenziali CDSE disponibili; non inserire credenziali nel repository.

## 5. Architettura minima proposta

Per questa demo è sufficiente una pipeline batch, senza API di produzione o orchestrazione:

1. **Catalogo CEMS:** richiesta limitata a EMSR865; registrazione della risposta originale, versione dei prodotti, identificativi, date, checksum e link.
2. **Catalogo Sentinel-1:** ricerca STAC per area e finestre temporali; conservazione degli item STAC e degli asset selezionati.
3. **Verifica abbinamenti:** confronto esplicito tra geometrie, tempi e metadati CEMS/CDSE; scarto o marcatura come non verificata delle associazioni dubbie.
4. **Preprocessing:** allineamento di griglie e CRS, gestione di NoData, conversione/preparazione SAR documentata e rasterizzazione delle etichette con ignore 255.
5. **Dataset:** registrazione della provenienza per patch e divisione riproducibile con blocchi spaziali/temporali, segnalando che non è una validazione indipendente per evento.
6. **Modello e report:** un modello binario leggero di segmentazione, ad esempio U-Net PyTorch, usando l'assunzione didattica sui negativi approvata dall'utente e dichiarandone l'incertezza in tutti i report.
7. **Inferenza:** esportazione GeoTIFF georeferenziato e revisione dei risultati in QGIS.

Non sono giustificati in questa demo MLflow, DVC, Airflow, FastAPI o un WebGIS. Possono essere rivalutati solo se emerge una necessità concreta.

## 6. Modelli candidati

- **U-Net binaria:** scelta minima per una demo di segmentazione, se il dataset risultasse sufficiente. La loss deve ignorare 255 e le metriche devono escludere gli stessi pixel.
- **DeepLabV3:** possibile alternativa, ma più complessa del necessario finché non è nota la quantità e varietà effettiva delle etichette.
- **Change detection pre/post:** interessante per il segnale di allagamento, ma richiede coppie temporalmente coerenti e registrate, oltre a una politica per differenze di piattaforma/orbita. Non va assunto come input iniziale.

Non è opportuno promettere metriche di generalizzazione: con una sola attivazione e una sola AOI si possono produrre al massimo risultati esplorativi sullo stesso evento.

## 7. Dipendenze e requisiti hardware preliminari

**Dipendenze essenziali candidate:** Python, `requests` o `httpx` per le API, `rasterio`, `geopandas`, `shapely`, `pyproj`, `numpy`, `pandas` e `torch`. `PyYAML` è utile se si adotta una configurazione YAML; `matplotlib` è sufficiente per grafici basilari. Rasterio copre i principali flussi raster della demo; l'uso diretto di binding GDAL separati va deciso solo se serve una funzione non coperta.

**Hardware:** la ricerca API non richiede GPU. Preprocessing e un modello piccolo possono funzionare su CPU, ma CUDA è preferibile per training iterativo. Gli asset Sentinel-1 interrogati sono grandi (centinaia di MB per polarizzazione e scena; gli archivi completi superano 1 GB per item nei record restituiti): usare finestre COG o ritagli mirati, se compatibili con il metodo di accesso, riduce download e disco. La memoria e lo spazio disco esatti dipendono da numero di scene, mosaici, risoluzione e dimensione patch e andranno stimati dopo aver definito l'area da elaborare.

## 8. Criticità, assunzioni e informazioni mancanti

1. **Maschera negativa/valida:** manca una conferma che lo spazio interno all'AOI sia interamente osservato e che l'assenza di un poligono flood significhi non allagato. Non codificare il background come negativo senza tale prova.
2. **Acqua permanente:** non è stato trovato un livello esplicito nelle layer list consultate. Non inventare una classe o una fonte maschera.
3. **Generalizzazione per evento:** il vincolo EMSR865-only rende impossibile dividere gli eventi tra training, validation e test. Uno split casuale di patch introdurrebbe leakage; uno split spaziale sarebbe soltanto esplorativo e non dimostrerebbe generalizzazione a nuovi eventi.
4. **Allineamento immagine-etichetta:** i tempi CEMS e STAC sono vicini per i candidati verificati, ma i record non forniscono un identificativo comune. Serve un controllo esplicito delle geometrie e delle metadata.
5. **Più acquisizioni e versioni:** usare l'ultima versione di ciascun prodotto va distinto dal combinare prodotti con date diverse. Ogni monitoraggio è una diversa istantanea dell'evento.
6. **Schema API non perfettamente allineato:** il file OpenAPI locale non descrive tutta la struttura osservata nella risposta live. Tra le discrepanze, il catalogo live include campi di paginazione; il dato `resolutionClass` osservato è `HR+`, non incluso nell'enumerazione locale. Il parser futuro deve usare il JSON effettivo, gestire valori inattesi in modo esplicito e mantenere test/fixture aggiornabili. Non modificare la specifica upstream come se fosse normativa.
7. **Preprocessing SAR:** i COG sono descritti come ampiezza `uint16`; non è ancora stata definita né verificata la trasformazione radiometrica, la normalizzazione né la scelta fra input pre/post e stack multitemporali.

## 9. Raccomandazione e gate per la fase successiva

**Fattibilità:** acquisizione e prototipazione geospaziale su EMSR865 sono fattibili. È possibile procedere con un training didattico soltanto rendendo visibile l'assunzione sui pixel negativi e limitando le conclusioni. I dati e i vincoli attualmente verificati non consentono una valutazione generalizzabile; i risultati non potranno essere descritti come generalizzazione su eventi distinti finché il perimetro resterà limitato a EMSR865.

**Decisione dell'utente dopo la revisione:** lo scopo è didattico; è autorizzato usare, soltanto nella demo, il background interno all'AOI e non coperto dai poligoni di allagamento come classe negativa, dichiarandolo esplicitamente come assunzione non verificata. Non esiste una maschera o specifica aggiuntiva fornita dall'utente. La decisione non cambia i limiti statistici né autorizza a descrivere il risultato come ground truth o generalizzazione su eventi distinti.

La Fase 1 è approvata. La Fase 2 può progettare il dataset con questa assunzione dichiarata e mantenere una valutazione esplorativa per EMSR865.

