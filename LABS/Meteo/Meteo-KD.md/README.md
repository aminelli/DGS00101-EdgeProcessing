# Previsioni meteo: Roma e Italia

Progetto dimostrativo per previsioni giornaliere a 24 ore, con un modello ML addestrato sullo storico Open-Meteo e una previsione NWP operativa consultata separatamente. La soluzione ibrida espone un blend configurabile, ma non lo considera validato: non viene scaricata una serie storica di previsioni NWP per calibrare o fare backtest del blend.

## Ipotesi e ambito

- Località iniziale: Roma, coordinate `41.9028`, `12.4964`.
- Storico: ultimi dieci anni di calendario fino a ieri UTC, acquisiti in blocchi di 90 giorni; `history_years` consente di scegliere un altro periodo.
- Orizzonte ML: domani in UTC; poiché lo storico è acquisito fino a ieri, il target è due giorni dopo l'ultimo giorno UTC completo. La data effettiva compare sempre nella risposta API.
- Target continui: temperatura massima giornaliera (`°C`), precipitazione totale (`mm`), velocità massima del vento (`km/h`).
- Dati storici Open-Meteo Archive sono dati di rianalisi/modello su griglia, non misure di una stazione romana. Per una validazione con osservazioni puntuali si deve aggiungere una fonte di stazioni, ad esempio SIARL/ARSIAL, dopo averne verificato copertura e condizioni d'accesso.
- La pagina operativa di Roma usa il modello della singola località di training. Coordinate diverse richiedono un modello addestrato e verificato per quella località.
- La nuova pipeline nazionale è separata: usa un teacher unico e uno student per ogni regione e unità territoriale provinciale. I risultati sono riferiti ai **capoluoghi**, non rappresentano l'intera superficie amministrativa.
- I pesi ML/NWP sono un parametro operativo configurabile, non appresi da un backtest storico.

## Architettura e file

```text
meteo/
  config.py       Configurazione e percorsi
  data.py         Download Archive e forecast, controlli qualità
  features.py     Aggregazione giornaliera e feature causali D -> D+2
  training.py     Split cronologico, baseline, training, export ONNX
  inference.py    Caricamento modello, chiamata NWP e blend
  cli.py          Comandi pipeline
  national.py     Catalogo 2026 e acquisizione nazionale con budget
  distillation.py Teacher nazionale, student territoriali e valutazione
  web.py          API FastAPI e pagine web
  static/         Interfaccia HTML, CSS e JavaScript
data/
  raw/            Risposte JSON archiviate (generate)
  processed/      CSV orari e feature (generati)
  sample/         Esempio sintetico, non utilizzabile per valutare il modello
artifacts/        Modelli, metriche ed esportazioni (generati)
tests/            Test offline
```

## Requisiti e installazione

Richiede Python 3.10 o successivo. Dalla cartella del progetto:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Se il progetto era già installato, ripetere `python -m pip install -e ".[test]"` per aggiungere Matplotlib, necessario ai grafici.

Per esportare ONNX e provare la quantizzazione:

```powershell
python -m pip install -e ".[edge]"
```

La quantizzazione dinamica dipende dagli operatori supportati da ONNX Runtime. L'esportatore verifica l'equivalenza ONNX/Scikit-learn e registra esplicitamente se il formato quantizzato non è supportato o non supera il controllo di qualità.

## Configurazione e comandi

Copiare `config.example.json` in `config.json` e modificare i valori necessari. I percorsi relativi nel file di configurazione sono risolti rispetto alla cartella in cui si trova il file. Non sono necessarie API key per l'uso base di Open-Meteo; è necessaria una connessione Internet.

```powershell
python -m meteo.cli --config config.json fetch-history
python -m meteo.cli --config config.json prepare
python -m meteo.cli --config config.json train
python -m meteo.cli --config config.json export
```

`fetch-history` salva le risposte originali sotto `data/raw/` e il CSV normalizzato in `data/processed/weather_hourly.csv`. `prepare` scrive `training_features.csv`. `train` salva i modelli e le metriche in `artifacts/`. `export` è opzionale e richiede gli extra `edge`.

Per usare i dieci anni in un modello già addestrato, dopo `fetch-history` rieseguire `prepare` e `train` oppure `experiment`: aggiornare il CSV storico non riaddestra automaticamente i modelli esistenti. I comandi `train` ed `experiment` sovrascrivono i rispettivi artefatti nella cartella configurata.

Le chiamate API possono richiedere diversi minuti. Timeout ed errori HTTP interrompono il comando con un messaggio esplicito; i blocchi completati restano disponibili come JSON grezzi. Rilanciare il comando ripete la raccolta, non fonde in modo incrementale dati parziali.

### Campi e unità del CSV orario

| Colonna                               | Unità         |
| ------------------------------------- | ------------- |
| `timestamp`                           | ISO 8601 UTC  |
| `location`                            | Nome località |
| `temperature_2m`, `dew_point_2m`      | °C            |
| `relative_humidity_2m`, `cloud_cover` | %             |
| `precipitation`                       | mm nell'ora   |
| `pressure_msl`                        | hPa           |
| `wind_speed_10m`                      | km/h          |
| `wind_direction_10m`                  | gradi         |
| `shortwave_radiation`                 | W/m²          |

Nei riepiloghi giornalieri `precipitation_sum` è in mm/giorno, `shortwave_radiation_sum` in Wh/m²/giorno; i target mantengono le unità della relativa variabile.

## Training e valutazione

Gli esempi sono costruiti ordinando i giorni in modo cronologico. Per prevedere domani, il target di ogni riga è il giorno `D+2` e le feature includono le statistiche osservate fino al giorno `D`, i lag di 1, 2, 3 e 7 giorni e la stagionalità annuale. Questo allineamento consente di usare l'ultimo giorno completo disponibile (ieri) per produrre la previsione di domani. I giorni con meno di 24 osservazioni sono esclusi. La suddivisione è 70% train, 15% validation, 15% test in ordine temporale; gli score di validation e test sono riportati separatamente.

Per ogni target viene addestrato un Random Forest e confrontato su validation e test con la persistenza del valore del giorno precedente. Le metriche `MAE`, `RMSE` e `R²` sono salvate in `artifacts/metrics.json`. Sono metriche di regressione: non si genera una matrice di confusione perché i target non sono classi. L'export ONNX salva in `artifacts/export_report.json` dimensioni e tempi di inferenza misurati sul batch di verifica; i tempi sono indicativi e dipendono dall'hardware.

Il comando `train` stampa una tabella con MAE, RMSE e R² del **test**, separatamente per modello ML e baseline di persistenza. Salva inoltre in `artifacts/plots/` un'immagine PNG per target: a sinistra la serie cronologica dei valori storici, delle previsioni ML e della baseline; a destra il confronto tra valori storici e previsioni ML, con la diagonale ideale. Le date rappresentano il giorno target `D+2` in UTC. I grafici si generano anche senza interfaccia grafica e vengono sovrascritti a ogni training.

Non viene stampata una matrice di confusione: temperatura, precipitazione e velocità del vento sono target continui. Per ottenerne una occorrerebbe definire un problema di classificazione e soglie motivate, diverse dalle metriche di regressione attuali.

### Esperimenti di training

Per confrontare i candidati senza modificare il modello usato da API ed export:

```powershell
python -m meteo.cli --config config.json experiment
```

Il comando salva modelli, metriche e grafici separatamente in `artifacts/experiments/`. Non attiva i modelli sperimentali nell'app e non genera i loro file ONNX; il comando `train` e i relativi artefatti rimangono invariati.

Per ogni target confronta Random Forest con due valori di `min_samples_leaf` ed Extra Trees. Aggiunge medie causali sugli ultimi tre giorni per temperatura, precipitazioni e vento. Per la temperatura prova anche un modello che apprende la correzione rispetto alla persistenza. Per la pioggia prova un modello a due fasi: probabilità di precipitazione giornaliera maggiore di 0 mm e quantità condizionata alla pioggia; la previsione continua è il prodotto delle due. La soglia di 0 mm è una definizione sperimentale, non una soglia di allerta meteorologica.

La selezione avviene sulla base del **MAE di validation**, mentre il test non partecipa alla selezione. Il report confronta il candidato selezionato con la persistenza sul test e riporta anche il MAE su tre sottointervalli cronologici di validation, ottenuti con lo stesso modello: non sono un backtest rolling con riaddestramento. Gli esempi ai confini train/validation e validation/test vengono esclusi quando il loro target `D+2` non è ancora noto all'inizio del periodo successivo. Lo storico deve avere giornate complete consecutive, altrimenti l'esperimento segnala un errore invece di associare target a date errate.

Per la fase di classificazione pioggia, il report include Brier score, matrice di confusione sul test (soglia di probabilità 0,5; righe reali e colonne previste nell'ordine asciutto/piovoso) e MAE della quantità condizionata sui soli giorni piovosi del test, se presenti. Queste metriche descrivono il modello a due fasi anche se non viene scelto per la regressione. Se nel training manca una delle due classi, il candidato viene escluso e il motivo è riportato. La soglia 0,5 non è calibrata.

Questi confronti sono esperimenti su rianalisi Open-Meteo, non garantiscono un miglioramento; il test va mantenuto separato e non riutilizzato per scegliere ulteriori parametri. Una valutazione operativa richiede più finestre rolling, osservazioni indipendenti e, per il blend ibrido, previsioni NWP storiche disponibili al momento della previsione.

### Esperimento avanzato: NWP, boosting e rolling

Per temperatura massima e vento massimo si può eseguire un confronto separato con previsioni archiviate:

```powershell
python -m meteo.cli --config config.json fetch-nwp-history --start-date 2024-01-01
python -m meteo.cli --config config.json advanced-experiment
```

Il primo comando usa la [Previous Model Runs API](https://open-meteo.com/en/docs/previous-runs-api) di Open-Meteo, modello GFS `gfs_seamless`, per temperatura e vento orari con anticipo nominale fisso di **72 ore** (`previous_day3`). Aggrega le previsioni giornaliere massime in `data/processed/nwp_previous_runs_daily.csv`; esclude e segnala i giorni con dati orari incompleti. Il periodo predefinito termina ieri UTC e può essere ristretto con `--start-date` e `--end-date`. La copertura dipende dall'archivio e non equivale ai dieci anni dello storico di rianalisi.

Per una riga di feature `D` il target è `D+2`: il valore nominale NWP più recente per quel giorno risale a `D-1 23:00 UTC`, prima della fine di `D`. **Questo è un margine prudenziale, non una garanzia di disponibilità effettiva:** Previous Runs non fornisce i tempi di pubblicazione o eventuali revisioni di ogni dato. Senza una registrazione indipendente della disponibilità, il backtest NWP è indicativo e non certifica l'assenza di leakage. Le feature NWP archiviate non vengono utilizzate nell'API operativa.

Il comando avanzato confronta Random Forest, Extra Trees e `HistGradientBoostingRegressor`; per la temperatura prova anche il boosting della correzione rispetto alla persistenza. Le feature includono le medie recenti e, per il giorno `D`, dispersione oraria di temperatura, pressione e vento, variazione tra prime e ultime sei ore di temperatura e pressione, e componenti seno/coseno medie della direzione del vento. I dati orari necessari devono essere completi; date mancanti o provenienza NWP incoerente generano un errore esplicito.

Il test finale è l'ultimo 15% delle giornate allineate. Prima del test, tre finestre **rolling espandenti** riaddestrano i candidati, escludendo dai training i target `D+2` non ancora noti all'inizio della rispettiva validation. Il MAE aggregato sui tre fold seleziona il candidato; questo viene riaddestrato sul pre-test e valutato una sola volta sul test, rispetto a persistenza e NWP. Il report mostra MAE per stagione e, per il vento, per terzili dell'intensità **osservata**: queste classi descrivono il test a posteriori, non sono previsioni di intensità.

Risultati, modelli e grafici sono separati in `artifacts/advanced_experiments/`; non sostituiscono API o modelli ONNX operativi. Le metriche su periodi differenti non sono confrontabili direttamente. Per una verifica operativa servono tempi effettivi di disponibilità NWP e osservazioni indipendenti.

L'API `/api/forecast` restituisce il risultato ML per domani, i dati NWP operativi per sette giorni e il blend quando le date coincidono. Se il dato NWP target non è disponibile, il blend è omesso e lo stato lo dichiara. Il modello NWP è fornito dal servizio Open-Meteo; non viene implementata una simulazione atmosferica proprietaria.

### Teacher nazionale e student territoriali

La pipeline di distillazione non modifica il modello e gli artefatti operativi di Roma. Il catalogo amministrativo 2026 comprende 20 regioni, 110 unità territoriali di livello provinciale (incluse città metropolitane, province autonome e unità statistiche) e 118 punti. Il punto regionale è il capoluogo regionale; il modello provinciale può ricevere più punti, ad esempio Barletta, Andria e Trani. I valori sono **previsioni puntuali per ciascun capoluogo**, non medie su regioni o province e non coprono uniformemente l'intero territorio italiano.

Il catalogo usa i codici provinciali `COD_PROV` della [geografia ISTAT 2026 distribuita da geojson-italy](https://github.com/guglielmo/geojson-italy/tree/2026.1), i punti territoriali della [Protezione Civile](https://github.com/pcm-dpc/COVID-19) e la geocodifica Open-Meteo per i capoluoghi speciali e multipli. I dati ISTAT/geojson-italy e Protezione Civile sono attribuiti alle rispettive fonti, distribuite con licenza CC BY; la geocodifica è fornita da Open-Meteo. Il catalogo è uno **snapshot 2026**, non si aggiorna automaticamente: verificare variazioni amministrative e capoluoghi prima di un uso operativo. Le riforme sarde del 2026 sono incluse; i punti aggiunti sono geocodificati e non costituiscono confini amministrativi.

Per ciascuno dei tre target un teacher `HistGradientBoostingRegressor` apprende dagli esempi dei capoluoghi, usando anche latitudine e longitudine. Ogni territorio ha uno student leggero `StandardScaler` + `Ridge` per target; i tre regressori sono raccolti in un file student indipendente. Se il territorio ha più capoluoghi, lo student è unico ma riceve le coordinate del punto selezionato. Gli obiettivi per il training dello student sono `0,7 × previsione teacher + 0,3 × valore storico`: per una perdita quadratica equivale, a parità di regolarizzazione, a bilanciare apprendimento dal teacher e dai dati reali. Il peso è una scelta sperimentale fissata, non ottimizzata sul test. Precipitazione e vento previsti sono limitati inferiormente a zero.

Lo split è cronologico sui giorni comuni a tutti i punti (70% training, 15% validation, 15% test); i target `D+2` ai confini sono esclusi quando non ancora disponibili. Teacher e student sono addestrati **solo sul training**. La validation è riservata e non viene usata per scegliere il peso; il test confronta MAE, RMSE e R² di student, teacher, uno student supervisionato senza distillazione e persistenza. Nel caso di più capoluoghi, le metriche territoriali aggregano gli esempi dei punti. Non è una validazione su località mai viste, né un confronto su osservazioni indipendenti: la distillazione può anche peggiorare i risultati.

Comandi, con gli stessi percorsi predefiniti usati dal server:

```powershell
python -m meteo.cli --config config.json national-catalog
python -m meteo.cli --config config.json national-fetch
python -m meteo.cli --config config.json national-train
python -m meteo.cli --config config.json national-refresh
```

Il catalogo e i CSV sono in `data/national/`, uno storico per punto; il training richiede **tutti** i punti e segnala quelli mancanti. I modelli e report vengono pubblicati come versioni in `artifacts/distillation/runs/`: ogni versione contiene `teacher.joblib` e uno student indipendente per territorio in `students/`. L'interfaccia confronta la dimensione effettiva del file teacher con quella del file student, entrambi relativi a tre target. Il puntatore `current` viene aggiornato solo dopo il completamento della nuova versione; una versione incompleta non viene attivata. Le versioni precedenti non sono eliminate automaticamente. Gli student non sono esportati in ONNX da questi comandi. L'API previsioni nazionali usa lo storico locale disponibile; `national-refresh` aggiunge i giorni successivi senza riaddestrare. Per aggiornare anche i modelli, rieseguire `national-train`.

**Tempi e limiti di servizio:** Open-Meteo conta come più chiamate una richiesta storica oltre due settimane; 10 anni per 118 punti richiedono circa 31.000 chiamate equivalenti. Il piano gratuito, esclusivamente non commerciale, ha limiti pubblicati di 10.000/giorno, 5.000/ora e 600/minuto per IP/applicazione; controllare sempre i [termini aggiornati](https://open-meteo.com/en/terms) e il [conteggio delle chiamate](https://open-meteo.com/en/pricing). `national-fetch` e `national-refresh` impongono un budget locale conservativo di 4.000 chiamate equivalenti su 24 ore e 2.000 su un'ora, con almeno un secondo tra le richieste. `--daily-budget` consente di abbassare il massimo locale. Sono limiti **locali**, non possono misurare il traffico di altre applicazioni o utenti sullo stesso IP e non garantiscono l'assenza di `429`; in caso di limite o errore interrompere e riprendere più tardi, senza aggirarlo.

L'acquisizione si interrompe con messaggio esplicito quando esaurisce il budget; rieseguire `national-fetch` dopo l'intervallo indicato. Conserva i blocchi già ricevuti e fissa date, coordinate e configurazione per consentire la ripresa su più giorni; non rigenerare il catalogo o cambiare `history_years` durante un'acquisizione. `request-budget.json` conserva le richieste conteggiate e `fetch.lock` impedisce acquisizioni concorrenti; se un'interruzione forzata lascia il lock, verificare che nessuna acquisizione sia attiva prima di rimuoverlo. Con il budget predefinito il download completo richiede diversi giorni. Non lanciare download, refresh e training contemporaneamente sugli stessi dati.

## API e interfaccia

Avviare il server:

```powershell
uvicorn meteo.web:app --reload
```

- `GET /`: previsioni ML, NWP e blend con grafici e tabella; il pulsante aggiorna i dati.
- `GET /storico`: medie e grafici giornalieri degli ultimi 30, 90 o 365 giorni disponibili nel CSV.
- `GET /valutazioni`: metriche e grafici del modello operativo, degli esperimenti e dell'esperimento avanzato; mostra la matrice di confusione della pioggia quando disponibile.
- `GET /italia`: selezione regione/provincia e capoluogo, previsione student e confronto delle metriche della distillazione.
- `GET /health`: stato dell'applicazione e presenza del modello.
- `GET /api/forecast`: previsioni ML, NWP e blend.
- `GET /api/history?days=90`: giornate complete dello storico, fino all'ultima data disponibile; `days` va da 1 a 3650.
- `GET /api/evaluations/{report}`: report `operativo`, `sperimentale` o `avanzato`; i relativi PNG sono disponibili in `/api/evaluations/{report}/plots/{target}`.
- `GET /api/national/catalog`: catalogo nazionale, oppure lo snapshot del modello pubblicato.
- `GET /api/national/forecast?level=province&territory=001&point=p001-1`: previsione dello student nel punto selezionato, con data dell'ultimo storico e data target UTC.
- `GET /api/national/evaluation?level=province&territory=001`: metriche del territorio; i parametri `level` accettano `region` o `province`.
- `GET /docs`: documentazione interattiva OpenAPI.

Prima di eseguire il training, `/api/forecast` risponde `503` indicando che il modello non è disponibile. Errori del servizio dati esterno sono esposti come `502` senza dettagli interni.

L'interfaccia usa JavaScript e grafici SVG locali, senza CDN. Lo storico e i report già generati sono consultabili offline; la previsione richiede il servizio Open-Meteo. Un report o grafico mancante richiede il relativo `train`, `experiment` o `advanced-experiment` (i comandi sovrascrivono i propri artefatti). I risultati sperimentali non modificano l'API operativa e non sono direttamente confrontabili tra periodi diversi; il backtest NWP avanzato resta indicativo.

Le previsioni student nazionali usano invece i CSV già scaricati e non richiedono una richiesta NWP al momento della consultazione. Se lo storico non è aggiornato, la data target può non coincidere con domani UTC e la pagina lo segnala; finché non è stato completato `national-train`, le relative API rispondono con un errore esplicito.

Il server usa le impostazioni predefinite di `Settings`, inclusi coordinate e percorsi del progetto; non legge automaticamente `config.json` utilizzato dai comandi CLI. Se la CLI usa percorsi diversi, il server non vedrà automaticamente quegli artefatti.

## Test

```powershell
python -m pip install -e ".[test]"
pytest
```

I test sono offline e usano dati sintetici generati in memoria. `data/sample/weather_hourly_synthetic.csv` è un piccolo esempio leggibile, esplicitamente sintetico e troppo breve per addestrare o valutare un modello.

## Open-Meteo: fonti e limiti

- **Archivio storico**: `https://archive-api.open-meteo.com/v1/archive`; il codice richiede le nove variabili orarie dichiarate in `meteo/data.py`.
- **Previsione operativa**: `https://api.open-meteo.com/v1/forecast`; il servizio seleziona il modello NWP disponibile e restituisce sette giorni di target giornalieri.
- **Timezone**: `UTC`, usato per mantenere coerenti aggregazione e orizzonte. Le date locali di Roma possono non coincidere con il giorno UTC.
- **Attribuzione/licenza**: verificare i termini aggiornati di Open-Meteo e includere l'attribuzione richiesta nella distribuzione dei dati o del servizio.
- Le API possono aggiornare dati, modelli e disponibilità. Il codice salva i JSON grezzi e controlla la forma delle risposte, ma non congela la versione del modello NWP remoto.
- Il servizio di previsione richiede accesso a Internet; training, test e preparazione si possono eseguire offline dopo aver acquisito o fornito il CSV.

## Troubleshooting

- **Timeout o HTTP error**: verificare connettività, limiti del servizio e riprovare; i JSON dei blocchi già scaricati non vengono cancellati.
- **Dati incompleti / pochi esempi**: controllare che il CSV copra un periodo sufficiente e contenga giornate complete; il training richiede almeno 100 esempi giornalieri validi.
- **Errore ONNX**: installare gli extra `edge`; se un operatore non è convertibile, l'esportazione fallisce esplicitamente senza dichiarare l'artefatto verificato.
- **Quantizzazione non disponibile**: verificare `export_report.json`; alcuni modelli ad alberi non offrono operatori quantizzabili nella versione/runtime installata.
- **Errore 502 nell'app**: verificare la disponibilità dell'endpoint Open-Meteo e la connessione del server.

## Limiti noti

Il sistema è un prototipo didattico, addestrato su una sola località e su dati di rianalisi. Non è progettato per decisioni di sicurezza o allerta. La validazione temporale non sostituisce una valutazione meteorologica operativa su dati osservativi. Il blend ML/NWP usa un peso configurato e deve essere considerato sperimentale finché non viene calibrato e valutato con previsioni NWP archiviate e osservazioni indipendenti.
