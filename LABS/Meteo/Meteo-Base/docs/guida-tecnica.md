# Guida tecnica: meteorologia, modelli e deployment edge

Questa guida spiega i concetti usati nel progetto. Per installazione, comandi, API e struttura operativa consultare [README.md](../README.md).

## Dizionario delle sigle

| Sigla | Espansione                        | Significato nel progetto                                                                            |
| ----- | --------------------------------- | --------------------------------------------------------------------------------------------------- |
| API   | Application Programming Interface | Interfaccia HTTP usata per richiedere dati meteo e servire previsioni.                              |
| CSV   | Comma-Separated Values            | Formato tabellare dei dati orari e delle feature elaborate.                                         |
| ETL   | Extract, Transform, Load          | Processo di acquisizione, pulizia e persistenza dei dati.                                           |
| GUI   | Graphical User Interface          | Pagina web per richiedere e leggere i risultati.                                                    |
| HTTP  | Hypertext Transfer Protocol       | Protocollo di trasporto delle API Open-Meteo e dell'API applicativa.                                |
| JSON  | JavaScript Object Notation        | Formato delle risposte API, della configurazione e delle metriche.                                  |
| MAE   | Mean Absolute Error               | Errore assoluto medio tra previsione e valore effettivo, nella stessa unità del target.             |
| ML    | Machine Learning                  | Modello statistico addestrato sui dati storici e sulle feature temporali.                           |
| NWP   | Numerical Weather Prediction      | Previsione ottenuta da modelli numerici dell'atmosfera; in questo progetto è fornita da Open-Meteo. |
| ONNX  | Open Neural Network Exchange      | Formato portabile per rappresentare ed eseguire modelli compatibili su runtime diversi.             |
| R²    | Coefficiente di determinazione    | Misura di quanto il modello descrive la variabilità del target; può essere negativo su dati test.   |
| REST  | Representational State Transfer   | Stile architetturale usato dagli endpoint HTTP.                                                     |
| RMSE  | Root Mean Squared Error           | Radice dell'errore quadratico medio; penalizza maggiormente errori grandi.                          |
| UTC   | Coordinated Universal Time        | Riferimento orario comune usato per timestamp e aggregazioni giornaliere.                           |

## Machine learning meteorologico

Il modello usa una tabella di feature per stimare grandezze future. Qui ogni riga rappresenta un giorno `D`; il target è il riepilogo del giorno `D+2`, così l'ultimo giorno storico completo (ieri) può essere usato per prevedere domani. Le feature includono osservazioni aggregate fino a `D`, lag temporali e codifica ciclica del giorno dell'anno. Una Random Forest media le previsioni di molti alberi decisionali, riducendo la dipendenza da un singolo albero.

La qualità va confrontata con un baseline semplice. La persistenza assume che il valore del giorno successivo sia uguale a quello di oggi: un modello più complesso è utile solo se migliora il baseline su periodi non usati per l'addestramento.

## NWP e soluzione ibrida

Un modello NWP risolve numericamente equazioni fisiche dell'atmosfera usando condizioni iniziali, osservazioni e schemi di discretizzazione. La sua realizzazione richiederebbe modelli, dati e risorse computazionali che non fanno parte di questo prototipo. Il progetto usa quindi la previsione NWP operativa restituita da Open-Meteo e la mantiene distinguibile dall'output ML.

Il blend è una media pesata `w * ML + (1 - w) * NWP`. Il peso è configurabile. Senza previsioni NWP archiviate riferite agli stessi istanti e agli stessi valori osservati, il peso non può essere calibrato correttamente e la qualità storica dell'ibrido non è dimostrata.

## Feature engineering e leakage

L'aggregazione converte le misure orarie in riepiloghi giornalieri. I lag forniscono al modello una memoria del tempo recente; seno e coseno del giorno dell'anno rappresentano la ciclicità annuale evitando una discontinuità artificiale tra fine e inizio anno.

Il leakage si verifica quando in una feature entra informazione non disponibile al momento in cui la previsione viene emessa. Per evitarlo, l'esempio del giorno `D` usa i dati di `D` e dei giorni precedenti per prevedere `D+2`; la divisione train/validation/test mantiene l'ordine temporale.

Una validazione **rolling espandente** addestra e valuta ripetutamente su finestre cronologiche successive: ogni training precede la rispettiva validation e comprende solo target già disponibili al suo inizio. Le previsioni archiviate a un anticipo nominale fisso possono fornire feature NWP, ma l'anticipo rispetto all'ora valida non certifica l'ora effettiva di pubblicazione o l'assenza di revisioni successive.

## Metriche

- **MAE**: media di `|previsione - osservazione|`; intuitiva e robusta rispetto agli outlier rispetto alla metrica quadratica.
- **RMSE**: radice della media degli errori al quadrato; espressa nell'unità del target e sensibile agli errori ampi.
- **R²**: confronto con una predizione costante pari alla media del target; non è una percentuale di accuratezza e può essere inferiore a zero.

La matrice di confusione è appropriata per classi (ad esempio classi di rischio pioggia definite con soglie esplicite), non per temperatura o quantità di precipitazione usate come valori continui.

Nell'esperimento a due fasi, un classificatore stima la probabilità di precipitazione maggiore di 0 mm e un regressore stima la quantità nei giorni piovosi. La matrice di confusione misura le previsioni binarie ottenute applicando una soglia alla probabilità; il **Brier score** è la media dell'errore quadratico tra probabilità prevista ed evento osservato (0 o 1): più è basso, meglio è. Il MAE condizionato valuta la quantità prevista soltanto sui giorni effettivamente piovosi e non misura gli errori sui giorni asciutti.

## ONNX, quantizzazione ed edge computing

ONNX rappresenta un modello con un grafo di operatori standardizzati; ONNX Runtime esegue tale grafo senza richiedere l'intero stack di training. L'esportazione deve essere verificata confrontando le predizioni ONNX con quelle originali su un insieme di input rappresentativo.

La quantizzazione rappresenta alcuni parametri con precisione ridotta (per esempio interi a 8 bit), con l'obiettivo di ridurre memoria o accelerare l'inferenza. Non tutti gli operatori e modelli sono quantizzabili: l'esportazione deve quindi controllare il supporto, confrontare gli output e non promuovere artefatti degradati. Per i modelli ad alberi usati qui la quantizzazione potrebbe non essere supportata dal runtime.

Un dispositivo edge esegue l'inferenza vicino alla sorgente dei dati, con vincoli di CPU, memoria, energia e connettività. Il file ONNX e l'inferenza Python sono due opzioni distinte; per operare senza rete occorre inoltre rendere disponibili localmente gli input, i modelli e le dipendenze runtime. La chiamata al servizio NWP remoto non è offline.
