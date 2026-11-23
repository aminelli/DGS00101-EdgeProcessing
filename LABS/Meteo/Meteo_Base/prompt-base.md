# Sviluppo di modelli AI per previsioni Meteo

## Contesto
Il contesto è previsioni meteo per la città di Roma in Italia.

## Obiettivo
L'obiettivo è realizzare un sistema in grado di prevedere le condizioni metereologiche di una determinata località come ad esempio Roma in Italia.


## Datasets
Come dataset storico usa i dati del servizio `Open-Meteo (Historical Weather API)`.
Recupera gli ultimi 3 anni di dati storici su Roma.


## Approcci metodologici

Devi implementare due approcci:
1. L'approccio per la generazione del modello deve essere "Machine Leaning" metereologico: il modello deve apprendere da dati storici, dalle osservazioni e dalle variabili atmosferiche.
2. Numerical weather Prediction (NWP); utilizza modelli matematici e fisici che simulano l'evoluzione atmosferica.

Dei gestire la seguente architetura logica:
- Data Source: Acquisizione dati dalla sorgente Open-Meteo
- Data Engineering: Clean, normalizzazione dati etc., più costruzione delle features
- Training
- Validation e Testing
- Inference

## Output

Codice da generare:

- Genera uno script python per:
  - Acquisire i dati storici su Roma degli ultimi 3 anni.
  - Genera uno script python per gestire la pulizia e la normalizzazione
  - Estrazione delle features
  - Salva tutti i dati in formato CSV
- Genera uno script python per il training:
  - usa la combinazione dei seguenti 2 approcci:
    - Machine Leaning metereologico
    - Numerical weather Prediction (NWP)
  - Valutazione dei modelli generati e delle features, matrice di confusione e tutto quello che è necessario per individuare la precisione dei modelli
  - Per i modelli addestrati prevedi:
    - Salvataggio modelli
    - Salvataggio modelli con quantizzazione per l'uso dei modelli su EDGE AI device:
      - Prevedi estrazione formato ONNX con uso in ONNX Runtime
      - Prevedi estrazione per esecuzione diretta su Edge device che consentono l'esecuzione di codice python
- Genera un'app web-based python per testare il modello, in particolare devi generare:   
  - Strato API costruito sul modello
  - GUI per testare il modello

# Tecnologie

- Python
- SDK e/o frameworks python per AI: Scegli tu le librerie migliori per lo scenario e il contesto descritto.


## Note generali

Usa una nomenclatura inglese per il codice, script etc. 
Commenta in italiano il codice con un approccio documentale chiaro ed esplicativo sullo scope.

Genera un dictionary e documentazione in markdown di tutte le sigle che hai adottato con relativa spiegazione; 
fai lo stesso per i topics toccati di maggior interesse e sui dettagli tecnologici; questo file deve spiegare gli aspetti tecnologici (non è quindi la documentazione del progetto python)

Genera in markdown una documentazione di porogetto in italiano che sia dettagliata e contenente tutto il necessario (comandi, info utili etc.); questo deve fungere da documentazione di progetto.

Sii preciso, tecnico, e spiega bene i passaggi.

Se hai bisogno di altro per aspetti che non sono riuscito a descrivere o che ho descritto male, ponimi le domande prima di iniziare a generare il tutto.