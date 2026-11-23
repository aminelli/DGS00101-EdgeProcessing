# Prompt: sistema ibrido di previsione meteorologica per Roma

## Ruolo e obiettivo

Agisci come un ingegnere software e data scientist specializzato in machine learning, meteorologia e sistemi edge. Progetta e implementa un progetto Python completo per prevedere le condizioni meteorologiche di una località, usando Roma (Italia) come caso iniziale.

Il progetto deve combinare:

1. **Machine learning meteorologico**, addestrato su dati storici e variabili atmosferiche.
2. **Numerical Weather Prediction (NWP)**, integrando previsioni prodotte da un modello NWP disponibile tramite un servizio esterno. Non presentare un modello addestrato sui soli dati storici come un sistema NWP: se non è realistico implementare la dinamica atmosferica da zero, usa i dati previsionali NWP di Open-Meteo e descrivi chiaramente il ruolo di ciascun componente nella soluzione ibrida.

Prima di scrivere codice, verifica che i requisiti siano sufficienti. Poni domande concise solo per le decisioni che impediscono una soluzione corretta (per esempio orizzonte di previsione, variabili da prevedere o vincoli obbligatori di deployment). Se non ricevi risposta e puoi procedere in modo ragionevole, dichiara le ipotesi adottate e rendile configurabili.

## Requisiti funzionali

### Dati

- Usa le API Open-Meteo:
  - **Historical Weather API** per acquisire i dati osservati/storici.
  - Un endpoint di previsione basato su NWP per ottenere le previsioni da usare nel componente ibrido.
- Scarica gli ultimi tre anni disponibili per Roma. Definisci le coordinate in un unico punto configurabile (default: latitudine `41.9028`, longitudine `12.4964`).
- Documenta gli endpoint, i parametri, le unità, il fuso orario e gli eventuali limiti o ritardi di disponibilità dei dati.
- Gestisci paginazione o intervalli temporali, errori HTTP, timeout, limiti del servizio e risposte incomplete senza perdere o duplicare silenziosamente i dati.
- Conserva i dati grezzi e salva i dati elaborati in CSV. Includi timestamp, località e unità delle variabili; usa un formato coerente e documentato.

### Data engineering e feature engineering

- Implementa controlli di qualità, gestione dei valori mancanti e duplicati, conversione di unità e ordinamento temporale.
- Costruisci feature utili, quali componenti temporali cicliche e lag, evitando di usare informazioni future rispetto al momento della previsione.
- Separa chiaramente i dati osservati dalle previsioni NWP e conserva la provenienza delle feature.
- Rendi la pipeline ripetibile: la stessa configurazione e gli stessi dati devono produrre gli stessi dataset e risultati, ove applicabile.

### Modelli e valutazione

- Definisci esplicitamente le variabili target, la granularità temporale e gli orizzonti di previsione; se non sono forniti, proponi valori iniziali sensati e documentali come ipotesi.
- Confronta almeno:
  1. un modello di riferimento semplice;
  2. un modello di machine learning addestrato sui dati storici;
  3. una soluzione ibrida che combina il modello ML con le previsioni NWP.
- Usa una suddivisione cronologica train/validation/test, senza mescolare casualmente le osservazioni temporali e senza leakage.
- Scegli metriche coerenti con il tipo di target: metriche di regressione per grandezze continue e metriche di classificazione, inclusa matrice di confusione, solo per target categoriali definiti. Riporta i risultati per orizzonte e variabile, confrontandoli con i baseline.
- Spiega limiti, incertezza e condizioni in cui i risultati potrebbero non generalizzare. Non dichiarare precisione o prestazioni non misurate.
- Salva modelli, configurazioni, metriche e artefatti necessari per riprodurre la valutazione.

### Inferenza, API e interfaccia web

- Crea un'applicazione web Python con:
  - un livello API per caricare e invocare il modello;
  - una GUI semplice per selezionare la località, richiedere una previsione e visualizzare risultati e informazioni essenziali.
- Definisci input e output API, validazione, gestione degli errori e istruzioni di avvio.
- Evita di esporre stack trace o dettagli interni agli utenti; registra gli errori in modo utile per il debug.
- Distingui i risultati del modello ML, i dati NWP in ingresso e l'eventuale previsione ibrida.

### Esportazione e uso su dispositivi edge

- Esporta il modello in ONNX e documenta l'inferenza con ONNX Runtime, quando il modello e le librerie selezionate lo consentono.
- Valuta la quantizzazione e confronta qualità, dimensione e prestazioni con il modello non quantizzato; non applicarla se non è supportata o peggiora in modo inaccettabile i risultati.
- Fornisci anche una modalità di esecuzione Python adatta a dispositivi edge compatibili.
- Indica dipendenze, prerequisiti hardware/software, passaggi di esportazione e limiti di compatibilità. Se un artefatto non può essere generato o verificato nell'ambiente disponibile, dichiaralo esplicitamente.

## Architettura richiesta

Organizza il progetto in componenti chiari, con responsabilità separate:

1. acquisizione dei dati;
2. pulizia e trasformazione;
3. feature engineering;
4. training e valutazione;
5. salvataggio ed esportazione dei modelli;
6. inferenza;
7. API e interfaccia web.

Scegli framework e librerie Python adatti, motivando brevemente le decisioni. Mantieni configurazioni, percorsi e coordinate fuori dalla logica applicativa quando opportuno. Includi dipendenze dichiarate, controlli degli input e logging essenziale. Non aggiungere complessità o servizi non necessari.

## Deliverable

Genera un progetto eseguibile che includa:

- script o moduli Python per acquisizione dati, data engineering e feature engineering;
- script o moduli Python per training, confronto dei modelli e valutazione;
- codice per esportazione e inferenza, inclusi gli artefatti ONNX/quantizzati quando realizzabili;
- applicazione web con API e GUI;
- file di configurazione di esempio e file delle dipendenze;
- test essenziali per le parti principali, inclusi i casi di errore più importanti;
- esempi di CSV o dati sintetici per consentire test senza dipendere sempre dalle API esterne, chiaramente identificati come sintetici;
- istruzioni per installazione, configurazione, esecuzione, training, test e avvio dell'applicazione.

## Documentazione

Scrivi la documentazione in italiano e includi:

1. **Documentazione del progetto**: architettura, struttura dei file, prerequisiti, configurazione, comandi, flussi operativi, API, test, troubleshooting e limiti noti.
2. **Guida tecnica**: glossario/dizionario delle sigle e spiegazione dei temi tecnologici rilevanti (ML meteorologico, NWP, feature engineering, metriche, ONNX, quantizzazione ed edge computing). Questa guida deve spiegare i concetti e le tecnologie, non duplicare la documentazione operativa del progetto.

## Convenzioni e qualità

- Usa nomi inglesi per file, moduli, funzioni, classi e variabili.
- Scrivi commenti e documentazione in italiano, con commenti nel codice solo dove chiariscono scopo o scelte non ovvie.
- Mantieni il codice modulare, tipizzato dove utile e conforme alle convenzioni Python.
- Gestisci gli errori in modo esplicito; non nascondere fallimenti con valori predefiniti ingannevoli.
- Verifica il progetto con i test e i controlli disponibili. Riporta i comandi eseguiti e l'esito, distinguendo ciò che è stato verificato da ciò che non è stato possibile verificare.
- Procedi per fasi: prima definisci ipotesi e architettura, poi implementa, testa e documenta. Al termine fornisci un riepilogo dei deliverable e delle limitazioni.
