genera un prompt completo e approfondito utile per AIGen (esempio githucopilot) per realizzare un progetto python che come obiettivo abbia la generazione di un modello secondo le ultime specifiche che mi hai fornito precedentemente; alcune precisazioni:

- resta su datase Sentinel-1 e/o Sentinel-2
- usa xView3
- Valuterei i tre modelli che hai proposto ovvero YOLO,Faster R-CNN, DETR
- Voglio fare statistica sulle prestazioni e ottimizzazioni in modo approfondito

----
Ouput:
https://chatgpt.com/s/t_6b05a59870b48191bbb22e959bef553a


Ti propongo un prompt pensato per essere utilizzato con **GitHub Copilot Agent, Claude Code o un altro AIGen con capacità di generazione e gestione di progetti software complessi**.

L'obiettivo non è semplicemente creare uno script di training, ma sviluppare una vera piattaforma sperimentale Python per la progettazione, l'addestramento, il confronto, la valutazione e l'ottimizzazione di modelli AI dedicati alla **Ship Detection su immagini satellitari Copernicus Sentinel-1 e, come possibile estensione, Sentinel-2**.

Ho dato particolare importanza a:

- Confronto scientificamente corretto tra YOLO, Faster R-CNN e DETR.
- Utilizzo di xView3 come dataset principale.
- Analisi statistica approfondita delle prestazioni.
- Hyperparameter tuning e ottimizzazione.
- Analisi degli errori e generalizzazione geografica.
- Riproducibilità degli esperimenti.
- Produzione di report, grafici e confronti quantitativi.

Una precisazione progettuale: xView3 è un dataset basato su immagini SAR Sentinel-1. Non bisogna quindi considerare automaticamente le sue annotazioni applicabili a Sentinel-2. L'eventuale supporto Sentinel-2 dovrà essere implementato come pipeline separata, con dati e annotazioni ottiche compatibili.

# **Prompt completo per AIGen**

# PROGETTO: SENTINEL VESSEL AI

## Sviluppo di una piattaforma Python per Ship Detection mediante immagini satellitari Copernicus Sentinel-1 e Sentinel-2

---

# 1. RUOLO E COMPETENZE RICHIESTE

Agisci come un Senior AI/ML Engineer, Computer Vision Engineer, Remote Sensing Specialist e Python Software Architect.

Devi progettare e sviluppare un progetto software Python completo, modulare, documentato e riproducibile, finalizzato alla realizzazione e alla valutazione sperimentale di modelli di Artificial Intelligence per il rilevamento di imbarcazioni (Ship/Vessel Detection) attraverso immagini satellitari.

Il progetto dovrà utilizzare principalmente immagini SAR provenienti dal programma Copernicus Sentinel-1 e il dataset pubblico xView3.

Dovrai inoltre predisporre un'architettura estendibile per supportare immagini ottiche Sentinel-2, mantenendo separate le pipeline di acquisizione, preprocessing, annotazione e addestramento quando i dataset non sono direttamente compatibili.

Non limitarti alla generazione di codice dimostrativo o di semplici notebook.

Realizza un'applicazione Python strutturata, con:

- Gestione dei dataset.
- Data preprocessing.
- Data augmentation.
- Gestione delle annotazioni.
- Addestramento dei modelli.
- Validazione.
- Test.
- Analisi statistica.
- Hyperparameter optimization.
- Benchmark.
- Analisi comparativa.
- Generazione automatica di report.
- Esportazione dei modelli addestrati.

Il risultato dovrà essere utilizzabile per successive sperimentazioni e per lo sviluppo di un sistema operativo di Maritime Surveillance.

# 2. OBIETTIVI DEL PROGETTO

L'obiettivo principale è sviluppare un sistema AI capace di:

1. Identificare imbarcazioni all'interno di immagini satellitari SAR Sentinel-1.
2. Individuare la posizione dei target nell'immagine.
3. Associare a ogni rilevamento un confidence score.
4. Stimare, quando supportato dalle annotazioni, la lunghezza dell'imbarcazione.
5. Classificare le imbarcazioni secondo le categorie disponibili nel dataset.
6. Distinguere i target dalle caratteristiche di disturbo presenti nelle immagini radar.
7. Trasformare le coordinate dei rilevamenti in coordinate geografiche.
8. Confrontare le prestazioni di differenti architetture di Deep Learning.
9. Ottimizzare automaticamente gli iperparametri.
10. Analizzare statisticamente i risultati degli esperimenti.
11. Identificare le condizioni nelle quali ciascun modello presenta le migliori prestazioni.
12. Valutare la capacità di generalizzazione su immagini geograficamente e temporalmente differenti da quelle utilizzate durante il training.

Il progetto dovrà concentrarsi inizialmente sulla Ship Detection.

La classificazione avanzata delle tipologie di imbarcazione, la correlazione AIS e il rilevamento di dark vessels dovranno essere predisposti come possibili sviluppi successivi, senza attribuire al dataset informazioni che non contiene.

# 3. DATASET E FONTI INFORMATIVE

## 3.1 Dataset principale: xView3

Utilizzare xView3 come dataset principale per l'addestramento e la valutazione dei modelli.

Fonte:\
[https://iuu.xview3.net/](https://iuu.xview3.net/)

Il dataset utilizza immagini SAR Sentinel-1 e contiene annotazioni relative ai target marittimi.

Prima di implementare il DataLoader:

- Analizzare la documentazione ufficiale.
- Analizzare la struttura effettiva dei file.
- Identificare i formati delle immagini.
- Identificare i metadati disponibili.
- Identificare le annotazioni.
- Identificare il sistema di coordinate.
- Identificare le polarizzazioni disponibili.
- Verificare le informazioni relative alle dimensioni dei target.
- Verificare le informazioni relative alla classificazione dei target.
- Verificare le informazioni relative alla confidenza delle annotazioni.
- Verificare le modalità ufficiali di valutazione del benchmark.

Non inventare campi, attributi, categorie o formati.

Le annotazioni originali devono essere conservate integralmente.

Creare un modulo specifico per la lettura e la normalizzazione delle annotazioni xView3.

## 3.2 Gestione delle annotazioni

Implementare un sistema di conversione delle annotazioni verso i formati richiesti dalle diverse architetture.

Prevedere almeno:

- Formato originale xView3.
- Formato interno normalizzato.
- Formato YOLO.
- Formato COCO per Faster R-CNN.
- Formato richiesto dall'implementazione DETR scelta.

Prestare particolare attenzione al fatto che xView3 utilizza annotazioni e metriche specifiche per la detection marittima.

Non assumere che tutte le annotazioni siano bounding box.

Se il modello richiede bounding box e il dataset fornisce coordinate puntuali:

1. Definire una strategia documentata di conversione.
2. Utilizzare le informazioni disponibili sulla dimensione dei target.
3. Specificare chiaramente le eventuali approssimazioni geometriche.
4. Conservare le coordinate originali.
5. Evitare di introdurre dimensioni arbitrarie non supportate dai dati.
6. Valutare l'impatto della conversione sulle prestazioni.
7. Mantenere una modalità di valutazione basata sulle annotazioni originali.

La conversione delle annotazioni non deve alterare il benchmark originale.

## 3.3 Sentinel-1

Prevedere il supporto per prodotti Sentinel-1 GRD.

Caratteristiche da considerare:

- Modalità IW.
- Polarizzazione VV.
- Polarizzazione VH, quando disponibile.
- Prodotti GRD.
- Coefficienti di backscatter.
- Sigma0.
- Georeferenziazione.
- Metadati di acquisizione.
- Risoluzione spaziale.
- Incidenza radar.
- Orbit direction.

Il sistema dovrà poter gestire immagini con uno o più canali radar.

Prevedere la possibilità di utilizzare:

- Solo VV.
- Solo VH.
- VV + VH.
- VV + VH + eventuali feature derivate, se giustificate sperimentalmente.

Non assumere che tutti i prodotti Sentinel-1 contengano le medesime polarizzazioni.

## 3.4 Sentinel-2

Predisporre un'estensione architetturale per immagini Sentinel-2.

Questa estensione dovrà essere separata dalla pipeline xView3.

Prevedere:

- Prodotti Sentinel-2 L2A.
- Bande RGB.
- Banda NIR.
- Eventuali bande aggiuntive.
- Risoluzioni differenti delle bande.
- Normalizzazione radiometrica.
- Gestione della copertura nuvolosa.
- Georeferenziazione.

Non utilizzare automaticamente le annotazioni SAR di xView3 come ground truth per immagini Sentinel-2.

L'addestramento ottico dovrà essere attivato solamente quando saranno disponibili immagini ottiche e annotazioni compatibili.

Prevedere una futura architettura multimodale SAR + Optical, senza implementare fusioni non supportate dai dati disponibili.

# 4. ARCHITETTURE DI DEEP LEARNING DA CONFRONTARE

Implementare tre famiglie di modelli.

Tutti i modelli dovranno essere addestrati e valutati utilizzando gli stessi split dei dati, quando tecnicamente possibile.

La conversione delle annotazioni e le eventuali differenze metodologiche dovranno essere esplicitamente documentate.

## 4.1 YOLO

Utilizzare un'implementazione YOLO moderna e mantenuta.

Valutare almeno due configurazioni dimensionali del modello, ad esempio una versione small e una medium.

Obiettivi:

- Ship Detection.
- Bounding box prediction.
- Confidence score.
- Eventuale classificazione delle categorie supportate dalle annotazioni.

Prevedere:

- Transfer learning.
- Fine tuning.
- Training da zero come esperimento opzionale.
- Gestione di immagini multicanale.
- Ottimizzazione delle dimensioni delle immagini.
- Ottimizzazione del batch size.
- Ottimizzazione delle soglie di confidence.
- Ottimizzazione delle soglie di NMS.
- Gestione di piccoli oggetti.

Non limitare il progetto all'utilizzo dei pesi pre-addestrati RGB.

Verificare la compatibilità dei pesi iniziali con gli input SAR e documentare le eventuali modifiche al primo layer convoluzionale.

## 4.2 Faster R-CNN

Utilizzare PyTorch e torchvision.

Prevedere:

- Backbone ResNet-50 con FPN.
- Feature extraction multiscala.
- Region Proposal Network.
- ROI pooling.
- Detection Head.
- Bounding box regression.
- Confidence score.

Valutare:

- Transfer learning.
- Differenti configurazioni del backbone.
- Numero di region proposals.
- Dimensioni degli anchor.
- Aspect ratio.
- Numero di region proposal utilizzate.
- Batch size.
- Learning rate.
- Ottimizzatore.
- Scheduler.

Gli anchor dovranno essere analizzati in funzione delle dimensioni effettive dei target nel dataset.

Non utilizzare configurazioni standard senza verificare la loro adeguatezza alle dimensioni delle imbarcazioni.

## 4.3 DETR

Implementare un modello basato su Detection Transformer.

Utilizzare un'implementazione mantenuta e documentata.

Valutare una configurazione DETR standard e, se compatibile con l'ambiente software, una variante più efficiente come Deformable DETR.

Prevedere:

- Backbone.
- Transformer Encoder.
- Transformer Decoder.
- Object Queries.
- Prediction Heads.
- Hungarian Matching.
- Classification Loss.
- Bounding Box Loss.

Valutare:

- Numero di object queries.
- Dimensione del backbone.
- Numero di encoder e decoder layers.
- Learning rate.
- Batch size.
- Weight decay.
- Numero di epoche.
- Strategie di augmentation.

Documentare eventuali limitazioni relative al rilevamento di target molto piccoli e alla convergenza del modello.

# 5. PREPROCESSING DELLE IMMAGINI

Implementare una pipeline di preprocessing configurabile.

Per i prodotti Sentinel-1 GRD prevedere, quando applicabili:

1. Apply Orbit File.
2. Thermal Noise Removal.
3. Radiometric Calibration.
4. Speckle Filtering opzionale.
5. Terrain Correction.
6. Conversione in GeoTIFF.
7. Normalizzazione radiometrica.
8. Preparazione dei canali.
9. Image Tiling.

Utilizzare ESA SNAP per le operazioni SAR che richiedono le elaborazioni specifiche dei prodotti Sentinel-1.

Integrare SNAP tramite Graph Processing Tool (GPT), evitando dipendenze da operazioni manuali nella GUI.

Non applicare automaticamente tutte le operazioni a prodotti già preprocessati.

Verificare la provenienza dei dati e lo stato delle elaborazioni per evitare doppi preprocessing.

## 5.1 Tiling

Implementare una pipeline per suddividere le immagini in patch.

Prevedere dimensioni configurabili:

- 256 × 256.
- 512 × 512.
- 1024 × 1024.

Prevedere:

- Overlapping configurabile.
- Gestione dei bordi.
- Conservazione delle coordinate geografiche.
- Conservazione delle annotazioni.
- Eliminazione controllata delle patch prive di target.
- Controllo della distribuzione dei target tra le patch.

Valutare sperimentalmente l'impatto delle differenti dimensioni delle patch.

Non eseguire lo split del dataset dopo il tiling in modo da rischiare che patch provenienti dalla stessa immagine originale finiscano in training e test.

Lo split deve avvenire a livello di scena o acquisizione originale.

# 6. DATA AUGMENTATION

Implementare tecniche di augmentation appropriate alle immagini SAR.

Valutare:

- Rotazioni di 90, 180 e 270 gradi.
- Flip orizzontale.
- Flip verticale.
- Random crop.
- Random scaling.
- Random brightness.
- Random contrast.
- Noise injection.
- Speckle noise.
- Eventuali trasformazioni radiometriche.

Non applicare indiscriminatamente trasformazioni pensate per immagini RGB.

Ogni augmentation dovrà essere configurabile e disattivabile.

Le annotazioni dovranno essere trasformate coerentemente con le immagini.

Prevedere esperimenti comparativi per determinare quali augmentation migliorano effettivamente le prestazioni.

# 7. DATASET SPLITTING

Implementare una suddivisione riproducibile:

- Training set.
- Validation set.
- Test set.

Prevedere inizialmente una suddivisione indicativa:

- 70% Training.
- 15% Validation.
- 15% Test.

Le percentuali dovranno essere configurabili.

Lo split dovrà avvenire a livello di scena geografica e/o acquisizione, non a livello di singola patch.

Prevenire:

- Data leakage geografico.
- Data leakage temporale.
- Duplicazione delle immagini.
- Sovrapposizione delle patch tra differenti split.

Prevedere un ulteriore scenario di test geografico, quando i dati lo consentono.

Il test set dovrà rimanere escluso da qualsiasi operazione di ottimizzazione degli iperparametri.

# 8. TRAINING MANAGEMENT

Realizzare un sistema centralizzato per gestire gli esperimenti di training.

Parametri configurabili:

- Model architecture.
- Model variant.
- Batch size.
- Epochs.
- Learning rate.
- Optimizer.
- Weight decay.
- Scheduler.
- Input image size.
- Number of workers.
- Mixed precision.
- Gradient clipping.
- Early stopping.
- Data augmentation.
- Loss functions.
- Confidence threshold.

Prevedere:

- Checkpoint periodici.
- Best model checkpoint.
- Last model checkpoint.
- Ripresa del training.
- Logging delle metriche.
- Salvataggio delle configurazioni.
- Registrazione dei seed.
- Registrazione delle versioni delle librerie.
- Gestione degli errori.
- Monitoraggio della GPU e della memoria.

Supportare CUDA quando disponibile e CPU come modalità alternativa.

Prevedere Mixed Precision Training (AMP).

# 9. VALUTAZIONE DELLE PRESTAZIONI

Questa sezione è particolarmente importante.

Non limitare la valutazione a una singola metrica.

Realizzare un sistema approfondito di valutazione quantitativa e statistica.

## 9.1 Metriche di Object Detection

Calcolare, quando applicabili:

- Precision.
- Recall.
- F1-score.
- Average Precision.
- AP50.
- AP75.
- mAP50-95.
- False Positive Rate.
- False Negative Rate.
- Number of True Positives.
- Number of False Positives.
- Number of False Negatives.

Per la valutazione xView3 implementare anche le metriche ufficiali del benchmark, rispettando le relative definizioni e modalità di matching.

Non confondere le metriche standard di object detection con quelle ufficiali xView3.

Documentare eventuali differenze tra le due valutazioni.

## 9.2 Analisi dei piccoli oggetti

Considerando che le imbarcazioni possono occupare una porzione molto ridotta dell'immagine, prevedere un'analisi specifica per dimensione del target.

Creare fasce dimensionali configurabili.

Analizzare:

- Precision per dimensione.
- Recall per dimensione.
- AP per dimensione, quando applicabile.
- Errori di localizzazione.
- Falsi positivi su clutter radar.

Verificare se un modello è particolarmente efficace nel rilevamento di target piccoli.

## 9.3 Analisi della localizzazione

Valutare:

- Errore medio di localizzazione.
- Errore mediano.
- Distribuzione degli errori.
- Distanza tra centro del target previsto e centro del ground truth.
- Errore espresso in pixel.
- Errore espresso in metri, quando la georeferenziazione lo consente.

Prevedere la valutazione delle coordinate geografiche.

## 9.4 Metriche operative

Misurare:

- Training time.
- Inference time.
- Throughput.
- Images per second.
- GPU memory consumption.
- CPU utilization.
- Model size.
- Numero di parametri.
- FLOPs, quando calcolabili.

Distinguere il tempo di preprocessing dal tempo di inference del modello.

## 9.5 Analisi delle soglie

Analizzare il comportamento dei modelli al variare della confidence threshold.

Produrre:

- Precision-Recall Curve.
- F1-Confidence Curve.
- Precision-Confidence Curve.
- Recall-Confidence Curve.

Individuare le soglie ottimali in funzione degli obiettivi operativi.

Non utilizzare il test set per selezionare le soglie.

# 10. ANALISI STATISTICA APPROFONDITA

Implementare un modulo specifico per l'analisi statistica dei risultati.

Non limitarsi a confrontare i valori medi delle metriche.

Prevedere:

## 10.1 Statistiche descrittive

Calcolare:

- Mean.
- Median.
- Standard deviation.
- Variance.
- Minimum.
- Maximum.
- Percentiles.
- Interquartile range.

Applicare le statistiche alle metriche di valutazione, ai tempi di training e ai tempi di inference.

## 10.2 Ripetibilità degli esperimenti

Eseguire esperimenti ripetuti utilizzando differenti random seed.

Prevedere almeno 3 seed per gli esperimenti di confronto principali, con possibilità di aumentare il numero.

Registrare:

- Risultati per ogni seed.
- Media.
- Deviazione standard.
- Intervalli di confidenza.
- Variabilità delle prestazioni.

## 10.3 Intervalli di confidenza

Calcolare intervalli di confidenza al 95%.

Utilizzare bootstrap quando appropriato.

Specificare l'unità di ricampionamento, evitando di trattare patch correlate della stessa scena come osservazioni indipendenti.

## 10.4 Test statistici

Implementare, quando le ipotesi statistiche sono soddisfatte:

- Shapiro-Wilk.
- Levene.
- Paired t-test.
- Wilcoxon signed-rank.
- Mann-Whitney U.
- ANOVA.
- Kruskal-Wallis.

Per i confronti multipli prevedere correzioni come Holm-Bonferroni.

Non applicare automaticamente tutti i test.

Selezionare il test in funzione del disegno sperimentale e della distribuzione dei dati.

Documentare:

- Ipotesi nulla.
- Ipotesi alternativa.
- p-value.
- Effect size.
- Interpretazione del risultato.

Non considerare una differenza statisticamente significativa automaticamente rilevante dal punto di vista operativo.

## 10.5 Correlazioni

Analizzare eventuali correlazioni tra:

- Dimensione delle immagini.
- Dimensione dei target.
- Polarizzazione.
- Numero di target.
- Condizioni di acquisizione.
- Confidence score.
- Prestazioni dei modelli.

Utilizzare Pearson o Spearman in funzione della natura dei dati.

## 10.6 Error Analysis

Realizzare un sistema di analisi degli errori.

Identificare:

- Falsi positivi.
- Falsi negativi.
- Target mancati.
- Detection duplicate.
- Errori di localizzazione.
- Confusione tra target e clutter.
- Prestazioni differenti in contesti costieri e offshore.

Generare una raccolta di immagini contenenti esempi di errori per ogni modello.

# 11. HYPERPARAMETER OPTIMIZATION

Implementare un sistema di ottimizzazione automatica degli iperparametri.

Valutare l'utilizzo di Optuna.

Prevedere:

- Grid Search per esperimenti controllati.
- Random Search.
- Bayesian Optimization.
- TPE Sampler.
- Pruning degli esperimenti non promettenti.

Parametri da ottimizzare:

- Learning rate.
- Batch size.
- Weight decay.
- Optimizer.
- Scheduler.
- Input image size.
- Confidence threshold.
- NMS threshold.
- Data augmentation parameters.
- Backbone configuration.
- Anchor configuration.
- Number of epochs.
- Number of workers.

Definire spazi di ricerca specifici per ciascuna architettura.

Non utilizzare necessariamente gli stessi iperparametri per tutti i modelli.

Prevedere:

- Salvataggio dei trial.
- Ranking degli esperimenti.
- Best trial.
- Parametri ottimali.
- Grafici di convergenza.
- Analisi della relazione tra iperparametri e metriche.

Separare nettamente l'ottimizzazione dal test finale.

# 12. ANALISI COMPARATIVA DEI MODELLI

Realizzare un sistema di benchmark che confronti YOLO, Faster R-CNN e DETR.

Il confronto dovrà includere:

| Area           | Indicatori                     |
| -------------- | ------------------------------ |
| Detection      | Precision, Recall, F1          |
| Accuracy       | AP, mAP e metriche xView3      |
| Small objects  | AP e Recall per dimensione     |
| Localization   | Errore in pixel e metri        |
| Performance    | Inference time, throughput     |
| Training       | Training time, epoche          |
| Hardware       | GPU memory, CPU usage          |
| Complexity     | Parameters, FLOPs, model size  |
| Robustness     | Variabilità tra seed e scenari |
| Generalization | Prestazioni su scene non viste |

Generare classifiche basate su differenti criteri.

Non definire automaticamente un unico modello vincitore.

Consentire di identificare:

- Modello con migliore Recall.
- Modello con migliore Precision.
- Modello con migliore F1.
- Modello con migliore AP.
- Modello con migliori prestazioni sui piccoli target.
- Modello con minore latenza.
- Modello con minore consumo di memoria.
- Modello con migliore stabilità statistica.

Prevedere una valutazione multi-obiettivo.

# 13. VISUALIZZAZIONE DEI RISULTATI

Implementare un modulo dedicato alla generazione dei grafici.

Utilizzare Matplotlib, Seaborn e Plotly, quando appropriato.

Prevedere almeno:

1. Training Loss Curve.
2. Validation Loss Curve.
3. Precision-Recall Curve.
4. F1 Curve.
5. Confusion Matrix.
6. AP Comparison.
7. Model Performance Comparison.
8. Training Time Comparison.
9. Inference Time Comparison.
10. GPU Memory Comparison.
11. Hyperparameter Importance.
12. Optuna Optimization History.
13. Metric Distribution.
14. Boxplot dei risultati tra differenti seed.
15. Confidence Interval Plot.
16. Error Distribution.
17. Detection Examples.
18. False Positive Examples.
19. False Negative Examples.
20. Performance by Target Size.

Prevedere grafici comparativi tra le tre architetture.

Ogni grafico dovrà essere esportabile in PNG e SVG.

# 14. ANALISI GEO-SPAZIALE

Implementare un modulo geospaziale basato su Rasterio, GeoPandas, Shapely e PyProj.

Obiettivi:

- Convertire le coordinate pixel in coordinate geografiche.
- Associare i rilevamenti al sistema di riferimento originale.
- Generare geometrie geografiche.
- Esportare i risultati in GeoJSON.
- Preservare i metadati di acquisizione.
- Visualizzare le imbarcazioni rilevate su una mappa.

Prevedere la possibilità di esportare:

- Coordinate WGS84.
- Bounding box.
- Punto centrale del target.
- Confidence score.
- Lunghezza stimata, quando disponibile.
- Identificativo dell'immagine.
- Data di acquisizione.

Non confondere le coordinate del prodotto radar originale con quelle del prodotto preprocessato.

# 15. MODEL OPTIMIZATION

Prevedere una fase specifica di ottimizzazione dei modelli addestrati.

Valutare:

- PyTorch model optimization.
- Mixed Precision.
- FP16.
- INT8 Quantization.
- Dynamic Quantization, quando supportata.
- Static Quantization, quando supportata.
- ONNX Export.
- ONNX Runtime.
- TensorRT, quando disponibile.
- Pruning.
- Knowledge Distillation.

Per ogni tecnica verificare:

- Riduzione delle dimensioni del modello.
- Riduzione della memoria utilizzata.
- Riduzione dei tempi di inference.
- Variazione di Precision.
- Variazione di Recall.
- Variazione di AP.
- Impatto sulle metriche ufficiali xView3.

Non applicare tecniche di ottimizzazione senza misurarne gli effetti.

Generare un report prima/dopo l'ottimizzazione.

# 16. MODEL EXPORT E INFERENCE

Prevedere l'esportazione dei modelli addestrati.

Formati:

- PyTorch (.pt o .pth).
- ONNX.
- Eventuale TensorRT.

Realizzare un modulo di inference indipendente dal training.

Il modulo dovrà ricevere un'immagine Sentinel-1 preprocessata e restituire:

- Target rilevati.
- Coordinate pixel.
- Bounding box o posizione del target secondo il modello.
- Confidence score.
- Eventuale classificazione.
- Coordinate geografiche.
- File GeoJSON.

Prevedere un sistema di configurazione delle soglie di detection.

Documentare le eventuali differenze tra preprocessing di training e preprocessing di inference.

# 17. STRUTTURA DEL PROGETTO

Organizzare il progetto secondo una struttura modulare simile alla seguente:

sentinel-vessel-ai/\
│\
├── configs/\
│   ├── dataset/\
│   ├── models/\
│   ├── training/\
│   ├── evaluation/\
│   └── optimization/\
│\
├── data/\
│   ├── raw/\
│   ├── processed/\
│   ├── annotations/\
│   ├── splits/\
│   └── tiles/\
│\
├── src/\
│   ├── data/\
│   │   ├── xview3_loader.py\
│   │   ├── sentinel1_loader.py\
│   │   ├── sentinel2_loader.py\
│   │   ├── annotation_converter.py\
│   │   ├── preprocessing.py\
│   │   ├── tiling.py\
│   │   └── augmentation.py\
│   │\
│   ├── models/\
│   │   ├── yolo/\
│   │   ├── faster_rcnn/\
│   │   ├── detr/\
│   │   └── model_factory.py\
│   │\
│   ├── training/\
│   │   ├── trainer.py\
│   │   ├── callbacks.py\
│   │   ├── checkpoint.py\
│   │   └── experiment_manager.py\
│   │\
│   ├── evaluation/\
│   │   ├── metrics.py\
│   │   ├── xview3_metrics.py\
│   │   ├── error_analysis.py\
│   │   ├── statistical_analysis.py\
│   │   └── benchmark.py\
│   │\
│   ├── optimization/\
│   │   ├── optuna_optimizer.py\
│   │   ├── pruning.py\
│   │   ├── quantization.py\
│   │   └── model_export.py\
│   │\
│   ├── geospatial/\
│   │   ├── coordinate_transform.py\
│   │   ├── geojson_export.py\
│   │   └── visualization.py\
│   │\
│   ├── inference/\
│   │   ├── predictor.py\
│   │   └── postprocessing.py\
│   │\
│   └── utils/\
│       ├── logging.py\
│       ├── reproducibility.py\
│       ├── configuration.py\
│       └── hardware_monitor.py\
│\
├── experiments/\
├── models/\
├── reports/\
│   ├── statistics/\
│   ├── benchmarks/\
│   ├── figures/\
│   └── model_comparison/\
│\
├── notebooks/\
├── scripts/\
├── tests/\
├── docs/\
│\
├── pyproject.toml\
├── requirements.txt\
├── README.md\
└── .gitignore

La struttura è indicativa e può essere migliorata in funzione delle esigenze tecniche.

Non inserire i dataset originali nel repository Git.

# 18. CONFIGURAZIONE E GESTIONE DEGLI ESPERIMENTI

Utilizzare file YAML per configurare:

- Dataset.
- Preprocessing.
- Architettura.
- Training.
- Evaluation.
- Optimization.
- Hardware.

Ogni esperimento dovrà avere un identificativo univoco.

Registrare:

- Data e ora.
- Architettura.
- Versione del codice.
- Dataset utilizzato.
- Split utilizzato.
- Configurazione.
- Random seed.
- Hardware.
- Versioni delle dipendenze.
- Metriche.
- Checkpoint.
- Log.

Valutare l'integrazione di MLflow per il tracking degli esperimenti.

Ogni esperimento dovrà essere riproducibile attraverso la propria configurazione.

# 19. TECNOLOGIE E LIBRERIE

Utilizzare Python 3.11 o versione compatibile con le dipendenze selezionate.

Valutare:

- PyTorch.
- torchvision.
- Ultralytics.
- Hugging Face Transformers, se necessario per DETR.
- Albumentations.
- Rasterio.
- GDAL.
- GeoPandas.
- Shapely.
- PyProj.
- NumPy.
- Pandas.
- SciPy.
- Scikit-learn.
- Matplotlib.
- Seaborn.
- Plotly.
- Optuna.
- MLflow.
- TensorBoard.
- PyYAML.
- Hydra, se utile per la configurazione.

ESA SNAP dovrà essere gestito come dipendenza esterna, verificandone l'installazione e la disponibilità del comando GPT.

Non introdurre dipendenze non necessarie.

Verificare la compatibilità delle versioni prima di generare il file di installazione.

# 20. TESTING E QUALITÀ DEL SOFTWARE

Realizzare test automatici utilizzando pytest.

Prevedere test per:

- Dataset loader.
- Lettura immagini.
- Lettura annotazioni.
- Conversione delle annotazioni.
- Preprocessing.
- Tiling.
- Coordinate geografiche.
- DataLoader.
- Forward pass dei modelli.
- Training su dataset ridotto.
- Calcolo metriche.
- Valutazione xView3.
- Export ONNX.
- Inference.
- Generazione GeoJSON.

Prevedere anche test di integrazione su un piccolo sottoinsieme del dataset.

Non eseguire il download dell'intero dataset durante i test automatici.

# 21. DOCUMENTAZIONE

Produrre documentazione tecnica completa.

README.md dovrà contenere:

- Descrizione del progetto.
- Architettura software.
- Requisiti hardware.
- Requisiti software.
- Installazione.
- Configurazione.
- Download e preparazione xView3.
- Preprocessing Sentinel-1.
- Training YOLO.
- Training Faster R-CNN.
- Training DETR.
- Evaluation.
- Statistical Analysis.
- Hyperparameter Optimization.
- Benchmark.
- Model Export.
- Inference.
- Troubleshooting.

Documentare inoltre:

- Formati dei dati.
- Struttura delle annotazioni.
- Conversioni effettuate.
- Metriche utilizzate.
- Limiti metodologici.
- Licenze dei dataset.
- Limiti di generalizzazione.

# 22. REPORT FINALE AUTOMATICO

Il progetto dovrà generare automaticamente un report comparativo degli esperimenti.

Il report dovrà includere:

1. Dataset utilizzato.
2. Numero di immagini.
3. Numero di target annotati.
4. Distribuzione dei target.
5. Configurazioni sperimentali.
6. Architetture utilizzate.
7. Iperparametri.
8. Risultati di training.
9. Metriche di validazione.
10. Metriche di test.
11. Metriche ufficiali xView3.
12. Analisi statistica.
13. Intervalli di confidenza.
14. Test statistici.
15. Analisi degli errori.
16. Confronto delle prestazioni.
17. Analisi dei consumi hardware.
18. Risultati dell'ottimizzazione.
19. Confronto prima/dopo ottimizzazione.
20. Considerazioni conclusive.

Prevedere esportazione in HTML e PDF.

# 23. MODALITÀ DI SVILUPPO RICHIESTA

Non generare tutto il progetto in un'unica operazione senza verifiche intermedie.

Procedere per fasi.

## Fase 1 – Analisi e progettazione

Prima di scrivere il codice:

- Analizzare il problema.
- Verificare le specifiche xView3.
- Verificare le librerie.
- Definire le architetture.
- Definire la gestione delle annotazioni.
- Definire la pipeline SAR.
- Definire le metriche.
- Definire il piano sperimentale.
- Definire la struttura del repository.

Produrre un documento ARCHITECTURE.md e un documento EXPERIMENT_PLAN.md.

Non inventare informazioni mancanti.

Evidenziare le decisioni progettuali che richiedono una scelta.

## Fase 2 – Dataset e preprocessing

Implementare:

- xView3 loader.
- Annotation parser.
- Annotation converter.
- Preprocessing Sentinel-1.
- Tiling.
- Dataset splitting.
- Data augmentation.

Verificare il funzionamento con un sottoinsieme reale del dataset.

## Fase 3 – Implementazione modelli

Implementare separatamente:

- YOLO.
- Faster R-CNN.
- DETR.

Verificare il forward pass e la compatibilità con i dati.

## Fase 4 – Training

Implementare il sistema centralizzato di training.

Eseguire inizialmente esperimenti ridotti per verificare l'intera pipeline.

## Fase 5 – Evaluation

Implementare metriche standard e metriche ufficiali xView3.

Verificare la correttezza dei risultati rispetto alle specifiche del benchmark.

## Fase 6 – Statistical Analysis

Implementare:

- Esperimenti ripetuti.
- Analisi descrittiva.
- Intervalli di confidenza.
- Test statistici.
- Analisi degli errori.
- Confronti tra architetture.

## Fase 7 – Optimization

Implementare Optuna e le tecniche di ottimizzazione dei modelli.

## Fase 8 – Benchmark e reporting

Generare grafici, classifiche, report e documentazione.

## Fase 9 – Inference ed export

Realizzare il modulo di inference e l'esportazione dei modelli.

# 24. REGOLE FONDAMENTALI

Rispettare rigorosamente le seguenti regole:

1. Non inventare dati, annotazioni o caratteristiche dei dataset.
2. Non inventare metriche o risultati sperimentali.
3. Non dichiarare completata un'operazione se non è stata effettivamente eseguita.
4. Non utilizzare dati sintetici come sostituti dei dati reali senza dichiararlo.
5. Non confondere Sentinel-1 SAR con Sentinel-2 Optical.
6. Non alterare le annotazioni originali xView3.
7. Non introdurre data leakage.
8. Non confrontare metriche metodologicamente incompatibili senza evidenziarlo.
9. Non utilizzare il test set durante l'ottimizzazione.
10. Non considerare il valore di mAP come unica misura della qualità del modello.
11. Non assumere che una configurazione standard sia ottimale per i piccoli target radar.
12. Non applicare trasformazioni alle immagini senza valutarne gli effetti.
13. Documentare tutte le approssimazioni.
14. Separare i risultati realmente misurati dalle stime teoriche.
15. Mantenere la riproducibilità degli esperimenti.

Se una funzionalità non può essere implementata per mancanza di dati, dipendenze o risorse hardware, predisporre l'interfaccia software e documentare chiaramente il limite.

# 25. CRITERI DI ACCETTAZIONE

Il progetto potrà essere considerato completo quando sarà possibile:

- Caricare il dataset xView3.
- Analizzare le annotazioni.
- Preparare i dati Sentinel-1.
- Creare gli split.
- Addestrare YOLO.
- Addestrare Faster R-CNN.
- Addestrare DETR.
- Valutare i tre modelli.
- Calcolare le metriche ufficiali xView3.
- Effettuare analisi statistiche.
- Ripetere gli esperimenti con differenti seed.
- Ottimizzare gli iperparametri.
- Confrontare i modelli.
- Analizzare i falsi positivi e falsi negativi.
- Misurare i tempi di inference.
- Ottimizzare i modelli.
- Esportare i modelli.
- Eseguire inference su nuove immagini Sentinel-1.
- Produrre risultati georeferenziati.
- Generare report automatici.

Il risultato finale dovrà essere un progetto Python professionale, modulare, estendibile e scientificamente verificabile.

La priorità assoluta è la correttezza metodologica degli esperimenti e la qualità delle analisi comparative, non la semplice produzione di un modello addestrato.

# **Indicazioni aggiuntive per ottenere risultati migliori da GitHub Copilot**

Ti suggerisco di non chiedere a Copilot di sviluppare tutto il progetto in una sola sessione. Utilizza il prompt precedente come specifica generale e poi procedi con richieste progressive.

## **1. Ordine di sviluppo consigliato**

1. **Analisi architetturale**

   Generazione di `ARCHITECTURE.md`, `EXPERIMENT_PLAN.md` e struttura del repository.

   Prima attività
2. **Dataset e preprocessing**

   Implementazione xView3 loader, annotazioni, preprocessing Sentinel-1 e tiling.
3. **Primo modello**

   Implementazione e validazione della pipeline YOLO.
4. **Modelli alternativi**

   Implementazione Faster R-CNN e DETR mantenendo gli stessi split.
5. **Benchmark statistico**

   Metriche, seed multipli, bootstrap, intervalli di confidenza e test statistici.
6. **Ottimizzazione**

   Optuna, ricerca degli iperparametri e ottimizzazione del modello.
7. **Reporting e deployment**

   Report comparativi, esportazione ONNX e inference geospaziale.

## **2. Prompt supplementare per l'analisi statistica**

Questo è un prompt aggiuntivo che utilizzerei dopo aver realizzato la prima versione funzionante del progetto.

Analizza approfonditamente il progetto Sentinel Vessel AI e potenzia il sottosistema di valutazione statistica e benchmark.

L'obiettivo è realizzare un framework sperimentale rigoroso per confrontare YOLO, Faster R-CNN e DETR nell'ambito della Ship Detection su immagini SAR Sentinel-1.

Non limitarti alle metriche standard.

Implementa:

1. Esperimenti ripetuti con almeno 5 random seed per i confronti definitivi.
2. Intervalli di confidenza al 95%.
3. Bootstrap con ricampionamento a livello di scena geografica.
4. Analisi della varianza delle metriche.
5. Test statistici appropriati al disegno sperimentale.
6. Effect size.
7. Correzione per confronti multipli.
8. Analisi della stabilità dei modelli.
9. Analisi della sensibilità agli iperparametri.
10. Analisi delle prestazioni per dimensione del target.
11. Analisi delle prestazioni per polarizzazione.
12. Analisi delle prestazioni per contesto geografico.
13. Analisi dei falsi positivi e falsi negativi.
14. Analisi della latenza e del consumo di memoria.
15. Confronto prima/dopo quantizzazione e pruning.

Implementa inoltre un sistema di ranking multi-obiettivo che consenta di confrontare i modelli secondo differenti priorità operative.

Prevedi almeno tre profili:

- Maximum Detection: privilegiare Recall e riduzione dei falsi negativi.
- Balanced Detection: bilanciare Precision e Recall.
- Efficient Detection: privilegiare latenza, throughput e consumo di memoria.

Per ogni esperimento conserva configurazione, seed, versione del codice, dataset, checkpoint e metriche.

Genera automaticamente grafici comparativi e report HTML/PDF.

Non utilizzare il test set per ottimizzare iperparametri o soglie.

Verifica che le osservazioni utilizzate nei test statistici siano indipendenti o che il modello statistico tenga conto della dipendenza tra immagini appartenenti alla stessa scena.

Non produrre risultati simulati o statistiche non derivanti da esperimenti effettivamente eseguiti.

Prima di modificare il codice, analizza l'architettura esistente e presenta un piano delle modifiche necessarie.

## **3. Alcune scelte tecniche che considero importanti**

Per evitare che il progetto diventi eccessivamente complesso già nella prima fase, adotterei queste scelte:

| Componente            | Scelta consigliata                         |
| --------------------- | ------------------------------------------ |
| Dataset principale    | xView3                                     |
| Sensore               | Sentinel-1                                 |
| Formato immagini      | GeoTIFF / formato originale xView3         |
| Framework DL          | PyTorch                                    |
| Object Detection      | YOLO, Faster R-CNN, DETR                   |
| Preprocessing SAR     | ESA SNAP + Rasterio                        |
| Data augmentation     | Albumentations                             |
| Hyperparameter tuning | Optuna                                     |
| Experiment tracking   | MLflow                                     |
| Metriche              | PyTorch + implementazione ufficiale xView3 |
| Analisi statistica    | SciPy + Scikit-learn                       |
| Visualizzazione       | Matplotlib + Plotly                        |
| Geospatial            | Rasterio + GeoPandas                       |
| Model export          | ONNX                                       |

**Una scelta metodologica che ritengo particolarmente importante:** non baserei il confronto esclusivamente su mAP50-95. xView3 ha un proprio paradigma di detection e metriche specifiche, che devono essere mantenute nel benchmark. Le metriche standard YOLO/COCO saranno complementari, non sostitutive.

## **4. Un ultimo suggerimento: prevedere due benchmark distinti**

**Benchmark A – Academic Performance**

Qualità della detection

- Metriche ufficiali xView3.
- Precision e Recall.
- F1-score.
- AP e mAP.
- Errori di localizzazione.
- Prestazioni sui piccoli target.
- Stabilità statistica.

**Benchmark B – Operational Performance**

Efficienza operativa

- Tempo di inference.
- Immagini elaborate al secondo.
- Consumo GPU.
- Consumo RAM.
- Dimensione del modello.
- Latenza.
- Prestazioni dopo quantizzazione.

Questo ti permetterà di capire non soltanto quale modello rileva meglio le imbarcazioni, ma anche quale modello è più adatto a un'eventuale implementazione operativa, ad esempio su una piattaforma Edge AI.

**La mia raccomandazione finale:** realizza inizialmente un benchmark rigoroso su xView3/Sentinel-1, con i tre modelli e una pipeline dati comune. Solo dopo aver identificato le differenze prestazionali introdurrei Sentinel-2 e la fusione SAR/Optical. In questo modo avrai risultati scientificamente confrontabili e non confonderai i miglioramenti dovuti all'architettura con quelli dovuti a dataset o sensori differenti.
