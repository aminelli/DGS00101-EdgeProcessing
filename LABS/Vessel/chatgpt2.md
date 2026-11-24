link chatgpt:
https://chatgpt.com/s/t_6b05a73a7a288191a4f3891cc8835366

----

La modifica che suggerisco è ampliare il progetto trasformandolo da una semplice piattaforma di training e benchmarking in una **piattaforma di sviluppo, ottimizzazione e validazione di modelli AI destinati all'esecuzione su dispositivi Edge AI e piattaforme satellitari**.

Il punto fondamentale è che ogni modello ottimizzato dovrà essere confrontato con il corrispondente modello originale (Full Precision), misurando non soltanto la riduzione delle dimensioni, ma anche il degrado delle prestazioni di detection.

Aggiungerei quindi un sottosistema specifico denominato **Edge AI Model Optimization & Validation**, integrato con le pipeline YOLO, Faster R-CNN e DETR.

Di seguito trovi il prompt completo aggiornato, pronto per essere utilizzato con GitHub Copilot Agent o altri AIGen.

# PROGETTO: SENTINEL VESSEL AI

## Piattaforma Python per Ship Detection, Model Optimization, Model Compression e Edge AI Deployment su immagini satellitari Copernicus Sentinel-1 e Sentinel-2

---

# 1. RUOLO E COMPETENZE RICHIESTE

Agisci come un Senior AI/ML Engineer, Computer Vision Engineer, Remote Sensing Specialist, Edge AI Engineer e Python Software Architect.

Devi progettare e sviluppare un progetto software Python completo, modulare, documentato e riproducibile, finalizzato alla realizzazione, all'addestramento, alla valutazione e all'ottimizzazione di modelli di Artificial Intelligence per il rilevamento di imbarcazioni (Ship/Vessel Detection) attraverso immagini satellitari.

Il progetto dovrà utilizzare principalmente immagini SAR provenienti dal programma Copernicus Sentinel-1 e il dataset pubblico xView3.

Dovrà inoltre predisporre un'architettura estendibile per supportare immagini ottiche Sentinel-2, mantenendo separate le pipeline di acquisizione, preprocessing, annotazione e addestramento quando i dataset non sono direttamente compatibili.

Oltre alla realizzazione dei modelli di Deep Learning, il progetto dovrà implementare un sistema completo di compressione, quantizzazione, ottimizzazione e validazione dei modelli, con particolare attenzione al loro utilizzo su:

- Satelliti.
- Piattaforme satellitari con capacità di elaborazione onboard.
- Edge AI Devices.
- Embedded AI Systems.
- Sistemi ARM.
- NVIDIA Jetson.
- Sistemi basati su GPU embedded.
- Sistemi con acceleratori AI dedicati.
- Dispositivi con risorse computazionali e memoria limitate.

L'obiettivo non è soltanto ottenere il modello con la migliore capacità di detection, ma identificare il miglior compromesso tra accuratezza, dimensioni, memoria, latenza, consumo energetico e capacità computazionale.

Il progetto dovrà distinguere sempre:

1. Modello principale addestrato.
2. Modello ottimizzato.
3. Modello quantizzato.
4. Modello compattato.
5. Modello ottimizzato per uno specifico hardware target.

Ogni variante dovrà essere confrontata con il modello principale attraverso esperimenti quantitativi e riproducibili.

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
- Model compression.
- Model quantization.
- Model pruning.
- Knowledge distillation.
- Hardware-aware optimization.
- Edge AI benchmarking.
- Analisi comparativa.
- Generazione automatica di report.
- Esportazione dei modelli.
- Validazione delle versioni ottimizzate.

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
13. Generare versioni compattate dei modelli.
14. Generare versioni quantizzate dei modelli.
15. Generare modelli ottimizzati per differenti piattaforme hardware.
16. Misurare il degrado delle prestazioni causato dalle tecniche di compressione e quantizzazione.
17. Identificare il miglior compromesso tra accuratezza e risorse hardware.
18. Verificare la compatibilità dei modelli esportati con gli ambienti di esecuzione Edge AI.

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

Caratteristiche:

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

# 4. ARCHITETTURE DI DEEP LEARNING DA CONFRONTARE

Implementare tre famiglie di modelli.

Tutti i modelli dovranno essere addestrati e valutati utilizzando gli stessi split dei dati, quando tecnicamente possibile.

## 4.1 YOLO

Utilizzare un'implementazione YOLO moderna e mantenuta.

Valutare almeno due configurazioni dimensionali del modello, ad esempio una versione small e una medium.

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

Verificare la compatibilità dei pesi pre-addestrati con gli input SAR e documentare le eventuali modifiche al primo layer convoluzionale.

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

Valutare le dimensioni degli anchor in funzione delle dimensioni effettive dei target.

## 4.3 DETR

Implementare un modello basato su Detection Transformer.

Valutare una configurazione DETR standard e, se compatibile, una variante più efficiente come Deformable DETR.

Prevedere:

- Backbone.
- Transformer Encoder.
- Transformer Decoder.
- Object Queries.
- Prediction Heads.
- Hungarian Matching.
- Classification Loss.
- Bounding Box Loss.

Documentare eventuali limitazioni relative ai target molto piccoli e alla convergenza del modello.

# 5. PREPROCESSING DELLE IMMAGINI

Implementare una pipeline configurabile.

Per Sentinel-1 GRD prevedere, quando applicabili:

1. Apply Orbit File.
2. Thermal Noise Removal.
3. Radiometric Calibration.
4. Speckle Filtering opzionale.
5. Terrain Correction.
6. Conversione in GeoTIFF.
7. Normalizzazione radiometrica.
8. Preparazione dei canali.
9. Image Tiling.

Utilizzare ESA SNAP tramite Graph Processing Tool (GPT).

Non applicare automaticamente operazioni già effettuate durante un preprocessing precedente.

## 5.1 Tiling

Prevedere dimensioni configurabili:

- 256 × 256.
- 512 × 512.
- 1024 × 1024.

Implementare:

- Overlapping.
- Gestione dei bordi.
- Conservazione delle coordinate geografiche.
- Conservazione delle annotazioni.
- Controllo della distribuzione dei target.

Lo split del dataset dovrà avvenire a livello di scena o acquisizione originale, prima della generazione delle patch, per evitare data leakage.

# 6. DATA AUGMENTATION

Implementare augmentation appropriate alle immagini SAR:

- Rotazioni di 90, 180 e 270 gradi.
- Flip orizzontale.
- Flip verticale.
- Random crop.
- Random scaling.
- Random brightness.
- Random contrast.
- Noise injection.
- Speckle noise.

Ogni trasformazione dovrà essere configurabile e disattivabile.

Valutare sperimentalmente l'impatto delle augmentation sulle prestazioni.

# 7. DATASET SPLITTING

Implementare:

- Training set.
- Validation set.
- Test set.

Suddivisione iniziale indicativa:

- 70% Training.
- 15% Validation.
- 15% Test.

Prevenire data leakage geografico e temporale.

Il test set dovrà essere escluso dalle operazioni di hyperparameter tuning e model selection.

# 8. TRAINING MANAGEMENT

Realizzare un sistema centralizzato per gestire gli esperimenti.

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
- Monitoraggio GPU e memoria.

# 9. VALUTAZIONE DELLE PRESTAZIONI

Implementare metriche standard:

- Precision.
- Recall.
- F1-score.
- Average Precision.
- AP50.
- AP75.
- mAP50-95.
- False Positive Rate.
- False Negative Rate.
- True Positives.
- False Positives.
- False Negatives.

Implementare anche le metriche ufficiali xView3, rispettando le relative definizioni.

Prevedere analisi specifiche per:

- Dimensione del target.
- Polarizzazione.
- Contesto geografico.
- Contesto costiero/offshore.
- Localizzazione.
- Confidence threshold.

Misurare:

- Training time.
- Inference time.
- Throughput.
- GPU memory.
- CPU utilization.
- Model size.
- Numero di parametri.
- FLOPs.

# 10. ANALISI STATISTICA APPROFONDITA

Implementare:

- Mean.
- Median.
- Standard deviation.
- Variance.
- Percentiles.
- Interquartile range.
- Confidence intervals al 95%.
- Bootstrap a livello di scena.
- Shapiro-Wilk.
- Levene.
- Paired t-test.
- Wilcoxon signed-rank.
- Mann-Whitney U.
- ANOVA.
- Kruskal-Wallis.
- Holm-Bonferroni.
- Effect size.

Prevedere almeno 3 seed per gli esperimenti iniziali e almeno 5 seed per i confronti definitivi.

Documentare le ipotesi statistiche e non considerare automaticamente ogni differenza significativa come operativamente rilevante.

# 11. HYPERPARAMETER OPTIMIZATION

Implementare Optuna.

Prevedere:

- Grid Search.
- Random Search.
- Bayesian Optimization.
- TPE Sampler.
- Pruning.

Ottimizzare i parametri specifici di ogni architettura.

Salvare:

- Trial.
- Best trial.
- Ranking.
- Parametri ottimali.
- Optimization history.
- Hyperparameter importance.

Non utilizzare il test set per il tuning.

# 12. MODEL COMPRESSION & QUANTIZATION

Questa sezione è un requisito fondamentale del progetto.

Implementare un sottosistema dedicato alla generazione di modelli ottimizzati per dispositivi Edge AI e piattaforme satellitari.

Il modello principale addestrato e validato in Full Precision dovrà essere considerato il riferimento (Baseline Model).

A partire dal Baseline Model dovranno essere generate differenti varianti ottimizzate.

Ogni variante dovrà mantenere un collegamento esplicito al modello originale dal quale deriva.

## 12.1 Baseline Model

Prevedere una versione di riferimento:

- FP32.
- Pesi originali.
- Architettura originale.
- Configurazione di training registrata.
- Metriche complete.
- Checkpoint originale.
- Versione del codice.
- Dataset e split utilizzati.

Il Baseline Model non dovrà essere sovrascritto dalle operazioni di ottimizzazione.

Tutte le versioni successive dovranno essere confrontate con esso.

## 12.2 Quantizzazione

Implementare differenti strategie di quantizzazione.

### FP16

Floating Point 16-bit.

Valutare:

- Riduzione delle dimensioni dei pesi.
- Riduzione della memoria.
- Compatibilità hardware.
- Variazione della precisione.
- Variazione della latenza.

### BF16

Implementare quando supportato dall'architettura e dall'hardware target.

Verificare la reale compatibilità con il runtime di destinazione.

### INT8 PTQ

Post Training Quantization.

Implementare:

- Calibration Dataset.
- Calibration Procedure.
- Scale.
- Zero Point.
- Quantization Parameters.

Il calibration dataset dovrà essere rappresentativo delle immagini operative, senza utilizzare il test set.

### INT8 QAT

Quantization Aware Training.

Prevedere:

- Fine tuning con Fake Quantization.
- Quantization Simulation.
- Quantization Configuration.
- Validation.
- Export.

Confrontare i risultati con INT8 PTQ.

### INT4

Valutare la quantizzazione INT4 quando supportata dal runtime e dall'hardware target.

Documentare eventuali limitazioni per operatori, layer e architetture.

Non considerare automaticamente INT4 compatibile con tutti i modelli.

## 12.3 Model Pruning

Implementare tecniche di pruning.

Prevedere:

- Unstructured Pruning.
- Structured Pruning.
- Channel Pruning.
- Filter Pruning.
- Layer Pruning, quando applicabile.

Valutare differenti percentuali di pruning, ad esempio:

- 10%.
- 20%.
- 30%.
- 40%.
- 50%.

Per ogni configurazione misurare:

- Parametri rimossi.
- Dimensione effettiva del modello.
- Sparsity.
- Latenza.
- Memoria.
- Degrado delle metriche di detection.

Distinguere la sparsità teorica dalla riduzione effettiva dei tempi di inference.

Il pruning dovrà essere seguito da fine tuning quando necessario.

## 12.4 Knowledge Distillation

Implementare Knowledge Distillation per trasferire le capacità del modello principale verso un modello Student più compatto.

Prevedere:

- Teacher Model.
- Student Model.
- Soft Targets.
- Temperature.
- Distillation Loss.
- Detection Loss.
- Feature Distillation, quando applicabile.

Il Teacher Model dovrà essere uno dei modelli originali addestrati.

Valutare differenti architetture Student, privilegiando modelli con minore numero di parametri e minore costo computazionale.

Confrontare il modello Student con il Teacher.

Misurare quanto delle prestazioni originali viene mantenuto.

## 12.5 Modelli compattati

Valutare architetture leggere o varianti efficienti.

Esempi:

- YOLO Nano/Small.
- MobileNet Backbone.
- EfficientNet-Lite, se compatibile.
- Backbone compatti per Faster R-CNN.
- DETR con backbone ridotto.
- Architetture detection specificamente progettate per Edge AI.

Non limitare la compressione alla sola riduzione della precisione numerica.

Valutare anche la riduzione della complessità architetturale.

## 12.6 Combinazione delle tecniche

Prevedere esperimenti combinati:

- FP16 + Pruning.
- INT8 PTQ + Pruning.
- INT8 QAT + Pruning.
- Knowledge Distillation + INT8.
- Knowledge Distillation + FP16.
- Knowledge Distillation + INT8 + Pruning.

Non applicare automaticamente tutte le tecniche a tutte le architetture.

Verificare la compatibilità tecnica e misurare il risultato effettivo.

Ogni combinazione dovrà essere un esperimento indipendente.

# 13. EDGE AI MODEL BENCHMARKING

Realizzare un sottosistema specifico per valutare i modelli ottimizzati su differenti classi di hardware.

Prevedere profili di riferimento:

### Classe A – Satellite Onboard Computing

Sistemi con risorse computazionali, memoria, potenza elettrica e dissipazione termica limitate.

Valutare:

- CPU embedded.
- ARM.
- Acceleratori AI.
- Memoria disponibile.
- Latenza.
- Throughput.
- Consumo energetico.

### Classe B – Embedded AI

Prevedere profili per:

- NVIDIA Jetson.
- ARM CPU.
- Sistemi con NPU.
- Sistemi con acceleratori dedicati.

### Classe C – Edge AI Industrial

Prevedere:

- CPU x86.
- GPU NVIDIA.
- Sistemi embedded GPU.
- Acceleratori AI esterni.

Il benchmark dovrà essere configurabile in funzione dell'hardware realmente disponibile.

Non inventare risultati relativi a hardware non disponibile.

Se il dispositivo fisico non è presente, distinguere chiaramente:

- Risultati misurati.
- Risultati simulati.
- Stime teoriche.
- Risultati ottenuti mediante emulatori o runtime alternativi.

## 13.1 Runtime da valutare

Prevedere supporto per:

- PyTorch.
- ONNX Runtime.
- TensorRT.
- OpenVINO, quando applicabile.
- TFLite / LiteRT, quando applicabile.
- Runtime ARM/NPU compatibili con i modelli esportati.

La scelta del runtime dovrà essere subordinata alla compatibilità effettiva dell'architettura e degli operatori.

Non presumere che un modello ONNX possa essere convertito automaticamente in tutti i runtime.

## 13.2 Metriche Edge AI

Misurare per ogni modello:

- Model Size in MB.
- Number of Parameters.
- FLOPs.
- Peak RAM Usage.
- Peak VRAM Usage.
- Inference Latency.
- Average Latency.
- P95 Latency.
- P99 Latency.
- Throughput.
- CPU Usage.
- GPU Usage.
- Energy Consumption.
- Power Consumption.

Quando possibile, effettuare misurazioni energetiche reali tramite strumenti e sensori hardware.

Distinguere consumo energetico per inferenza, potenza media e potenza di picco.

Calcolare anche:

- Joules per Image.
- Images per Joule.
- Energy per Detection, quando appropriato.

## 13.3 Hardware-aware optimization

Implementare un sistema di configurazione per il dispositivo target.

Parametri:

- Hardware type.
- CPU architecture.
- GPU architecture.
- Accelerator type.
- Available RAM.
- Available VRAM.
- Available storage.
- Power budget.
- Runtime.
- Supported numerical precision.
- Maximum acceptable latency.

Il sistema dovrà utilizzare questi parametri per selezionare e confrontare le varianti del modello.

Prevedere la possibilità di definire profili hardware personalizzati.

# 14. MODEL DEGRADATION ANALYSIS

Questa sezione è essenziale.

Per ogni modello quantizzato o compattato, implementare una valutazione del degrado rispetto al corrispondente Baseline Model.

La valutazione dovrà utilizzare esattamente lo stesso test set e le stesse annotazioni.

## 14.1 Confronto delle metriche

Confrontare:

- Precision.
- Recall.
- F1-score.
- AP.
- AP50.
- AP75.
- mAP50-95.
- Metriche ufficiali xView3.
- Localization Error.
- False Positive Rate.
- False Negative Rate.

Calcolare per ogni metrica:

Absolute Difference:

Delta = Optimized Metric - Baseline Metric

Relative Degradation:

Degradation (%) = ((Baseline Metric - Optimized Metric) / Baseline Metric) × 100

Per le metriche nelle quali un valore inferiore è migliore, come la latenza, calcolare separatamente il miglioramento relativo.

Non applicare la formula di degrado indistintamente a metriche con significato opposto.

## 14.2 Detection Consistency

Oltre alle metriche aggregate, analizzare la differenza tra le singole detection del modello originale e quelle del modello ottimizzato.

Prevedere:

- Detection agreement.
- Detection mismatch.
- Newly detected objects.
- Lost detections.
- Confidence score variation.
- Bounding box displacement.
- Matching tra predizioni originali e ottimizzate.

Identificare quali target vengono persi dopo la quantizzazione o il pruning.

Valutare se il degrado interessa prevalentemente:

- Piccole imbarcazioni.
- Target con basso contrasto.
- Target offshore.
- Target in prossimità della costa.
- Target con confidence score iniziale ridotto.

## 14.3 Statistical Degradation Analysis

Calcolare:

- Media del degrado.
- Mediana.
- Deviazione standard.
- Percentili.
- Intervalli di confidenza al 95%.
- Bootstrap.
- Test statistici appropriati.
- Effect size.

Analizzare la significatività delle differenze tra modello originale e modello ottimizzato.

Considerare la dipendenza delle osservazioni appartenenti alla stessa scena.

## 14.4 Accuracy Retention

Definire un indicatore di Accuracy Retention.

Accuracy Retention (%) = (Optimized Metric / Baseline Metric) × 100

Applicarlo alle metriche per le quali un valore maggiore rappresenta una prestazione migliore.

Per ciascuna variante generare un riepilogo:

- Baseline Metric.
- Optimized Metric.
- Absolute Degradation.
- Relative Degradation.
- Accuracy Retention.
- Confidence Interval.
- Statistical Significance.

## 14.5 Compression Efficiency

Calcolare:

- Compression Ratio.
- Model Size Reduction.
- Parameter Reduction.
- Memory Reduction.
- Latency Improvement.
- Throughput Improvement.
- Energy Improvement, quando misurabile.

Confrontare la riduzione delle dimensioni del file con la riduzione effettiva della memoria durante l'inference.

Non considerare equivalenti le due misure.

# 15. MODEL OPTIMIZATION EXPERIMENT MATRIX

Implementare un sistema automatico per generare e gestire una matrice di esperimenti.

Esempio:

| ID  | Modello      | Ottimizzazione         |
| --- | ------------ | ---------------------- |
| M01 | YOLO         | FP32 Baseline          |
| M02 | YOLO         | FP16                   |
| M03 | YOLO         | INT8 PTQ               |
| M04 | YOLO         | INT8 QAT               |
| M05 | YOLO         | INT4                   |
| M06 | YOLO         | Structured Pruning     |
| M07 | YOLO         | Knowledge Distillation |
| M08 | Faster R-CNN | FP32 Baseline          |
| M09 | Faster R-CNN | FP16                   |
| M10 | Faster R-CNN | INT8                   |
| M11 | Faster R-CNN | Structured Pruning     |
| M12 | DETR         | FP32 Baseline          |
| M13 | DETR         | FP16                   |
| M14 | DETR         | INT8                   |
| M15 | DETR         | Knowledge Distillation |

La matrice dovrà essere generata in funzione delle effettive compatibilità delle tecniche con le architetture.

Prevedere anche configurazioni combinate.

Non generare esperimenti non supportati tecnicamente.

Ogni esperimento dovrà essere tracciato individualmente.

# 16. MULTI-OBJECTIVE OPTIMIZATION

Implementare un sistema per identificare i modelli più adatti all'impiego Edge AI.

Considerare simultaneamente:

- Detection Performance.
- Model Size.
- RAM Consumption.
- Inference Latency.
- Energy Consumption.
- Throughput.

Implementare:

- Pareto Front.
- Multi-objective Optimization.
- Ranking configurabile.
- Trade-off Analysis.

Utilizzare Optuna Multi-Objective quando appropriato.

Prevedere almeno tre profili operativi:

### Maximum Detection

Privilegiare Recall e riduzione dei falsi negativi.

### Balanced Detection

Bilanciare Precision, Recall, accuratezza e risorse hardware.

### Maximum Edge Efficiency

Privilegiare:

- Dimensioni del modello.
- Latenza.
- Memoria.
- Consumo energetico.

Non identificare un unico vincitore assoluto.

Il sistema dovrà permettere di selezionare il modello migliore in funzione dei vincoli operativi.

# 17. MODEL EXPORT E EDGE DEPLOYMENT

Realizzare un sistema di esportazione dei modelli.

Prevedere, quando supportato:

- PyTorch.
- ONNX.
- TensorRT Engine.
- TFLite / LiteRT.
- Altri formati specifici dei runtime target.

Ogni modello esportato dovrà essere accompagnato da:

- File di configurazione.
- Metadati.
- Architettura.
- Versione.
- Precisione numerica.
- Tecnica di compressione.
- Dimensione del file.
- Hash del modello.
- Metriche di validazione.
- Risultati del benchmark.
- Compatibilità hardware verificata.

Prevedere una procedura di verifica dell'equivalenza tra il modello PyTorch e quello esportato.

Verificare:

- Output numerici.
- Detection.
- Confidence score.
- Bounding box.
- Differenze di precisione.
- Eventuali operatori non supportati.

## 17.1 Edge Inference Runtime

Realizzare un modulo di inference indipendente dal training.

Il modulo dovrà poter utilizzare differenti backend attraverso un'interfaccia comune.

Prevedere:

- PyTorch Runtime.
- ONNX Runtime.
- TensorRT Runtime.
- LiteRT Runtime, quando compatibile.

Il sistema dovrà poter caricare il modello appropriato in funzione del dispositivo target.

## 17.2 Satellite Onboard AI

Predisporre una configurazione specifica per sistemi satellitari.

Considerare:

- Risorse computazionali limitate.
- Memoria limitata.
- Power budget.
- Vincoli termici.
- Latenza.
- Capacità di elaborazione autonoma.
- Limitazioni di comunicazione con la ground station.

Prevedere la possibilità di effettuare la detection direttamente a bordo, producendo risultati compatti da trasmettere a terra anziché trasferire necessariamente tutte le immagini originali.

Non dichiarare un modello flight-ready soltanto perché è stato esportato in INT8 o ONNX.

Documentare i requisiti aggiuntivi per un'eventuale qualificazione spaziale, inclusi fault tolerance, radiation tolerance e validazione dell'hardware.

# 18. VISUALIZZAZIONE DEI RISULTATI

Implementare un modulo di visualizzazione.

Prevedere grafici per:

- Training Loss.
- Validation Loss.
- Precision-Recall.
- F1.
- Confusion Matrix.
- AP Comparison.
- Training Time.
- Inference Time.
- Memory Consumption.
- Model Size.
- Compression Ratio.
- Accuracy Retention.
- Performance Degradation.
- Detection Agreement.
- Hyperparameter Importance.
- Optimization History.
- Confidence Intervals.
- Pareto Front.
- Energy Efficiency.
- Performance per Target Size.

Prevedere confronti tra:

- Modello originale.
- Modello FP16.
- Modello INT8 PTQ.
- Modello INT8 QAT.
- Modello INT4, se supportato.
- Modello Pruned.
- Modello Distilled.
- Modello combinato.

Esportare i grafici in PNG e SVG.

# 19. ANALISI GEO-SPAZIALE

Implementare un modulo basato su:

- Rasterio.
- GeoPandas.
- Shapely.
- PyProj.

Prevedere:

- Conversione coordinate pixel/geografiche.
- Bounding box.
- Coordinate centrali.
- Confidence score.
- Eventuale lunghezza stimata.
- Identificativo immagine.
- Data di acquisizione.
- Esportazione GeoJSON.

Preservare i metadati originali.

# 20. STRUTTURA DEL PROGETTO

Utilizzare una struttura modulare simile alla seguente:

sentinel-vessel-ai/\
│\
├── configs/\
│   ├── dataset/\
│   ├── models/\
│   ├── training/\
│   ├── evaluation/\
│   ├── optimization/\
│   └── hardware/\
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
│   ├── models/\
│   │   ├── yolo/\
│   │   ├── faster_rcnn/\
│   │   └── detr/\
│   │\
│   ├── training/\
│   ├── evaluation/\
│   ├── statistics/\
│   ├── optimization/\
│   │   ├── quantization/\
│   │   ├── pruning/\
│   │   ├── distillation/\
│   │   ├── compression/\
│   │   └── export/\
│   │\
│   ├── edge/\
│   │   ├── hardware_profiles/\
│   │   ├── runtimes/\
│   │   ├── benchmarking/\
│   │   ├── energy_monitoring/\
│   │   └── deployment/\
│   │\
│   ├── geospatial/\
│   ├── inference/\
│   └── utils/\
│\
├── experiments/\
├── models/\
│   ├── baseline/\
│   ├── quantized/\
│   ├── pruned/\
│   ├── distilled/\
│   └── edge_optimized/\
│\
├── reports/\
│   ├── statistics/\
│   ├── benchmarks/\
│   ├── degradation/\
│   ├── edge_ai/\
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

La struttura è indicativa e può essere migliorata.

Non inserire i dataset originali nel repository Git.

# 21. CONFIGURAZIONE E TRACKING

Utilizzare file YAML.

Prevedere configurazioni distinte per:

- Dataset.
- Preprocessing.
- Modelli.
- Training.
- Evaluation.
- Quantization.
- Pruning.
- Distillation.
- Hardware.
- Edge Runtime.

Valutare MLflow per il tracking.

Ogni modello dovrà avere un identificativo univoco e una relazione con il proprio Baseline Model.

Registrare:

- Experiment ID.
- Model ID.
- Parent Model ID.
- Model Version.
- Dataset Version.
- Code Version.
- Random Seed.
- Hardware.
- Runtime.
- Precision.
- Compression Technique.
- Quantization Configuration.
- Metriche.
- Risultati statistici.

# 22. TECNOLOGIE E LIBRERIE

Valutare:

- Python 3.11 o versione compatibile.
- PyTorch.
- torchvision.
- Ultralytics.
- Hugging Face Transformers.
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
- ESA SNAP.
- ONNX.
- ONNX Runtime.
- TensorRT.
- OpenVINO.
- LiteRT / TFLite.
- PyTorch Quantization.
- Hardware monitoring libraries compatibili con i dispositivi disponibili.

Verificare le versioni e la compatibilità effettiva delle dipendenze.

Non introdurre librerie non necessarie.

# 23. TESTING E QUALITÀ DEL SOFTWARE

Utilizzare pytest.

Prevedere test per:

- Dataset loader.
- Annotation parser.
- Preprocessing.
- Tiling.
- DataLoader.
- Forward pass.
- Training.
- Metriche.
- Metriche ufficiali xView3.
- Statistical Analysis.
- Quantization.
- Pruning.
- Distillation.
- Model Export.
- ONNX Inference.
- Runtime Inference.
- GeoJSON.
- Model Degradation Analysis.

Verificare che le versioni ottimizzate producano output validi.

Prevedere test di regressione per individuare eventuali perdite significative di performance.

# 24. DOCUMENTAZIONE

Produrre:

- README.md.
- ARCHITECTURE.md.
- EXPERIMENT_PLAN.md.
- MODEL_OPTIMIZATION.md.
- EDGE_DEPLOYMENT.md.
- STATISTICAL_ANALYSIS.md.

Documentare:

- Installazione.
- Configurazione.
- Dataset.
- Preprocessing.
- Training.
- Evaluation.
- Statistical Analysis.
- Hyperparameter Optimization.
- Quantization.
- Pruning.
- Knowledge Distillation.
- Model Export.
- Edge Benchmarking.
- Model Degradation.
- Deployment.

# 25. REPORT FINALE AUTOMATICO

Il progetto dovrà generare report HTML e PDF.

Il report dovrà includere:

## A. Baseline Models

- Architettura.
- Parametri.
- Dimensione.
- Metriche.
- Tempi di training.
- Tempi di inference.
- Consumi hardware.

## B. Optimized Models

- Tecnica applicata.
- Precision numerica.
- Dimensione.
- Parametri.
- Compression Ratio.
- Memoria.
- Latenza.
- Throughput.
- Consumo energetico, se misurato.

## C. Degradation Analysis

- Metriche baseline.
- Metriche ottimizzate.
- Differenze assolute.
- Degrado percentuale.
- Accuracy Retention.
- Intervalli di confidenza.
- Significatività statistica.
- Detection Agreement.
- Target persi.

## D. Edge AI Suitability

- Compatibilità hardware.
- Runtime.
- Memoria richiesta.
- Latenza.
- Throughput.
- Power Consumption.
- Rispetto dei vincoli operativi.

## E. Final Ranking

Identificare i modelli migliori secondo differenti profili:

- Best Detection Performance.
- Best Compression Ratio.
- Best Accuracy/Size Trade-off.
- Best Accuracy/Latency Trade-off.
- Best Energy Efficiency.
- Best Overall Edge AI Candidate.

Non dichiarare un modello idoneo a un dispositivo senza averne verificato la compatibilità e le prestazioni.

# 26. MODALITÀ DI SVILUPPO

Procedere per fasi.

## Fase 1 – Analisi e progettazione

Analizzare il problema, verificare xView3, definire architettura, annotazioni, pipeline SAR e piano sperimentale.

Produrre:

- ARCHITECTURE.md.
- EXPERIMENT_PLAN.md.
- MODEL_OPTIMIZATION.md.
- EDGE_DEPLOYMENT.md.

Non iniziare il training prima di aver verificato la pipeline dati.

## Fase 2 – Dataset e preprocessing

Implementare loader, annotazioni, preprocessing, tiling e split.

## Fase 3 – Modelli principali

Implementare YOLO, Faster R-CNN e DETR.

## Fase 4 – Training e benchmark

Addestrare i modelli e confrontarli.

## Fase 5 – Analisi statistica

Implementare esperimenti ripetuti, intervalli di confidenza, test statistici e analisi degli errori.

## Fase 6 – Model Compression

Implementare quantizzazione, pruning, distillation e modelli compatti.

## Fase 7 – Model Degradation Analysis

Confrontare tutte le varianti con il relativo modello originale.

## Fase 8 – Edge AI Benchmarking

Valutare le versioni ottimizzate sui dispositivi disponibili.

## Fase 9 – Multi-objective Optimization

Individuare i migliori compromessi tra accuratezza, dimensione, latenza, memoria ed energia.

## Fase 10 – Export e Deployment

Esportare i modelli e implementare l'inference.

## Fase 11 – Reporting

Generare report comparativi e documentazione finale.

# 27. REGOLE FONDAMENTALI

1. Non inventare dati, annotazioni o risultati.
2. Non sovrascrivere i modelli originali.
3. Conservare sempre il Baseline Model.
4. Ogni variante ottimizzata deve essere tracciabile.
5. Non dichiarare completate operazioni non eseguite.
6. Non confondere risultati misurati con stime.
7. Non utilizzare il test set per ottimizzazione.
8. Non confondere Sentinel-1 con Sentinel-2.
9. Non alterare le annotazioni xView3.
10. Non introdurre data leakage.
11. Non confondere la dimensione del file con la memoria effettivamente utilizzata.
12. Non confondere la sparsità con un reale incremento di velocità.
13. Non presumere che INT8 sia sempre più veloce di FP16.
14. Non presumere che tutti gli acceleratori supportino le stesse operazioni.
15. Non dichiarare un modello flight-ready senza le necessarie verifiche.
16. Misurare il degrado delle metriche per ogni tecnica di compressione.
17. Mantenere la riproducibilità degli esperimenti.
18. Documentare tutte le limitazioni.

# 28. CRITERI DI ACCETTAZIONE

Il progetto sarà considerato completo quando sarà possibile:

- Caricare xView3.
- Preparare immagini Sentinel-1.
- Addestrare YOLO.
- Addestrare Faster R-CNN.
- Addestrare DETR.
- Valutare i tre modelli.
- Eseguire analisi statistiche.
- Ottimizzare gli iperparametri.
- Generare Baseline Models.
- Generare modelli FP16.
- Generare modelli INT8 PTQ.
- Generare modelli INT8 QAT, quando supportati.
- Valutare INT4, quando supportato.
- Eseguire pruning.
- Eseguire knowledge distillation.
- Generare modelli compatti.
- Esportare modelli per differenti runtime.
- Valutare la compatibilità hardware.
- Misurare le prestazioni Edge AI.
- Calcolare il degrado rispetto ai Baseline Models.
- Calcolare Accuracy Retention.
- Generare Pareto Front.
- Identificare i migliori modelli per differenti profili operativi.
- Produrre report comparativi HTML/PDF.

Il risultato dovrà essere una piattaforma professionale di sperimentazione e ottimizzazione AI, con particolare attenzione all'esecuzione di modelli di Ship Detection su dispositivi Edge AI e piattaforme satellitari.

La priorità assoluta è la correttezza metodologica degli esperimenti, la misurazione reale dei risultati e la capacità di confrontare quantitativamente accuratezza e costi computazionali.

# **2. Aspetti tecnici che ho aggiunto e che ritengo fondamentali**

La differenza sostanziale rispetto alla precedente versione è la gestione delle varianti del modello come veri e propri artefatti sperimentali.

Ad esempio, per ogni architettura avrai una struttura concettuale di questo tipo:

**BASELINE MODEL**

# **FP32**

Modello originale addestrato e validato

**Quantization**

- FP16
- INT8 PTQ
- INT8 QAT
- INT4

**Compression**

- Structured pruning
- Channel pruning
- Knowledge distillation
- Lightweight models

**EDGE OPTIMIZATION**

ONNX · TensorRT · LiteRT · Runtime specifici

**DEGRADATION & PERFORMANCE ANALYSIS**

Accuracy retention · Latency · RAM · Energy · Pareto front

# **3. Esempio di valutazione dei modelli**

Per chiarire il risultato che dovrà produrre il software, immagina di avere un modello YOLO FP32 e tre varianti ottimizzate.

I valori seguenti sono esclusivamente illustrativi, non risultati reali di xView3.

| Modello             | Dimensione | mAP   | Retention | Latenza |
| ------------------- | ---------- | ----- | --------- | ------- |
| YOLO FP32           | 100 MB     | 0,85  | 100%      | 45 ms   |
| YOLO FP16           | 50 MB      | 0,848 | 99,8%     | 27 ms   |
| YOLO INT8           | 26 MB      | 0,831 | 97,8%     | 18 ms   |
| YOLO INT8 + Pruning | 16 MB      | 0,805 | 94,7%     | 14 ms   |

Il sistema dovrà poter produrre automaticamente un grafico come questo:

Esempio di Accuracy Retention

Valori simulati per illustrare il confronto tra modelli.

90%92.5%95%97.5%100%FP32FP16INT8INT8 + Pruning

Ma non si fermerà qui: dovrà verificare anche se il miglioramento della latenza sia effettivamente ottenuto sull'hardware di destinazione.

# **4. Hardware target che prenderei in considerazione**

Per il tuo obiettivo satellitare ed Edge AI, predisporrei questi profili:

[NVIDIA Jetson Orin Nano Super Kit mit 67 TOPS Leistung als Dev-Kit ab 300 € vorgestellt  - PC Masters](https://images.openai.com/static-rsc-4/PjAPzfyFbqNyUa9H9H3ueX_7Nt1lY3kQX9RQqNT43MrS8Y5JDQTVjbnNcTrm2xFo3OJuD3CpWK174JPuFywnL6o4Uabtnhkzld952Z22GV5wC55O5sMY1pwMEDi6biqJZj387p2bAndLHVErM7MYxcJdPpzWJvE84z0TpwrK8LE?purpose=inline)

**NVIDIA Jetson Orin Nano**

Piattaforma di riferimento per sviluppare e misurare inferenza GPU embedded, TensorRT e modelli quantizzati.

Profilo Edge AI GPU

[Koprocesor Google Coral Edge TPU ARM Cortex M0, 0193575021935 • Cena • Opinie - Allegro](https://images.openai.com/static-rsc-4/gv0xQrtm9_doT-qWcPJS_D6Lhh9I2cEb8jLHsAHuIObeTkuvBmQO9uaAHdPk5DRF0QNORtujCzPB276s-PNXZXsHl3PbRNvYBfSyzN2KeDF6ILJn86nvyQEL9FNzL4RAzkRas6nd1pb5jZg6BwCdH8PHgoa7PHnWZlzfMoVX3N4?purpose=inline)

**Google Coral Edge TPU**

Interessante per studiare la conversione di modelli in INT8 e i vincoli degli operatori supportati dall'acceleratore.

Profilo AI accelerator

[STM32MP157F-DK2 STMicroelectronics - Geliştirme Kitleri - e-komponent.com](https://images.openai.com/static-rsc-4/vWuQ7qd9hBnNFgex2-97_zZf6ReOUnI3hW77xseRyLy27iMmSiFi7Mzk3rRtDUxiPxaySBTEG3yxyTY6yB1yKR6qx1LqlTbZf1nIbaOY1oULHl1jgfffl6EKov5qJudZ97G6s6PruEvU587tu0NAk-Z9R4szchNIWm-AztuIlII?purpose=inline)

**ARM Embedded**

Utile per valutare modelli compatti, inferenza CPU e ottimizzazione della memoria.

Profilo CPU embedded

[NOVI | VORAGO Technologies](https://images.openai.com/static-rsc-4/OdGngfw8ysDdyoYrW0TkSWtmILKyZ3fdAkl5tkGRLoZfj_mYmoh1CAdLDCIeeW9XMAORGuFc901dtpQ7dnlKCKz6lPzbVqQ6M4KwedY6qtl4DeRIM-H9TSSqbRKmh4OyV94UaeHwfM7BWd1gSTxO1ITikjqh5BKsPQULM7jV-Ew?purpose=inline)

**Satellite Onboard Computer**

Profilo astratto per simulare vincoli di memoria, potenza, latenza e capacità computazionale di un sistema satellitare.

Profilo Space AI

Un'importante distinzione: Jetson, Coral e ARM non sono automaticamente equivalenti a un computer qualificato per applicazioni spaziali. Servono ulteriori verifiche hardware, software e ambientali.

# **5. Un'ulteriore funzionalità che consiglio di introdurre**

Aggiungerei una funzione denominata **Model Selection under Hardware Constraints**.

In pratica, il sistema dovrebbe consentire di specificare un budget massimo e individuare automaticamente i modelli compatibili.

Esempio di configurazione:

```
edge_target:
  name: satellite_onboard_profile

  constraints:
    max_model_size_mb: 25
    max_ram_mb: 512
    max_latency_ms: 100
    max_power_w: 5

  optimization:
    allowed_precisions:
      - FP16
      - INT8

    allow_pruning: true
    allow_distillation: true

  selection:
    primary_metric: recall
    secondary_metric: latency
    min_accuracy_retention: 95
```

Questi valori sono parametri di esempio, non specifiche universali di un satellite.

Il sistema dovrebbe:

1. Escludere i modelli che non rispettano i vincoli.
2. Verificare la compatibilità con il runtime.
3. Confrontare i modelli rimanenti.
4. Calcolare il degrado rispetto al baseline.
5. Produrre una classifica dei candidati.
6. Generare un report con motivazioni e limitazioni.

**Questo passaggio è particolarmente importante:** il modello migliore per il tuo progetto non sarà necessariamente quello con il mAP più elevato, ma quello che mantiene una capacità di detection sufficientemente alta rispettando i vincoli computazionali ed energetici della piattaforma di destinazione.