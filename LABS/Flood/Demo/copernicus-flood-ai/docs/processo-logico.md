# Processo logico implementato

Il [diagramma Mermaid](processo-logico.mmd) descrive la pipeline attuale per l'attivazione EMSR865, dall'acquisizione dei dati all'inferenza.

1. **Ingestion e verifica:** si acquisiscono metadati e prodotti CEMS e si cercano scene Sentinel-1 VV/VH su CDSE STAC. Le associazioni sono verificate tramite identificativi sorgente, tempo e footprint; solo gli asset verificati possono essere scaricati esplicitamente tramite S3 o HTTPS con OIDC. Dimensioni e SHA-256 sono registrati localmente.
2. **Dataset:** il builder controlla che siano presenti tutti gli asset verificati e che coincidano con il registro. Riproietta i raster tramite CRS o GCP, mosaica e ritaglia sull'AOI; costruisce la maschera CEMS e genera patch con split spaziali. Se gli input sono incoerenti il preprocessing si ferma; se uno split non contiene entrambe le classi, training e valutazione restano bloccati.
3. **Training e valutazione:** le patch usano `log1p` e statistiche di normalizzazione calcolate sul solo train. Una U-Net viene addestrata con validazione ed early stopping; il checkpoint migliore viene valutato sullo split spaziale test.
4. **Inferenza e demo:** il checkpoint e lo stack VV/VH alimentano l'inferenza con AOI e soglia. Il processo produce maschera e probabilità GeoTIFF, poligoni GeoJSON, PNG e manifesto; il GeoPackage è generato quando sono presenti poligoni. La demo Streamlit applica questo flusso alla sola patch selezionata e permette di visualizzare e scaricare i risultati.

## Limiti

- Il background negativo nell'AOI è un'assunzione didattica, non una classe verificata da CEMS.
- La validazione è spaziale all'interno di EMSR865 e non dimostra generalizzazione ad altri eventi. Il checkpoint di smoke test è stato addestrato per una sola epoca e non è operativo.
- I raster usano ampiezza SAR grezza; non è implementata una calibrazione radiometrica.
- Esportazione ONNX, quantizzazione e deployment su dispositivi di bordo sono proposte future, **non** parti del processo implementato.


