# Progetto separato: quantizzazione meteo ONNX

Questo è un **secondo progetto Python indipendente**. Allena MLP (`MLPRegressor`) esportabili in ONNX e produce versioni INT8 con la quantizzazione dinamica di ONNX Runtime. Non importa codice dal progetto `Meteo` e non modifica alcun suo file: legge soltanto il CSV indicato dall'utente e scrive i propri artefatti nella cartella di questo progetto.

Il progetto è un esperimento di esportazione e quantizzazione, non una sostituzione automatica del modello esistente né una garanzia di previsioni meteo migliori. Le metriche del modello float, del modello INT8 e del baseline di persistenza vengono confrontate su un test cronologico.

## Requisiti e installazione

Richiede Python 3.10 o successivo. Da `Meteo-Quantization`:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Il progetto allena solo modelli supportati dal convertitore `skl2onnx`. L'architettura è un `StandardScaler` seguito da `MLPRegressor`, convertita in operatori ONNX standard usati dal quantizzatore dinamico. La quantizzazione viene eseguita da ONNX Runtime con pesi QInt8.

## Esecuzione con lo storico già acquisito

Il progetto usa come input il CSV delle feature preparato in precedenza, ma non lo riscrive:

```powershell
python -m quant_weather.cli `
  --data-csv ..\Meteo\data\processed\training_features.csv `
  --output-dir artifacts `
  --hidden-layers 64 32 `
  --max-iter 1200
```

Il percorso `--data-csv` può puntare a qualsiasi CSV con colonne numeriche di feature, `date`, i tre target `target_temperature_2m_max`, `target_precipitation_sum`, `target_wind_speed_10m_max` e le rispettive colonne baseline senza prefisso `target_`.

La pipeline divide i dati in ordine cronologico (70% training, 15% validation, 15% test), addestra un MLP per target, esporta una versione FP32 e crea il corrispondente file `.int8.onnx`. La quantizzazione resta nel report anche quando il controllo di fedeltà non viene superato, così il candidato è ispezionabile; `quantization_fidelity_accepted` indica solo se il peggioramento RMSE dell'INT8 rispetto al suo MLP FP32 rientra nella tolleranza predefinita del 5%. **Non** indica che il modello meteo sia migliore del baseline: per questo controllare separatamente le metriche test del report.

## Artefatti prodotti

```text
artifacts/
  onnx/                       Modelli ONNX FP32
  quantized/                  Modelli candidati ONNX INT8
  python_models.joblib        Modelli per inferenza Python
  quantization_report.json    Metriche, dimensioni e tempi misurati
```

Il report confronta MAE, RMSE e R² per FP32 e INT8 sul test e include la baseline di persistenza. Include inoltre dimensione dei file, tempi indicativi di inferenza batch CPU, differenza tra predizioni, esito di equivalenza ONNX/Python e variazione relativa RMSE dopo quantizzazione. I tempi dipendono dall'hardware e non rappresentano misure su un dispositivo edge.

## Verifica

```powershell
python -m pytest
```

I test generano dati sintetici in memoria, verificano il controllo dello schema e provano l'intero percorso MLP → ONNX → INT8 con ONNX Runtime CPU. Non contattano API esterne.

## Limiti

- La conversione non rende automaticamente il modello accurato: il report include il confronto con la persistenza e va usato per decidere se il modello è utile.
- La quantizzazione dinamica opera sui pesi e sugli operatori supportati; non garantisce file più piccoli o inferenza più veloce, soprattutto per reti piccole.
- L'artefatto INT8 viene conservato anche se `quality_accepted` è falso, ma non va considerato approvato per la distribuzione.
- La validazione usa dati storici Open-Meteo già elaborati e non dati di una stazione indipendente.
