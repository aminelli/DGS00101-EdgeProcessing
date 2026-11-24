Vessel (detection imbarcazioni da dati Sentinel-1 e/o Sentinel-2)
        1. Premessa: si vuole svolgere un monitoraggio in una particolare AOI in mare aperto intorno al bosforo per geolocalizzare le navi presenti
        2. Dati in input:
            1. immagine SAR acquisita con la missione ESA Copernicus Sentinel-1. Livello di produzione 1B (GRD).
            2. immagine elettro-ottica multispettrale acquisita con la missione ESA Copernicus Sentinel-2. Livello di produzione L2A (TOA).
        3. Task edge:
            1. Import del dato, geocoding su ellissoide nel caso SAR.
            2. Detection delle imbarcazioni
        4. Descrizione del prodotto: lista delle navi rilevate con coordinate e dimensione (se possibile in formato vettoriale)