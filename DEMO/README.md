# Jaskra — wersja demo (bez backendu)

Samodzielna kopia aplikacji mobilnej (`mobile/`) do prezentacji, gdy backend
na Azure jest niedostępny. `GlaucomaService.swift` w tym folderze nie wykonuje
żadnych połączeń sieciowych — logowanie/rejestracja zawsze się udają, a
analiza zdjęcia symuluje te same kroki co prawdziwy backend i zwraca
wiarygodny wynik (losowy CDR/pewność + rysowana nakładka na zdjęciu
udająca wykrycie tarczy/zagłębienia). Pacjenci i historia były już w pełni
lokalne, więc działają identycznie jak w wersji produkcyjnej.

## Jak odpalić w Swift Playgrounds

1. Skopiuj folder `DEMO` na Maca/iPada (np. przez AirDrop, iCloud Drive).
2. Otwórz go w aplikacji **Swift Playgrounds** — potraktuje go jak projekt
   aplikacji, tak samo jak folder `mobile/` normalnie.
3. Uruchom. Ekran logowania: dowolna nazwa użytkownika i hasło (min. 4 znaki)
   zadziała, albo użyj „Analizuj jako gość”.

## Gotowi pacjenci (prawdziwe wyniki naszego modelu)

Przy pierwszym uruchomieniu (pusta baza) apka sama zasiewa 29 pacjentów
zbudowanych z 30 zdjęć dna oka faktycznie przepuszczonych przez nasz
pipeline (YOLO ROI + U-Net++) — `hasGlaucoma`/CDR to prawdziwy wynik AI
(kolumna „AI” z `raport_wynikow_diagnozy.txt`), nie losowe dane. Obraz to
przycięte ROI z klasyczną nakładką (zielony dysk / czerwona miska) — czytelniejsze
niż całe zdjęcie, bo to różni pacjenci i tarcza jest w innym miejscu na
każdym zdjęciu. Jeden pacjent (`Pacjent REFUGE1-train-40`) ma dwie wizyty
złożone z dwóch różnych zdjęć z datasetu (sprzed 4 miesięcy i dziś), żeby
„Porównaj analizy” miało na czym pokazać prawdziwą różnicę CDR (0.25 → 0.69).
To nie jest faktyczna historia jednego pacjenta — czysto demonstracyjne.

„Pewność klasyfikacji” nie pochodzi z pipeline'u (próg CDR nie zwraca
prawdopodobieństwa) — to przybliżenie na podstawie odległości aCDR od progu
0.6, tylko na potrzeby wypełnienia UI.

Kod generujący te dane: `DemoPatientSeed.swift` (zaszyte base64 obrazów) +
`PatientStore.seedDemoPatients()`. Żeby zacząć od zera, usuń dane appki
(Ustawienia iOS → usuń i zainstaluj ponownie, albo w Playgrounds wyczyść
storage projektu) — seeding uruchamia się tylko gdy baza pacjentów jest pusta.

## Porównanie analiz (tylko w tej wersji)

Na karcie pacjenta z co najmniej 2 analizami pojawia się przycisk
„Porównaj analizy” — wybierasz analizę archiwalną i aktualną, przesuwasz
suwak, żeby nałożyć jedno zdjęcie na drugie, i widzisz różnicę CDR oraz
zmianę diagnozy. Segmentowany przełącznik u góry pozwala porównywać albo
same zdjęcia (przycięte ROI + nakładka), albo same maski tarczy/zagłębienia
(przycięta maska z pipeline'u).

## Co jest inne niż w `mobile/`

Zmieniony jest `Services/GlaucomaService.swift` (brak sieci, dodatkowo
generuje obraz maski dla nowych/na żywo wykonanych analiz),
`Models/NetworkModels.swift` (usunięte nieużywane typy odpowiedzi HTTP),
`Models/Patient.swift` i `Services/PatientStore.swift` (zapis maski obok
zdjęcia + seeding gotowych pacjentów) oraz `Patients/PatientDetailView.swift`
(przycisk porównania). Doszły dwa nowe pliki: `Analysis/CompareAnalysesView.swift`
i `Services/DemoPatientSeed.swift`. Reszta plików to dokładna kopia z `mobile/`.
Gdy backend wróci, prezentacje róbcie z `mobile/` — ten folder jest tylko na
wypadek awarii.
