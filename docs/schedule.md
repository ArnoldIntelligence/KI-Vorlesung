
# Zeitplan für KI-Vorlesung

| Tag 1    | Tag 2    | Tag 3    |
|:---------|:---------|:---------|
| 01 Einleitung | 03 Überwachtes Lernen | 05 CNNs                     |
| 02 Perzeptron | 04 Zahlen Erkennen    | 06 Nicht überwachtes Lernen |
|               |                       | 07 Reinforcement Learning      |

## Tag 1: Vorschlag für 4 Einheiten à 90 Minuten 

Basierend auf den Inhalten der beiden Notebooks 01-einleitung.py und 02-perzeptron.py würde ich die Stoffmenge in diese vier Blöcke aufteilen:

### Einheit 1 – Einleitung und Grundkonzepte des Machine Learning
Ziel: Verständnis für Begriffe, Klassifikation vs. Regression und die Grundidee des Perzeptrons.

- 10 min: Einstieg, Lernziele, Motivation
- 20 min: Begriffsdefinitionen aus 01-einleitung.py
  - Features, Labels, Train/Test, Forward-Berechnung, Loss, Kostenfunktion
- 25 min: Machine Learning im Überblick
  - Klassifikation, Regression
  - Unterschied zwischen Lernaufgaben und Inferenz
- 25 min: Einführung in das Perzeptron
  - Aufbau: Eingang, Gewichte, Aktivierungsfunktion, Ausgabe
  - Mathematische Formulierung
- 10 min: Kurztest / Verständnisfragen
- 15 min: Beispiel zum Nachrechnen: einfache Gewichtete Summe

Ergebnis: Die Studierenden kennen die zentralen Begriffe und können eine einfache Perzeptron-Formel nachvollziehen.

---

### Einheit 2 – Perzeptron als Klassifikator und Lernen
Ziel: Grundalgorithmus verstehen und selbst implementieren.

- 15 min: Wiederholung und kurze Einordnung
- 20 min: Forward-Berechnung mit einfachen Beispielen
  - Roboter mit zwei Sensoren
  - Basilikum mit drei Messstellen
- 30 min: Perzeptron-Lernregel
  - Heaviside-Funktion
  - Fehlerberechnung
  - Gewichtsanpassung
- 20 min: Programmierung des Lernalgorithmus
  - fit-Funktion
  - Training mit Beispiel 3: Kraftwerksschicht
- 5 min: Diskussion der Grenzen
- 20 min: Übung in Kleingruppen
  - Gewichte variieren, Startwerte ändern, Fehlerverlauf beobachten

Ergebnis: Die Studierenden können den Lernprozess des Perzeptrons erklären und mit einem eigenen kleinen Codebeispiel nachvollziehen.

---

### Einheit 3 – Datenanalyse mit echten Datensätzen
Ziel: lineare Trennbarkeit, Klassifikation mit realen Daten, Grenzen des einfachen Perzeptrons.

- 10 min: Rückblick auf die Lernregel
- 25 min: Iris-Datensatz
  - Visualisierung
  - lineare Trennung von setosa vs. Rest
  - Perzeptron-Training
- 25 min: Weiterführende Aufgaben
  - Iris-Arten 2 und 3
  - Diskussion, warum manche Trennungen nicht gelingen
- 20 min: Pinguin-Datensatz
  - Merkmale und Plot
  - Welche Features lassen sich linear trennen?
- 10 min: Analyse der Ergebnisse
- 20 min: Übung
  - eigene kleine Klassifikationsaufgabe mit zwei Features

Ergebnis: Die Studierenden verstehen, dass einfache Perzeptronen nur linear trennbare Muster lernen können.

---

### Einheit 4 – Adaline, Gradient Descent und Abschluss
Ziel: Übergang von einfachem Perzeptron zu modernen neuronalen Netzen und kritische Reflexion.

- 15 min: Wiederholung von Grenzen des klassischen Perzeptrons
- 25 min: Adaline-Perzeptron
  - Lernrate
  - Fehler als quadratischer Fehler
  - Gradient Descent
- 25 min: Vergleich
  - klassisches Perzeptron vs. Adaline
  - Schrittgröße, Fehlerverlauf, Stabilität
- 15 min: Diskussion der Motivation
  - Warum braucht man mehr als eine lineare Trennfunktion?
  - Übergang zu Mehrschicht-Netzen
- 10 min: Mini-Rekapitulation der wichtigsten Formeln
- 20 min: Abschlussübung / Quiz / Reflexionsfrage

Ergebnis: Die Studierenden können die grundlegenden Unterschiede zwischen Perzeptron und Adaline erklären und den Übergang zu modernen neuronalen Netzen motivieren.

---

## Empfohlene didaktische Struktur pro Einheit
Für jede Einheit empfehle ich diese Rhythmusfolge:

- 10–15 min: Einstieg / Wiederholung
- 30–40 min: Theorie und Erklärung
- 25–35 min: Code/Beispiel/Übung
- 10–15 min: Besprechung / Diskussion
- 5–10 min: Abschlussfrage oder kurze Zusammenfassung

---

## Warum diese Reihenfolge sinnvoll ist?
Diese Abfolge baut logisch auf:

1. Erst Begriffe und Grundideen
2. Dann Perzeptron als Lernalgorithmus
3. Dann reale Daten und Grenzen
4. Dann moderne Erweiterung (Adaline) und Reflexion

So entsteht ein klarer Lernpfad von “Verstehen” über “Implementieren” zu “Kritisch Denken”.
