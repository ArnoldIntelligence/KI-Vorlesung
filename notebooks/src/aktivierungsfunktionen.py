# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv (3.10.12)
#     language: python
#     name: python3
# ---

# # Aktivierungsfunktionen im maschinellen Lernen
#
# ## Überblick
#
# Aktivierungsfunktionen bestimmen, wie die affine Voraktivierung $$z=W x+b$$ eines Neurons in dessen Ausgabe $$a=\phi(z)$$ überführt wird. Erst eine **nichtlineare** Aktivierung macht gestapelte Schichten ausdrucksstärker als eine einzige affine Abbildung; ohne sie kollabiert ein tiefes Feedforward-Netz mathematisch zu einer linearen Transformation.[^1]
#
# Eine universell beste Aktivierungsfunktion gibt es nicht. Für gewöhnliche MLPs und CNNs ist **ReLU** weiterhin ein belastbarer Ausgangspunkt; **GELU** sowie **SiLU/Swish** sind häufige Optionen in modernen tiefen Netzen und Transformern; **Sigmoid**, **Softmax** und die lineare Identität werden vor allem passend zur Bedeutung der **Ausgabe** gewählt. Bei wissenschaftlichem maschinellem Lernen, insbesondere PINNs, sind die Glattheit und höhere Ableitungen oft wichtiger als die reine Trainingsgeschwindigkeit.
#
# ## Zentrale Auswahlkriterien
#
# Bei der Wahl sind insbesondere folgende Eigenschaften relevant:
#
# - **Sättigung:** Bei Sigmoid und tanh geht $$\phi'(z)$$ für große $$|z|$$ gegen null. Über viele Schichten multipliziert die Kettenregel solche kleinen Faktoren, wodurch Gradienten verschwinden können.
# - **Wertebereich und Zentrierung:** Nullzentrierte oder negative Ausgaben können die Optimierung erleichtern. Sigmoid liefert nur positive Werte; tanh, Leaky ReLU, ELU, GELU und SiLU erlauben negative Ausgaben.
# - **Glattheit:** ReLU ist bei null nicht differenzierbar und besitzt fast überall verschwindende Ableitungen ab zweiter Ordnung. Das ist für normale Backpropagation meist unproblematisch, für PDE-Residuen mit höheren Ableitungen aber potenziell ungeeignet.[^2][^3]
# - **Rechenkosten:** Stückweise lineare Funktionen wie ReLU sind besonders günstig; exponentielle, trigonometrische oder auf der Normalverteilung beruhende Funktionen sind aufwendiger, werden auf moderner Hardware jedoch häufig durch optimierte Kerne oder Approximationen umgesetzt.
# - **Sparsity:** ReLU setzt alle negativen Voraktivierungen exakt auf null und erzeugt damit dünn besetzte Aktivierungen; Glorot, Bordes und Bengio beobachteten solche exakten Nullen als charakteristisches Merkmal tiefer Rectifier-Netze.[^4][^5]
# - **Ausgabesemantik:** Für versteckte Schichten geht es primär um Optimierung und Repräsentation; in der Ausgabeschicht muss der Wertebereich zum Modell und zur Verlustfunktion passen.
#
# ## Typische Funktionen
#
# ### Klassische Funktionen
#
# | Funktion | Definition und Bereich | Vorteile | Nachteile | Typische Verwendung |
# |---|---|---|---|---|
# | **Binäre Stufe / Heaviside** | $$\phi(z)=\mathbf{1}_{z\ge 0}$$, Ausgabe $$\{0,1\}$$ | Sehr einfach; klare logische Entscheidung; geeignet für klassische Schwellwertneuronen | Ableitung fast überall null und am Schwellwert undefiniert; daher nicht mit gewöhnlicher Gradientenabstiegs-Backpropagation trainierbar | Historisches McCulloch-Pitts-Neuron, Perzeptron-Ausgabe, binäre Logik; heute eher Nachbearbeitung als Hidden-Layer-Aktivierung[^6][^7] |
# | **Linear / Identität** | $$\phi(z)=z$$, Ausgabe $$\mathbb{R}$$ | Keine Sättigung; unverzerrter unbeschränkter Output | In Hidden Layers keine zusätzliche Nichtlinearität; mehrere lineare Schichten bleiben insgesamt linear | Regressionsausgabe für unbeschränkte Zielgrößen; letzte Schicht eines Approximators |
# | **Logistische Sigmoidfunktion** | $$\sigma(z)=1/(1+e^{-z})$$, Ausgabe $$(0,1)$$ | Glatt; als Wahrscheinlichkeit interpretierbar; einfache Ableitung $$\sigma'(z)=\sigma(z)(1-\sigma(z))$$ | Sättigt beidseitig; verschwindende Gradienten; nicht nullzentriert; Exponentialfunktion nötig | Binäre oder unabhängige Multi-Label-Ausgaben; Gates in LSTM/GRU/GLU, nicht mehr Standard für tiefe Hidden Layers[^8][^9] |
# | **tanh** | $$\tanh(z)=(e^z-e^{-z})/(e^z+e^{-z})$$, Ausgabe $$(-1,1)$$ | Glatt und nullzentriert; stärkere Gradienten um null als Sigmoid; alle Ableitungsordnungen vorhanden | Sättigt für große $$|z|$$; weiterhin anfällig für verschwindende Gradienten | Klassische MLPs/RNNs; Zustände mit natürlicher Begrenzung; PINNs und Funktionsapproximation mit Ableitungsbedarf[^8][^10] |
# | **Softmax** | $$\phi_i(\mathbf z)=e^{z_i}/\sum_j e^{z_j}$$ | Koppelt Klassenlogits zu einer normierten kategorialen Verteilung; Komponenten liegen in $$(0,1)$$ und summieren sich zu eins | Kann bei großen Logits numerisch überlaufen, sofern nicht stabilisiert; Sättigung kann Gradienten verkleinern; ungeeignet für voneinander unabhängige Labels | Ausgabeschicht bei exklusiver Mehrklassenklassifikation; meist zusammen mit Cross-Entropy bzw. direkt als log-softmax[^9][^11] |
#
#

# +
import numpy as np
import matplotlib.pyplot as plt

# Define range for input z
z = np.linspace(-5, 5, 100)

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def heaviside(z):
    return np.heaviside(z, 0.5)

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis=0, keepdims=True) 

# Compute activations
sig = sigmoid(z)
t = tanh(z)
h = heaviside(z)
s = softmax(z)

# Plot
plt.figure(figsize=(8,8/1.61))
plt.plot(z, sig, label='Sigmoid', color='blue')
plt.plot(z, t, label='Tanh', color='green')
plt.plot(z, h, label='Heaviside', color='red')
plt.plot(z, s, label='Softmax', color='orange')

plt.title('Classic Activation Functions')
plt.xlabel('$z$')
plt.ylabel('$f(z)$')
plt.grid(True)
plt.legend()
plt.show()


# -

#
#
# ### Rectifier-Familie
#
# | Funktion | Definition | Vorteile | Nachteile | Typische Verwendung |
# |---|---|---|---|---|
# | **ReLU** | $$\max(0,z)$$ | Sehr günstig; kein positiver Sättigungsbereich; oft schnelle Optimierung; sparse Aktivierungen | Nullgradient für $$z<0$$, dadurch „dying ReLU“; nicht nullzentriert; unbeschränkt; Knick bei null; zweite Ableitung fast überall null | Standard-Ausgangspunkt für MLPs und CNNs, besonders wenn Einfachheit und Inferenzkosten wichtig sind[^12][^13] |
# | **Leaky ReLU** | $$\max(z,\alpha z)$$, meist kleines festes $$\alpha>0$$ | Nichtverschwindender Gradient im negativen Ast; reduziert tote Neuronen; nahezu so günstig wie ReLU | Zusätzlicher Hyperparameter; negative Seite bleibt unbeschränkt; Qualitätsgewinn gegenüber ReLU nicht garantiert | CNNs/MLPs mit vielen inaktiven ReLUs; verbreitet auch in GAN-Discriminators; ursprünglich in tiefen akustischen Modellen untersucht[^14][^15] |
# | **PReLU** | Wie Leaky ReLU, aber $$\alpha$$ wird gelernt | Negative Steigung passt sich Daten bzw. Kanal an; kann ReLU oder Leaky ReLU als Spezialfall annähern | Zusätzliche Parameter; mögliche Überanpassung; weniger portabel und etwas komplexer | Tiefe CNNs, wenn ein lernbarer Rectifier empirisch validiert werden kann; 2015 von He et al. eingeführt[^16][^17] |
# | **ELU** | $$z$$ für $$z\ge0$$, sonst $$\alpha(e^z-1)$$ | Negative Ausgaben; negativer Ast glatt und nach unten beschränkt; Aktivierungsmittel kann näher an null liegen | Exponentialfunktion teurer; negative Sättigung erzeugt dort kleine Gradienten; bei null nur unter passender Parameterwahl glatt | Tiefe MLPs/CNNs, wenn ReLU-Probleme auftreten und Rechenkosten zweitrangig sind[^18][^9] |
# | **SELU** | Skalierte ELU mit festgelegten Konstanten | Kann unter den vorgesehenen Annahmen Aktivierungen in Richtung Mittelwert null und Varianz eins treiben | Selbstnormalisierung hängt von Architektur, Initialisierung und geeignetem Dropout ab; kein beliebiger Drop-in-Ersatz | Speziell konstruierte vollverbundene selbstnormalisierende Netze, nicht allgemeiner Standard[^19][^20] |
# | **ReLU6 / Hard-Swish** | Gekappte bzw. stückweise lineare Approximationen | Begrenzter Wertebereich oder günstige Approximation; quantisierungs- und hardwarefreundlicher | Zusätzliche Knicke; Approximation kann Genauigkeit kosten | Mobile und eingebettete Netze; ReLU6 wird etwa als typische mobile Variante dokumentiert[^1][^8] |
#
#

# +
def relu(z):
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    return np.maximum(alpha * z, z)

# Compute activations
r = relu(z)
l = leaky_relu(z, alpha=0.04)

# Plot
plt.figure(figsize=(8,8/1.61))
plt.plot(z, r, label='ReLU', color='blue')
plt.plot(z, l, label='Leaky ReLU', color='green')
# plt.plot(z, r, label='ReLU', color='red')

plt.title('Common Activation Functions')
plt.xlabel('$z$')
plt.ylabel('$f(z)$')
plt.grid(True)
plt.legend()
plt.show()


# -

#
#
# ### Glatte moderne Funktionen
#
# | Funktion | Definition | Vorteile | Nachteile | Typische Verwendung |
# |---|---|---|---|---|
# | **Softplus** | $$\log(1+e^z)$$, ggf. skaliert mit $$\beta$$ | Glatte Approximation von ReLU; überall positiver Gradient; gut differenzierbar | Nie exakt null, daher keine echte Aktivierungssparsity; Exponential/Logarithmus teurer; positiver Ast nur asymptotisch linear | Positive Parametrisierungen, wissenschaftliches ML und Netze mit Glattheitsanforderungen[^8][^2] |
# | **GELU** | $$z\Phi(z)$$, $$\Phi$$: CDF der Standardnormalverteilung | Glatt; weiches, eingabeabhängiges Gating statt hartem Vorzeichen-Schwellwert; kleine negative Ausgaben möglich | Teurer als ReLU; exakte Auswertung erfordert $$\Phi$$ bzw. erf, meist Approximation; nicht sparse | Transformer-MLPs und große Sprach-/Visionmodelle; BERT verwendet GELU anstelle von ReLU[^21][^22][^23] |
# | **SiLU / Swish** | $$z\sigma(z)$$ | Glatt, nichtmonoton und selbst-gatend; positiver Ast asymptotisch linear; Gradient fließt auch für moderate negative Werte | Teurer als ReLU; nicht exakt sparse; empirischer Vorteil architekturabhängig | Moderne CNNs, MLPs und Transformer-Varianten; in Bibliotheken heißt Swish mit $$\beta=1$$ meist SiLU[^24][^9][^13] |
# | **Mish** | $$z\tanh(\operatorname{softplus}(z))$$ | Glatt, nichtmonoton, unbeschränkt positiv und negativ nach unten beschränkt | Noch höhere Rechenkosten; weniger standardisiert; Vorteile stark empirisch und aufgabenspezifisch | Experimentelle CNNs/Detektoren, wenn Benchmarks einen messbaren Nutzen zeigen[^25] |
# | **Sine** | $$\sin(\omega_0 z)$$ | Glatt in allen Ordnungen; periodischer induktiver Bias; kann hohe Frequenzen und Ableitungen darstellen | Sehr initialisierungs- und frequenzempfindlich; periodische Mehrdeutigkeit; kein allgemeiner Drop-in-Ersatz | Implizite neuronale Repräsentationen, Wellenfelder, Akustik und bestimmte PDE-Probleme; SIREN kombiniert Sinus mit spezieller Initialisierung[^26][^27] |
#
#

# +
def softplus(z, beta=1.0):
    return beta * np.log(1 + np.exp(z / beta))

def swish(z):
    return z * sigmoid(z)

# Compute activations
s = softplus(z, beta=1.0)
w = swish(z)

# Plot
plt.figure(figsize=(8,8/1.61))
plt.plot(z, s, label='Softplus', color='blue')
plt.plot(z, w, label='Swish', color='green')
# plt.plot(z, r, label='ReLU', color='red')

plt.title('Common Activation Functions')
plt.xlabel('$z$')
plt.ylabel('$f(z)$')
plt.grid(True)
plt.legend()
plt.show()
# -

#
#
# ## Ausgabeschicht richtig wählen
#
# Die Hidden-Layer-Aktivierung und die Ausgabetransformation erfüllen unterschiedliche Aufgaben. Eine praktische Zuordnung ist:
#
# | Aufgabe | Übliche Ausgabe | Begründung |
# |---|---|---|
# | Unbeschränkte Regression | Linear | Jede reelle Vorhersage bleibt möglich |
# | Strikt positive Größe | Softplus oder Exponentialfunktion | Positivität wird konstruktiv erzwungen; Softplus ist meist numerisch sanfter |
# | Wert im Intervall $$[0,1]$$ | Sigmoid | Direkte Beschränkung auf ein Einheitsintervall |
# | Wert im Intervall $$[-1,1]$$ | tanh | Symmetrisch begrenzte Ausgabe |
# | Binäre Klassifikation | Ein Logit während des Trainings; Sigmoid zur Wahrscheinlichkeitsausgabe | Numerisch stabile Loss-Funktionen kombinieren Sigmoid und binäre Cross-Entropy intern |
# | Exklusive Mehrklassenklassifikation | Logits plus Softmax/Cross-Entropy | Softmax erzeugt eine gemeinsame kategoriale Verteilung[^9][^11] |
# | Multi-Label-Klassifikation | Eine Sigmoid-Ausgabe pro Label | Labels sind nicht gegenseitig exklusiv; jede Komponente modelliert eine Bernoulli-Wahrscheinlichkeit |
#
# Für numerische Stabilität sollte man dem Loss nach Möglichkeit **Logits** statt bereits gerundeter oder manuell exponentierter Wahrscheinlichkeiten übergeben. Beispielsweise vermeiden kombinierte Cross-Entropy-Implementierungen unnötige Rundungs- und Überlaufprobleme.
#
# ## Historische Entwicklung
#
# ### Von Schwellwerten zu Gradienten
#
# - **1943 – McCulloch und Pitts:** Das künstliche Neuron wird als binäre Schwellwerteinheit formuliert: Es feuert, wenn die Gesamterregung einen festen Schwellenwert erreicht und keine hemmende Bedingung greift. Damit stand zunächst logische Berechenbarkeit, nicht differenzierbare Optimierung im Mittelpunkt.[^28][^7]
# - **1958 – Rosenblatt:** Das Perzeptron verwendet weiterhin ein Alles-oder-nichts-Modell mit festem Schwellenwert. Rosenblatt unterschied es ausdrücklich von kontinuierlichen Transduktormodellen.[^6][^29]
# - **1980er – Sigmoid und tanh:** Mit differenzierbaren S-Kurven ließen sich Gradienten durch mehrere Schichten zurückpropagieren. Rumelhart, Hinton und Williams popularisierten 1986 Backpropagation für Netze mit versteckten Einheiten und zeigten, dass diese interne Merkmale lernen können.[^30][^31]
# - **1989 – Approximationstheorie:** Cybenko bewies, dass endliche Linearkombinationen einer kontinuierlichen sigmoiden Nichtlinearität mit einer versteckten Schicht stetige Funktionen auf einem kompakten Gebiet beliebig gut approximieren können. Das ist ein Repräsentations-, kein Optimierungssatz: Er garantiert nicht, dass Training das passende Netz effizient findet.[^32][^33]
#
# ### Aufstieg der Rectifier
#
# - **1990er – Gradientenproblem:** Bei tiefen und rekurrenten Netzen wurde deutlich, dass wiederholte Multiplikation kleiner Aktivierungsableitungen Gradienten exponentiell abschwächen kann. Sättigende Sigmoid- und tanh-Funktionen verschärfen dieses Problem.[^34]
# - **2010 – ReLU in modernen Netzen:** Nair und Hinton verwendeten deterministisch $$\max(0,z)$$ für die Rückpropagation in rectified units und berichteten Vorteile gegenüber binären Hidden Units.[^12]
# - **2011 – Tiefe Rectifier-Netze:** Glorot, Bordes und Bengio zeigten, dass Rectifier-Netze gleich gute oder bessere Ergebnisse als tanh-Netze erreichen und echte sparse Repräsentationen bilden können.[^4]
# - **2012 – AlexNet:** Der großskalige Erfolg von AlexNet machte ReLU zum dominanten Standard in CNNs. Im zugehörigen Vergleich erreichte ein ReLU-Netz auf CIFAR-10 eine festgelegte Trainingsfehlerrate etwa sechsmal schneller als ein äquivalentes tanh-Netz.[^35]
# - **2013–2017 – Varianten:** Leaky ReLU, PReLU, ELU und SELU adressierten tote Neuronen, fehlende negative Aktivierungen oder die Stabilisierung der Aktivierungsstatistik.[^16][^18][^19]
#
#

# +
import numpy as np
import matplotlib.pyplot as plt

# Define range for input z
z = np.linspace(-5, 5, 100)

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

# Compute activations
sig = sigmoid(z)
t = tanh(z)
r = relu(z)

# Plot
plt.figure(figsize=(8,8/1.61))
plt.plot(z, sig, label='Sigmoid', color='blue')
plt.plot(z, t, label='Tanh', color='green')
plt.plot(z, r, label='ReLU', color='red')

plt.title('Common Activation Functions')
plt.xlabel('$z$')
plt.ylabel('$f(z)$')
plt.grid(True)
plt.legend()
plt.show()
# -

#
#
# ### Glatte und gegatete Funktionen
#
# - **2016 – GELU:** Hendrycks und Gimpel führten $$z\Phi(z)$$ als probabilistisch motivierte, glatte Nichtlinearität ein.[^21][^36]
# - **2017 – Swish:** Eine automatisierte Suche identifizierte die selbst-gegatete Funktion $$z\sigma(z)$$. Die ursprüngliche Studie berichtete bei einigen tiefen Bildmodellen kleine, aber messbare Vorteile gegenüber ReLU; solche Gewinne sind nicht als universelle Garantie zu verstehen.[^24][^37]
# - **2018 – Transformer-Ära:** BERT setzte GELU in seinen Feedforward-Blöcken ein und trug damit zur breiten Verwendung glatter Aktivierungen in Transformern bei.[^23]
# - **2020 – GLU-Varianten:** ReGLU, GEGLU und SwiGLU verbinden eine Aktivierung mit multiplikativem Gating zweier Projektionen. Experimente in Transformer-Feedforward-Schichten zeigten Vorteile gegenüber einfachen ReLU- oder GELU-Blöcken, allerdings mit einer anderen Parameterisierung und nicht bloß durch Austausch einer skalaren Funktion.[^38]
# - **2020er – domänenspezifische Aktivierungen:** Periodische SIREN-Netze adressieren implizite Signale und PDEs; KANs verschieben lernbare univariate Aktivierungen von Knoten auf Kanten und parametrisieren sie unter anderem mit Splines. Letztere sind ein eigener Architekturansatz und nicht einfach eine neue Standardaktivierung für MLPs.[^26][^39]
#
# ## Anwendungsempfehlungen
#
# | Szenario | Sinnvoller Startpunkt | Wann abweichen? |
# |---|---|---|
# | Standard-MLP auf tabellarischen Daten | ReLU oder SiLU | GELU/SiLU testen, wenn ReLU-Training instabil ist; Leaky ReLU bei vielen toten Einheiten |
# | CNN / Bildverarbeitung | ReLU; bei moderner Referenzarchitektur deren Aktivierung übernehmen | SiLU in dafür entworfenen Architekturen; ReLU6/Hard-Swish für mobile oder quantisierte Inferenz |
# | Transformer | Referenzarchitektur beibehalten: häufig GELU oder SwiGLU | Wechsel nur mit kontrolliertem Vergleich, da FFN-Breite, Initialisierung und Gating gekoppelt sind[^38][^23] |
# | Klassisches RNN | tanh für Zustand, Sigmoid für Gates | Moderne gated Zellen oder Normalisierung prüfen, wenn lange Abhängigkeiten problematisch sind |
# | GAN-Discriminator | Leaky ReLU als verbreiteter Startpunkt | Architektur- und domainspezifische Vorgaben beachten |
# | Positive physikalische Parameter | Softplus-Ausgabe | Exponentialfunktion, wenn log-lineare Parametrisierung fachlich sinnvoll ist; numerischen Wertebereich kontrollieren |
# | PINN mit PDE-Ableitungen | tanh, Sinus oder andere ausreichend glatte Funktion | Aktivierungsordnung an Differentialoperator und Lösungsspektrum anpassen; ReLU bei zweiten oder höheren klassischen Ableitungen vermeiden[^2][^3] |
# | Implizite Repräsentation / hochfrequentes Feld | SIREN oder Fourier-Features plus glattes MLP | Spezielle Initialisierung und Frequenzskalierung gemeinsam abstimmen[^26][^40] |
#
# ## Hinweise für PINNs
#
# Für ein PINN, dessen Residuum einen Differentialoperator $$\mathcal{L}$$ der Ordnung $$k$$ enthält, wird während des Trainings typischerweise $$\partial^k u_\theta/\partial x^k$$ mittels automatischer Differenzierung benötigt. ReLU ist stückweise linear: Die zweite Ableitung ist fast überall null und am Knick nicht klassisch definiert. Ein ReLU-PINN kann daher trotz universeller Approximationsfähigkeit als Funktion einen ungeeigneten Raum für das **starke** PDE-Residuum aufspannen.
#
# Glatte Funktionen wie tanh, Softplus, Sinus, GELU oder SiLU sind hierfür grundsätzlich plausibler, aber die Glattheit allein garantiert keine gute Optimierung. Aktuelle Analysen betonen die Rolle wohlerhaltener höherer Ableitungen; für bestimmte Differentialordnungen können Sinus- oder Cosinusaktivierungen günstiger als tanh sein.[^2] SIREN zeigt zudem, dass periodische Aktivierung und dazu passende Initialisierung gemeinsam betrachtet werden müssen, um Signale und deren Ableitungen robust darzustellen.[^26][^27]
#
# Für schwache oder variationelle Formulierungen kann die Anforderung geringer sein, weil partielle Integration die benötigte Ableitungsordnung des neuronalen Ansatzes reduziert. Die Wahl sollte daher nicht nur von der PDE, sondern auch von starker versus schwacher Form, Spektralgehalt, Skalierung der Koordinaten und Initialisierung abhängen.
#
# ## Praktischer Entscheidungsablauf
#
# 1. **Ausgabebereich festlegen:** Linear, Sigmoid, Softmax, tanh oder positive Parametrisierung anhand der Zielvariable wählen.
# 2. **Baseline übernehmen:** Bei etablierter Architektur zunächst die veröffentlichte Aktivierung und Initialisierung beibehalten; ein isolierter Austausch kann Signalvarianz und Optimierungsdynamik verändern.
# 3. **Einfach starten:** Für generische MLPs/CNNs ReLU verwenden; für Transformer das architekturspezifische GELU- oder GLU-Design; für PINNs zunächst tanh oder eine passend skalierte glatte Funktion.
# 4. **Diagnostizieren:** Anteil exakt nuller Aktivierungen, Verteilung der Voraktivierungen, Gradienten je Schicht, Trainings-/Validierungsverlust und Inferenzkosten protokollieren.
# 5. **Kontrolliert vergleichen:** Aktivierung, passende Initialisierung und gegebenenfalls Lernrate gemeinsam abstimmen, aber Datensplit, Seeds, Parameterbudget und Trainingsbudget konstant halten.
# 6. **Nicht nur Endfehler betrachten:** Konvergenzzeit, Robustheit zwischen Seeds, Kalibrierung, Speicherbedarf, Durchsatz und bei wissenschaftlichem ML zusätzlich Fehler in Ableitungen bzw. Residuen prüfen.
#
# ## Faustregeln
#
# - **ReLU** ist die einfachste robuste Baseline für viele Hidden Layers.
# - **Leaky ReLU** ist eine minimale Änderung, wenn tote ReLUs beobachtet werden.
# - **GELU oder SiLU** sind gute glatte Kandidaten für tiefe moderne Architekturen, aber kein automatischer Qualitätsgewinn.
# - **Sigmoid und Softmax** gehören primär in Ausgänge oder Gates, nicht als Default in tiefe Hidden Layers.
# - **tanh, Softplus und Sinus** verdienen bei PINNs und impliziten Feldern besondere Aufmerksamkeit, weil Ableitungsstruktur und induktiver Bias zählen.
# - Die Aktivierungsfunktion sollte stets zusammen mit **Initialisierung, Normalisierung, Residualverbindungen, Loss-Skalierung und Hardwareziel** bewertet werden; sie ist kein isolierter Hyperparameter.
#
# ---
#
# ## References
#
# 1. [activation.md.txt](https://docs.pytorch.org/cppdocs/_sources/api/nn/activation.md.txt)
#
# 2. [www.ijcai.org › proceedings › 2024Physics-Informed Neural Networks: Minimizing Residual Loss ...](https://www.ijcai.org/proceedings/2024/0647.pdf)
#
# 3. [Artificial to Spiking Neural Networks Conversion for Scientific Machine Learning Thanks: Citation: Authors. Title. Pages…. DOI:000000/11111.](https://arxiv.org/html/2308.16372v1)
#
# 4. [Deep Sparse Rectifier Neural Networks](https://proceedings.mlr.press/v15/glorot11a.html) - While logistic sigmoid neurons are more biologically plausible than hyperbolic tangent neurons, the ...
#
# 5. [Deep Sparse Rectifier Neural Networks](https://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf)
#
# 6. [web.engr.oregonstate.edu › rosenblatt-1958THE PERCEPTRON: A PROBABILISTIC MODEL FOR INFORMATION STORAGE ...](https://web.engr.oregonstate.edu/~huanlian/teaching/ML/2026fall/extra/rosenblatt-1958.pdf)
#
# 7. [Pencil-and-Paper Neural Networks: An Undergraduate ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC4640478/) - Although it has been more than 70 years since McCulloch and Pitts published their seminal work on ar...
#
# 8. [docs.pytorch.org › docs › maintorch.nn.functional — PyTorch main documentation](https://docs.pytorch.org/docs/main/nn.functional.html)
#
# 9. [NN ops - Keras](https://keras.io/api/ops/nn/) - Keras documentation: NN ops
#
# 10. [link.springer.com › chapter › 10Approximation Error of Sobolev Regular Functions with Tanh ...](https://link.springer.com/chapter/10.1007/978-3-031-70359-1_16) - Considering the key role played by derivatives in Partial Differential Equations (PDEs), using the t...
#
# 11. [Module: tf.keras.activations - TensorFlow 2.4 - W3cubDocs](https://docs.w3cub.com/tensorflow~2.4/keras/activations) - Built-in activation functions.
#
# 12. [[PDF] Rectified Linear Units Improve Restricted Boltzmann Machines](https://papers.baulab.info/papers/also/Nair-2010.pdf)
#
# 13. [pytorch/torch/nn/modules/activation.py at v2.10.0 · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/nn/modules/activation.py) - Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
#
# 14. [[PDF] Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)
#
# 15. [tf.nn.leaky_relu | TensorFlow v2.16.1](https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu) - Compute the Leaky ReLU activation function.
#
# 16. [Empirical Evaluation of Rectified Activations in Convolution Network](https://ar5iv.labs.arxiv.org/html/1505.00853) - In this paper we investigate the performance of different types of rectified activation functions in...
#
# 17. [[PDF] Deep Residual Learning for Image Recognition | Semantic Scholar](https://www.semanticscholar.org/paper/Deep-Residual-Learning-for-Image-Recognition-He-Zhang/2c03df8b48bf3fa39054345bafabfeff15bfd11d) - This work presents a residual learning framework to ease the training of networks that are substanti...
#
# 18. [V2](https://arxiv.org/abs/1511.07289v2) - We introduce the "exponential linear unit" (ELU) which speeds up learning in deep neural networks an...
#
# 19. [Self-Normalizing Neural Networks - alphaXiv](https://www.alphaxiv.org/abs/1706.02515) - Self-Normalizing Neural Networks (SNNs) introduce Scaled Exponential Linear Units (SELUs) which math...
#
# 20. [Activations](https://keras-contrib.readthedocs.io/en/latest/sources/activations/)
#
# 21. [[PDF] Gaussian Error Linear Units (GELUs) - Semantic Scholar](https://www.semanticscholar.org/paper/Gaussian-Error-Linear-Units-(GELUs)-Hendrycks-Gimpel/de5e7320729f5d3cbb6709eb6329ec41ace8c95d) - An empirical evaluation of the GELU nonlinearity against the ReLU and ELU activations is performed a...
#
# 22. [README.md - hendrycks/GELUs - GitHub](https://github.com/hendrycks/GELUs/blob/master/README.md) - A smoother activation function (undergrad code). Contribute to hendrycks/GELUs development by creati...
#
# 23. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/html/1810.04805v1)
#
# 24. [Swish: a Self-Gated Activation Function](https://arxiv.org/abs/1710.05941v1) - The choice of activation functions in deep networks has a significant effect on the training dynamic...
#
# 25. [Mish: A Self Regularized Non-Monotonic Activation Function](https://arxiv.org/html/1908.08681v3)
#
# 26. [Implicit Neural Representations with Periodic](https://proceedings.neurips.cc/paper/2020/file/53c04118df112c13a8c34b38343b9c10-Paper.pdf)
#
# 27. [Orals & Spotlights Track 13: Deep Learning/Theory - NeurIPS 2020](https://neurips.cc/virtual/2020/protected/session_oral_21093.html)
#
# 28. [A logical calculus of the ideas immanent in nervous activity](https://marlin.life.utsa.edu/mcculloch-and-pitts.html) - Discussion of “A logical calculus of the ideas immanent in nervous activity”, by McCulloch and Pits
#
# 29. [Psychological Review](https://web.engr.oregonstate.edu/~huanlian/teaching/ML/2025fall/extra/rosenblatt-1958.pdf)
#
# 30. [Learning representations by back-propagating errors - Nature](https://www.nature.com/articles/323533a0) - We describe a new learning procedure, back-propagation, for networks of neurone-like units. The proc...
#
# 31. [Geoffrey E. Hinton's Publications: Backpropagation](https://www.cs.toronto.edu/~hinton/backprop.html)
#
# 32. [Approximation by superpositions of a sigmoidal function](https://hal.science/hal-03753170v1/file/Cybenko1989.pdf)
#
# 33. [[PDF] Approximation by superpositions of a sigmoidal function - HAL](https://hal.science/hal-03753170v1/document)
#
# 34. [Vanishing gradient problem - Wikipedia](https://en.wikipedia.org/wiki/Vanishing-gradient_problem)
#
# 35. [CMU School of Computer Science](https://www.cs.cmu.edu/~jeanoh/16-785/papers/krizhevsky-nips2012-alexnet.pdf)
#
# 36. [Gaussian Error Linear Units (GELUs) - alphaXiv](https://www.alphaxiv.org/abs/1606.08415v5) - Gaussian Error Linear Units (GELUs) proposes a new activation function that inherently combines prop...
#
# 37. [Three Decades of Activations: A Comprehensive Survey of 400 ...](https://www.alphaxiv.org/abs/2402.09092) - Kunc and Kléma from Czech Technical University in Prague compiled a comprehensive survey of 400 real...
#
# 38. [GLU Variants Improve Transformer](https://www.alphaxiv.org/audio/2002.05202) - Google researcher Noam Shazeer demonstrates that replacing standard activation functions in Transfor...
#
# 39. [[2404.19756] KAN: Kolmogorov-Arnold Networks - arXiv.org](https://arxiv.org/abs/2404.19756) - Inspired by the Kolmogorov-Arnold representation theorem, we propose Kolmogorov-Arnold Networks (KAN...
#
# 40. [NeurIPS 2020 : Implicit Neural Representations with Periodic ...](https://neurips.cc/virtual/2020/public/poster_53c04118df112c13a8c34b38343b9c10.html)
#
#
