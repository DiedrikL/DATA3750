# The Energy of a Quantum Physical Two-Body System
Dette skriptet omhandler prosjektet “The Energy of a Quantum Physical Two-Body System” utført i forbindelse med emnet
DATA3750 (Anvendt kunstig intelligens og data science prosjekt) femte semester på dataingeniør-linjen ved OsloMet.

Skriptet vil, for gitte kvantesystemer med opptil to partikler, estimere grunntilstanden ved å minimere energien med gradient descent.
Beregningene kan visualiseres med plott.

## Hvordan bruke skriptet
Input-parametere mates inn ved hjelp av config-filen som liger i mappen quantum_energy. Du kan endre verdiene direkte i filen. Husk å lagre for å kjøre skriptet med de oppdaterte verdiene.
```
[PARAMS]
# One particle and two particle parameters
x0 = 3
a = 4
# Optional parameter for one particle
#b = 1

[TWO-PARTICLE]
# Interaction term for two-particle system
w0 = 1

[NUMERICS]
# Lenght of interval
L = 20
# Number of subintervals
N = 500
# Learning rate for gradient descent algorithm
lr = 0.5
# Maximum number of iterations in gradient descent
max_iter = 15000

[CONFIGURATION]
plot = True
function = func1
num_particles = 1
interactive = True
```
* **x0**, **a**: Verdier du ønsker å gjette på for x_0 og a/sigma.
* **b**: valgfritt tredje parameter for estimat med én partikkel
* **w_0**: Vekselvirking for systemer med to partikler
* **L**: Lengden på intervallet
* **N**: Antall subintervall
* **lr**: læringsraten til gradient descent algoritmen
* **max_iter**: Maks antall iterasjoner i gradient descent
* **plot**: Skriv **True** om du ønsker plott, **False** om du ikke ønsker plott
* **function**: Velg blant **func1** og **func2** (se besrkivelse nedenfor)
* **num_particles**: Her velger du om du ønsker estimering for én eller to partikler. b-parameteren må kommenteres ut ved to partikler.
* **interactive**: Om du ønsker å plotte flere stier kan du sette til **True**. Du får da mulighet til å gjette på (og plotte) så mange parametere du ønsker. Sett til **False** om du ikke ønsker dette.  
  
Når argumentene i config-filen er justert kan du kjøre skriptet fra kommandolinjen.
```
python -m quantum_energy
```

### Funksjoner


### Komme i gang med Python
Skriptet krever Python 3.8 eller senere. Det er ikke testet på tidligere versjoner av Python. Du kan sjekke hvilken versjon du har med kommandoen:
```
python --version
```
For å installere Python gå til [Python hjemmeside](https://www.python.org/downloads/).  
Om du bruker Linux kan du installere direkte fra terminalen ved å skrive:
```
$ sudo apt-get update
$ sudo apt-get install python3.8
```
### Avhengigheter
Skriptet importerer følgende biblioteker:
* NumPy  
NumPy kan installeres med Python package manager, pip, ved å skrive inn følgende kommando:
```
pip install numpy
```
* Matplotlib  
Matplotlib kan installeres med pip ved å skrive inn følgende kommando:
```
pip install matplotlib
```

## Laget av / kontaktopplysninger
Diedrik Leijenaar Oksnes, s181138, s181138@oslomet.no  
Sebastian Overskott, s331402, s331402@oslomet.no  
Aleksander Røv, s187428, s187428@oslomet.no  
Vegard Müller, s150315, s150315@oslomet.no 

## Lisens
