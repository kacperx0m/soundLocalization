# soundLocalization

kanały wydają się git, lewy w stosunku do prawego przed transpozycja od 1.36 do 0.73, load_mono, iterate_convolute

wykresy się nie pokrywają mojego ITD z tym ze zdjęcia, mój wykres idzie troche wyżej dla button
a dla horseDB jest lipa dla wartości ujemnych

ITD:
frogs git, horse ma skoki, music git
dlaczego w okolicy -60 i 60 są te skoki?

czy w sposobie w jaki obliczam ILD i ITD uwzględniam częstotliwości? chyba niekoniecznie
czat zalecił filtrowanie częstotliwości - ild i itd działają lepiej dla różnych zakresów

ILD:
music i frogs mają takie same ILD wtf o co chodzi??

może obliczyć ILD i ITD bezpośrednio na odpowiedzi HRTF

co oznacza 1, skala liniowa amp, 
chat : W większości kontekstów audio, wartość 1 na skali amplitudy oznacza maksymalny poziom sygnału, który można zapisać bez zniekształceń w systemach o znormalizowanej amplitudzie (np. pliki PCM z zakresem [-1, 1])

test3.py
zrobić sprawdzanie folderu czy pusty?

decibels relative to full scale - dBFS
loudness relative to full scale - LUFS

loudness normalization - chyba to
peak normalization

może użyć fftconvolve z scipy.signal?

normalizowanie białego szumu maxowaniem do 1 jest bez sensu


model ai:
regresja liniowa
na pewno 2 inputy, lewy i prawy kanał, a wysokość i odległość?

z kanałów ekstraktuję informacje takie jak ild, itd
itd przy niższych częstotliwościach

ild przy wyższych częstotliwościach
amplituda jednego kanału do drugiego

widmo dźwięku?
kanały powinny być zsynchronizowane czasowo

na wyjściu kąt

cnn?
beamforming do ogólnego lokalizowania
mfcc

dla -35 w lufs_horseDB dopiero left/right kanał wynosi 1
w przypadku horseDB w -5

jak jakiś kanał przebije 0.2 to wtedy szukam itd?
korelacja krzyżowa czat zaproponował
=========

Opisane powyżej metody (ILD, ITD) pozwalają na lokalizację dźwięków tylko w kierunku od lewej do prawej. 
Ich ograniczeniem jest też rozdzielczość czasowa słuchu (możliwość rozróżnienia dwóch dźwięków) i najmniejsza zauważalna zmiana głośności, różna w zależności od poziomu dźwięku. 
Bardziej precyzyjnym sposobem lokalizacji źródła dźwięku jest funkcja przejścia głowy (ang. HRTF – head-related transfer function), określająca w jaki sposób obecność naszej głowy wpływa na falę akustyczną. 
Funkcja przejścia głowy jest indywidualna dla każdego człowieka i zależy od wielkości głowy, kształtu uszu i innych anatomicznych szczegółów. 
Poszczególne elementy ciała mogą odbijać, pochłaniać lub rozpraszać dźwięki o określonych częstotliwościach i dochodzących z określonych kierunków, wpływając w ten sposób na to, co słyszymy. 
Dzięki niej możliwe jest określenie nie tylko, czy dźwięk dociera do słuchacza z lewej czy z prawej strony, lecz także czy dociera z przodu czy z tyłu. 
Odległość od źródła dźwięku jest określana przez udział wysokich częstotliwości, a także przez ocenę udziału dźwięków odbitych i bezpośrednich w sygnale. 
Kiedy rozumiemy, jak funkcjonuje słuch i przetwarzanie dźwięku w mózgu, zauważamy, jak ważna jest adaptacja akustyczna przestrzeni. 
Dzięki niej możliwe jest sprawne odnajdowanie źródeł dźwięku w przestrzeni, jak i uzyskanie zgodnego obrazu tego, co słyszymy z tym, co widzimy.


The shape of the ears, head, and torso definitely affect ILD because they modify the sound waves' frequency and intensity as they reach the ears. The small variations in your data (like 1.03 instead of exactly 1 at 0°) can be attributed to these natural physical effects

https://pressbooks.umn.edu/sensationandperception/chapter/interaural-time-difference-draft/

ILD for sources at least 1m away:
https://www.youtube.com/watch?v=t0P2uQP2U0k

https://courses.washington.edu/psy333/lecture_pdfs/Week9_Day2.pdf

prędkość propagacji dźwięku ~ 1238,4km/h wyliczone z 344m/s (ok 20 stopni) lub 1224.7km/h wyliczone z 761mph (ok 10 stopni)

z tej strony:
https://pubs.aip.org/asa/jel/article/1/4/044402/219435/A-set-of-equations-for-numerically-calculating-the
\
For ITDs, it is typical to use the expression (rθ + r sin θ)/c, where r is the radius of the head, θ is the azimuth of the sound source, in radians, and c is the speed of sound\
nie bierzę pod uwagę częstotliwości\
ILD = 0.18 * sqrt(f) * sin(θ) - Van Opstal, J. (2016). The Auditory System and Human Sound-Localisation Behavior (Academic Press, London).

wzory są też na wikipedii w sound_localization
