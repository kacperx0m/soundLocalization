# soundLocalization

czat:
bufor do finalnego algorytmu?
Buforowanie sygnału: Możesz przechowywać sygnał w buforze (pierwsze 50 ms) i przesuwać się o krok (np. 25 ms), aby analizować częściowo nakładające się okna. To zapewni bardziej płynne działanie systemu.
Filtracja na bieżąco: Można rozważyć przetwarzanie sygnału w czasie rzeczywistym przy użyciu sosfiltfilt, która pozwala filtrować na bieżąco, aby uniknąć filtrowania całych segmentów naraz.
czy przesuwanie w tak małych oknach będzie miało sens? w sumie opóźnienia dźwięku to mikrosekundy czyli zauważy

wyobrażam sobie że będzie pętla która nasłuchuje, jeżeli wartość sygnału odbieranego przez mikrofon nie przekroczy jakiejś wartości to nic nie liczę a jeżeli przekroczy to przez 50ms rejestruję sygnał, filtruję go i liczę ild oraz itd i zwracam wynik do aplikacji pygame która obraca igłą jak kompas. 
i tak w kółko żeby jak najwierniej oddać kątowe położenie dźwięku. ale z każdymi kolejnymi 50ms wykonywanie programu będzie dłużej trwać. z tego powodu wydaje mi się że będę musiał wykorzystać wątki albo procesy aby mój program działał szybciej
kąt otrzymuję od wcześniej wyuczonego modelu regresyjnego. ale wydaje mi się że wnioskowanie na podstawie dwóch skalarów - ild i itd będzie dość szybkie. aplikacja może działać z opóźnieniem nie większym niż pół sekundy?

kanały wydają się git, lewy w stosunku do prawego przed transpozycja od 1.36 do 0.73, load_mono, iterate_convolute

wykresy się nie pokrywają mojego ITD z tym ze zdjęcia, mój wykres idzie troche wyżej dla button
a dla horseDB jest lipa dla wartości ujemnych

ILD i ITD liczyć w oknach - tak będzie normalnie np 0.5s
ograniczyć do pewnych wartości bo może być pogłos i zepsuje wtedy
regresja liniowa, wrzucić tam pliki i liczyć

Zrobione:
- pliki są poza katlogami przy tworzeniu, trzeba mi je wrzucic do srodka
- ogarnac przy wczytywaniu z folderow bo jest miszmasz
- sprawdzić testowy model

TODO:
zrobić dataset, zapuścić trening na całym datasecie
from preprocessing import polynomialFeatures, SVR z kernelami
ridge, lasso, elastic net
nazwy w 01 do 18_backingvox sa do zmiany nazwy i tak samo z wynikowy-test?
sprawdzic wykresy ild i itd bo mozliwe ze sa bledy
sprawdzić które wykresy są tak wypłaszczone
skorzystać z ridge i polynomial
przesluchac wszystkie pliki bo niektore sa tragiczne zwlaszcza te anomalii

drzewa regresyjne, svm, klasyfikacja?, sgd, krr ciut szybszy od svr, knn
maxdepth= do drzew
.lasso do regresji liniowej
.ridge do regresji liniowej, regularyzacja l2 

lfilter, (sos)filtfilt, sosfilt
sprawdzić filtr butter na oryginalnym sygnale
filtry wyglądają dobrze korzystając z sosfreqz
sosfiltfilt , stack do ułożenia sygnałów
sosfiltfilt filtruje dwukierunkowo - najpierw przód a potem wstecz (ususwa przesunięcie fazowe)
sosfilt zwykły filtruje przyczynowo czyli wynik w danym momencie zależy od aktualnych i przeszłych wartości sygnału ale w efekcie mamy group delay
test3.py coś trochę modyfikuje mi sygnał, ucina niektóre częstotliwości, widać w audacity? 

ITD:
wygląda git dla horse i frogs
frogs git, horse ma skoki, music git
dlaczego w okolicy -60 i 60 są te skoki?

ILD:
wygląda git dla horse, fros trochę skacze
music i frogs mają takie same ILD wtf o co chodzi??

odsłuchać wyfiltrowanego sygnału górnego i dolnego

może obliczyć ILD i ITD bezpośrednio na odpowiedzi HRTF - slaby pomysl bo chcę poznać kąt z którego przychodzi odpowiedź

low pass oznacza że przepuszcza sygnały o f niższej niż f odcięcia a o wyższych f osłabia
high pass oznacze że przepuszcza sygnały o f wyższej niż f odcięcia a o niższych f osłabia
band pass oznacza że przepuszcza sygnały tylko z danego pasma f, low i high nałożone jednocześnie
fft wchodzi , parametry uwzględniane

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

=========================================================================
chat:

Bardziej zaawansowane rozwiązania do obliczania ILD i ITD:
Filtrowanie adaptacyjne: Zastosowanie filtrów adaptacyjnych (np. LMS, RLS) może poprawić dokładność obliczeń ILD, szczególnie w obecności szumu.
Metoda GCC-PHAT (Generalized Cross-Correlation with Phase Transform): Ta technika obliczania ITD jest bardziej odporna na szumy i zmieniające się warunki akustyczne. Działa na zasadzie korelacji krzyżowej z modyfikacją fazową, co zwiększa dokładność przy dużych odległościach między mikrofonami.
Dekonwolucja: W kontekście ILD można zastosować techniki dekonwolucji, aby usunąć wpływ odbić i pogłosów, które mogą zniekształcać wyniki.

Uśrednianie okienkowe polega na dzieleniu sygnału na okna o określonej długości i obliczaniu średniej dla każdej z tych sekcji. To pomaga wygładzić wykresy ILD i ITD oraz zredukować wpływ krótkotrwałych zmian amplitudy.
Możesz zaimplementować uśrednianie okienkowe na wynikach ILD i ITD, iterując po sygnale w oknach o określonym rozmiarze i stopniu nakładania się.

Zastosowanie uśredniania okienkowego do ILD i ITD:
Dlaczego to ważne?: Stosowanie uśredniania okienkowego pomoże w uzyskaniu bardziej płynnych i mniej skokowych wartości ILD i ITD, co przekłada się na bardziej stabilne predykcje w modelach ML.
Jak to zastosować?:
Po obliczeniu ILD lub ITD dla całego sygnału, zastosuj funkcję moving_average do wyników przed ich użyciem jako danych wejściowych do modelu ML.
W przypadku ITD, które są obliczane w czasie rzeczywistym (np. na fragmentach sygnału), można zastosować uśrednianie na bieżąco, aby redukować skoki i fluktuacje.

Zmiany i ulepszenia dla trybu quasi-rzeczywistego:
Obliczenia na bieżąco: W trybie quasi-rzeczywistym możesz używać technik obliczeń strumieniowych, takich jak sliding window z buforowaniem, aby przetwarzać dane w krótkich odcinkach, co pozwala na bieżące aktualizowanie wyników.
Redukcja obciążenia: Wprowadzenie zoptymalizowanych algorytmów ITD, takich jak GCC-PHAT, może zwiększyć dokładność przy jednoczesnym zmniejszeniu obciążenia obliczeniowego w porównaniu z pełną korelacją krzyżową.

Wyzwania i rozwiązania:
Złożoność obliczeniowa: Aby zachować efektywność, upewnij się, że obliczenia korelacji, RMS oraz filtrowanie są zoptymalizowane (np. poprzez użycie bibliotek takich jak numba do przyspieszenia pętli w Pythonie).
Szumy i artefakty: Stosowanie filtrów cyfrowych, takich jak filtry Butterwortha lub filtry medianowe, pomoże w redukcji szumów i artefaktów, które mogą wpływać na wyniki ILD i ITD.

