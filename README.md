### FOR ENGLISH SCROLL BELOW

# AI vs Real: detekcja i edycja obrazów generowanych przez AI

## Inspiracje

Obrazy generowane przez AI potrafią zachwycać. Kształty się zgadzają, kolory są na miejscu, kompozycja ma sens. A jednak portret z modelu dyfuzyjnego często wywołuje dyskomfort. Skóra wydaje się zbyt gładka, światło pada pod dziwnym kątem, kontrast sprawia wrażenie nienaturalnego. Trudno wskazać konkretną przyczynę. Obraz jest "podejrzany", ale powód pozostaje nieuchwytny.

To nie przypadek. Ludzki mózg ma osobny obszar do rozpoznawania twarzy, tzw. *fusiform face area*. Przez setki tysięcy lat ewolucja dostrajała ten układ do wyłapywania subtelności: emocji, intencji, autentyczności. Nawet drobne odchylenia od **rozkładu naturalnych twarzy** uruchamiają alarm.

Stąd pomysł na projekt, celowo staromodny. Konwolucyjna sieć neuronowa (CNN) wytrenowana na zadaniu "aparat vs. AI". Dla CNN obraz to nie pojęcia ("twarz", "uśmiech"), tylko **statystyczne wzorce pikseli**: tekstury, gradienty, mikro-szum. Dokładnie to, czego ludzkie oko nie rejestruje świadomie, a co podświadomie wpływa na ocenę. Istnieją też techniki wizualizacji, które pozwalają zobaczyć, na które fragmenty obrazu sieć patrzy przy podejmowaniu decyzji.

W trakcie pracy nad pierwszymi wynikami pojawiały się kolejne pytania. Ostatecznie zrodził się pomysł inspirowany latent space: wygenerować sekwencję obrazów od AI do realności. Analogicznie do przejścia w latent space od uśmiechniętej do smutnej twarzy.


<img  alt="image" src="https://github.com/user-attachments/assets/4ed65dd8-9b2f-4f34-8eb7-6d38602b89cf" />


## Proces

### Dataset

**Źródło.** Zbiór `Rajarshi-Roy-research/Defactify_Image_Dataset` udostępniony na platformie HuggingFace. Łącznie około 47 000 obrazów w sześciu kategoriach:

| Kategoria | Liczba | Pochodzenie |
| --- | --- | --- |
| Real | ~7 000 | Fotografie z ImageNet i COCO |
| Stable Diffusion 2.1 | ~7 000 | Open-source model dyfuzyjny |
| Stable Diffusion XL | ~7 000 | Następna generacja SD |
| Stable Diffusion 3 | ~7 000 | Najnowsza generacja SD |
| DALL-E 3 | ~7 000 | Model komercyjny OpenAI |
| Midjourney 6 | ~7 000 | Model komercyjny Midjourney |

Łącznie 5 generatorów AI naprzeciw jednej kategorii prawdziwych zdjęć. Daje to silną nierównowagę klas (około 6:1 na korzyść AI), kompensowaną w treningu przez ważenie funkcji straty. Klasa Real ma sześciokrotnie większą wagę niż AI.

**Etykietowanie.** Każdy obraz ma dwie etykiety: `Label_A` (binarne, Real=0, AI=1) oraz `Label_B` (sześć klas, identyfikuje konkretny generator). Klasyfikator korzysta tylko z `Label_A`, ale `Label_B` jest kluczowe dla późniejszej analizy per-generator.

### Model: transfer learning od EfficientNetB0

**Cel.** Wytrenować klasyfikator binarny mapujący obraz na prawdopodobieństwo $\text{prob}_{\text{AI}} \in [0, 1]$. Architektura musi być wystarczająco mocna do zadania, ale na tyle przejrzysta, by jej wnętrze dało się analizować technikami interpretacji.

**Punkt wyjścia.** EfficientNetB0 pretrenowany na ImageNet. To wybór celowy: około 5 milionów parametrów (model relatywnie mały), znana architektura, dostępne wagi, dobra równowaga między mocą wyrażania a kosztem treningu.

**Architektura.** Na backbone'ie EfficientNetB0 (kończącym się 1280-kanałową mapą cech 7×7) zbudowany jest prosty klasyfikator:

```
EfficientNetB0 → GlobalAveragePooling2D → Dropout(0.4) →
Dense(256, ReLU) → Dropout(0.3) → Dense(1, sigmoid)
```

**Kluczowa decyzja projektowa: częściowe odmrożenie backbone'u.** Standardowy transfer learning zamraża cały backbone i trenuje tylko head. W tym zadaniu takie podejście zawodzi z fundamentalnego powodu:

> ImageNet uczy sieć rozpoznawać *obiekty* (kot, samochód, most). Detekcja AI-generated wymaga rozpoznawania *artefaktów generatora*: subtelnych wzorców tekstury, statystyk częstotliwości, sygnatur kompresji. Cechy tego typu po prostu nie są zakodowane w wagach trenowanych na ImageNet.

Rozwiązanie: zamrożone są tylko pierwsze ~100 warstw (z około 236), pozostałe ~136 trenowane są razem z headem. Niskopoziomowe wykrywacze krawędzi i tekstur pozostają nienaruszone (są uniwersalne), a wyższe warstwy adaptują się do specyfiki AI vs Real.

## CNN debugging

Na co patrzy sieć, podejmując decyzję? Mamy dwie techniki: Grad-CAM oraz Saliency Map. Pozwalają lepiej zrozumieć, jakie części obrazu mają wpływ na decyzję AI vs Real. To może być właśnie to, co pokaże nam cechy obrazów AI.

### CAM (Class Activation Mapping)

CAM odpowiada na pytanie: **w którym miejscu obrazu** model znalazł istotne cechy? Sieć ma już ku temu wszystko, co trzeba. Mapy cech z ostatniej warstwy konwolucyjnej mówią *gdzie*, każda jest siatką aktywacji w przestrzeni obrazu. Wagi klasyfikatora mówią *co* się liczy dla danej klasy. Standardowo te dwie informacje się rozjeżdżają, bo warstwa GlobalAveragePooling uśrednia każdą mapę do jednej liczby i traci "gdzie". CAM odwraca kolejność: najpierw wymnaża mapy cech przez wagi klasyfikatora, dopiero potem uśrednia. Wynik to mapa przestrzenna, którą skaluje się do rozmiaru obrazu jako heatmapę.

### Saliency Map

Saliency odpowiada na pytanie: **które piksele** najmocniej zadecydowały o predykcji? Mechanizm jest taki sam jak w treningu sieci: backpropagation, czyli liczenie gradientu. Tylko inaczej zakończony. W treningu gradient zatrzymuje się na wagach, by je poprawić. W Saliency płynie dalej, aż do pikseli wejścia, i tam zostaje zarejestrowany jako mapa istotności. Jasne piksele to te, których drobna zmiana najmocniej zmieniłaby wynik. Ciemne to te, na które model jest niewrażliwy.

<img alt="01_xai_panel" src="https://github.com/user-attachments/assets/22430000-c27f-40b9-8aca-eaf22524c06c" />

## Latent Space

Skoro wiadomo już, które miejsca i piksele mają znaczenie, można spróbować zmodyfikować zdjęcia AI tak, by wyglądały bardziej realnie. Inspiracją były algorytmy VAE i możliwość morfowania obrazów w latent space. Najlepszy przykład to wygenerowana twarz przechodząca od uśmiechniętej do smutnej. Celem było otrzymanie analogicznego przykładu, tylko od twarzy wygenerowanej przez AI do realnej.

Latent space to **wewnętrzna przestrzeń modelu generatywnego** (np. VAE, GAN), z której można tworzyć obrazy. Każdy punkt w niej odpowiada jakiemuś prawdopodobnemu obrazowi. Jest skompresowana, gładka i ciągła. Bliskie punkty dekodują się do podobnych obrazów, a płynne przejście między dwoma punktami daje płynne przejście między dwoma obrazami.

Żeby zrozumieć, *dlaczego* latent space w ogóle ma sens, pomocna jest analogia. Pomięta kartka papieru rzucona do pokoju fizycznie żyje w trójwymiarowej przestrzeni. Ale wewnętrznie jest **dwuwymiarowa**, wystarczą dwie współrzędne, żeby wskazać dowolny punkt na jej powierzchni. Kartka jest *manifoldem*: strukturą o niskim wymiarze wewnętrznym, zanurzoną w przestrzeni o wyższym wymiarze.

Hipoteza manifoldu mówi, że **obrazy zachowują się tak samo**. Obraz 224×224 to formalnie 150 528 liczb (jeden wymiar na każdy piksel każdego kanału). Ale "prawdziwe" obrazy, fotografie twarzy, krajobrazów, zwierząt, zajmują w tej olbrzymiej przestrzeni tylko cienką, niskowymiarową warstwę. Większość kombinacji 150 528 liczb to po prostu szum.

**Edycja obrazu to ruch po manifoldzie.** Jeśli przesuniemy obraz w *byle jakim* kierunku w przestrzeni pikseli, prawie na pewno zejdziemy z manifoldu naturalnych obrazów i dostaniemy szum. Żeby się tego ustrzec, potrzebny jest model, który *zna geometrię tego manifoldu*. Właśnie tym jest latent space modelu generatywnego. Ruch w latent space to ruch wzdłuż manifoldu, więc każdy pośredni punkt dekoduje się do prawdopodobnego obrazu. To dokładnie ta własność, która umożliwia gładkie przejścia między obrazami w projekcie.

## Idea: klasyfikator jako kompas, SD-VAE jako manifold

**Klasyfikator daje oś (kierunek). SD-VAE daje manifold (przestrzeń naturalnych obrazów). Gradient descent łączy je w spójny ruch.**

```
        ┌──────────────────┐
        │  SD-VAE latent   │  ← tu modyfikujemy z
        │     (4096D)      │
        └────────┬─────────┘
                 │ decode (SD-VAE)
                 ▼
        ┌──────────────────┐
        │   pixel image    │
        │   (256×256×3)    │
        └────────┬─────────┘
                 │ classify (EfficientNet)
                 ▼
        ┌──────────────────┐
        │     prob_AI      │  ← chcemy żeby spadło
        └────────┬─────────┘
                 │ gradient (autograd)
                 ▼
            ∂prob_AI/∂z         ← pochodna po latencie SD-VAE
```

Klasyfikator wie, *w którą stronę* trzeba pójść, by obraz przestał wyglądać jak AI. SD-VAE wie, *jak trzymać się manifoldu*, by pośrednie obrazy wciąż były obrazami, a nie szumem. Bez SD-VAE gradient płynący prosto do pikseli produkowałby klasyczny *adversarial example*: niewidoczny szum oszukujący klasyfikator bez zmiany wizualnej. Z SD-VAE każdy ruch latentu daje prawdopodobny obraz.

## Co optymalizujemy: wektor z

Cała pętla przesuwa **pojedynczy wektor `z`** w 4096-wymiarowej przestrzeni latentnej SD-VAE. Tylko to. Wagi SD-VAE (~84 mln parametrów) i klasyfikatora (~5 mln parametrów) są zamrożone od początku do końca.

| Komponent | Status |
| --- | --- |
| SD-VAE (encoder + decoder) | zamrożony |
| Klasyfikator (EfficientNetB0 + head) | zamrożony |
| **Wektor `z` (4 096 liczb)** | **optymalizowany** |

**Punkt startowy.** Oryginalny obraz przepuszczamy raz przez encoder SD-VAE. Otrzymujemy `z_0`, czyli 4096 liczb opisujących ten konkretny obraz w latencie. Następnie kopiujemy: `z = z_0.clone()`. Teraz `z` to liczby, które będziemy aktywnie zmieniać, a `z_0` zostaje jako kotwica.

**Pętla 80 iteracji.** W każdej iteracji:

1. Bierzemy aktualne `z` (już zmodyfikowane przez poprzednie kroki).
2. Dekodujemy je do obrazu (SD-VAE).
3. Klasyfikator ocenia ten *nowy* obraz i daje `prob_AI`.
4. Liczymy stratę: jak wysokie jest `prob_AI`, plus kara za oddalanie się od `z_0`.
5. Autograd wskazuje, w którą stronę zmienić 4096 liczb w `z`, by strata spadła.
6. Aktualizujemy `z`: każda z 4096 liczb przesuwa się o mały krok.

**Co dzieje się z `z` w trakcie.** Na początku `z` to obraz AI w latencie. Po pierwszej iteracji 4096 liczb lekko się przesuwa. Dekodowany obraz jest *prawie* identyczny, ale `prob_AI` spadł odrobinę. Po drugiej iteracji znowu się przesuwa. Dla obrazów Stable Diffusion i Midjourney wystarczają **2 do 3 iteracji**, by `prob_AI` spadł poniżej 0.1. Klasyfikator zmienia zdanie z "AI" na "Real". Dla DALL-E nawet 80 iteracji nie wystarcza, jego obrazy leżą zbyt daleko od granicy decyzji.

**Po zakończeniu.** Mamy nowy wektor `z_final`, przesunięty względem `z_0`, ale wciąż blisko niego (dzięki kotwicy regularyzacyjnej). Wagi obu modeli są dokładnie takie same jak na początku. Klasyfikator niczego się nie nauczył. To my przemieściliśmy się w przestrzeni.

**alpha_0.00**

<img width="768" height="768" alt="alpha_0 22_aiprob_0022" src="https://github.com/user-attachments/assets/da03f6ac-8489-41cb-9142-da57cc5869f2" />

**alpha_0.11**

<img width="768" height="768" alt="alpha_0 00_aiprob_0783" src="https://github.com/user-attachments/assets/880d357e-7cb7-46fa-b142-d71d77cf1b73" />

**alpha_0.22**

<img width="768" height="768" alt="alpha_0 22_aiprob_0022" src="https://github.com/user-attachments/assets/422d66bf-33e1-4e2d-8886-f788fea0f1fe" />

**alpha_0.33**

<img width="768" height="768" alt="alpha_0 33_aiprob_0006" src="https://github.com/user-attachments/assets/25780525-3f56-4aeb-9da8-537a78e7a860" />

**alpha_0.44**

<img width="768" height="768" alt="alpha_0 44_aiprob_0002" src="https://github.com/user-attachments/assets/b186ba2e-3cf9-4acf-94e1-d6f29d94cf89" />

**alpha_0.56**

<img width="768" height="768" alt="alpha_0 56_aiprob_0001" src="https://github.com/user-attachments/assets/fdff7051-d6da-407c-8881-e497ca850684" />

**alpha_0.67**

<img width="768" height="768" alt="alpha_0 67_aiprob_0000" src="https://github.com/user-attachments/assets/45bcd93e-4e78-464c-80a6-6dfa44a14694" />

**alpha_0.78**

<img width="768" height="768" alt="alpha_0 78_aiprob_0000" src="https://github.com/user-attachments/assets/34d81644-07d3-4a61-a98a-46b5adb02638" />

**alpha_0.89**

<img width="768" height="768" alt="alpha_0 89_aiprob_0000" src="https://github.com/user-attachments/assets/ddb8610c-1808-469b-877c-b784d64f331d" />


**alpha_1.00**

<img width="768" height="768" alt="alpha_1 00_aiprob_0000" src="https://github.com/user-attachments/assets/592f2dec-7352-4cd1-97ff-12d8f38e1cc5" />


## Dlaczego to nie jest trening

Klasyczny trening sieci uczy *wag modelu*. Szuka uniwersalnej funkcji, która dla każdego obrazu zwróci sensowną predykcję. Optymalizuje miliony parametrów. Trwa godziny.

Tutaj nie uczymy żadnych wag. Szukamy **pojedynczego punktu w przestrzeni latentnej dla jednego konkretnego obrazu**. Optymalizujemy 4096 liczb. Trwa to kilka sekund.

Analogicznie: trenowanie sieci to *uczenie kierowcy*, który potem radzi sobie w każdej sytuacji. Nasze podejście to *nawigacja istniejącym samochodem*. Mamy SD-VAE (modelarz obrazów) i klasyfikator (sędzia autentyczności). Obaj są już wytrenowani. Nie uczymy ich. Pytamy: *zaczynamy w punkcie `z_0`, sędzia ocenia go jako AI. W którą stronę pójść, by ocenił go niżej?*

Stąd trzy praktyczne konsekwencje. **Konwergencja jest szybka**: 2 do 3 iteracji zamiast tysięcy kroków treningu. **Efekt jest lokalny**: `z_final` znaleziony dla obrazu A nie pomaga dla obrazu B, każdy obraz wymaga osobnej optymalizacji. **Nie ma klasycznego overfittingu**, bo nie zapamiętujemy datasetu, tylko nawigujemy w gotowej przestrzeni. Pojawia się natomiast inny problem: *adversarial leakage*. Optymalizacja może znaleźć punkt oszukujący *nasz konkretny* klasyfikator bez prawdziwej semantycznej zmiany. Stąd transfer rate tylko 36% na niezależny detektor.

## Wnioski

Analiza wygenerowanych przejść i map Grad-CAM pokazała, że klasyfikator wraz z SD-VAE konsekwentnie modyfikuje kilka konkretnych cech obrazu:

1. **Kolor skóry** – odcień i jego rozkład na twarzy
   
![Uploading image.png…]()
 
3. **Zmarszczki** – obecność i geometria drobnych zmarszczek mimicznych
4. **Szum** – mikro-tekstura tła i powierzchni
5. **Światło** – kierunek i miękkość oświetlenia
6. **Nieostrość** – nieostrość w miejscach, gdzie nie powinno jej być
7. **Włosy** – tekstura, połysk, pojedyncze pasma

To są obszary, w których generatory AI zostawiają najwięcej śladów i które klasyfikator wykorzystuje, podejmując decyzję.

## Podsumowanie

**Granica między obrazami AI a fotografiami istnieje.** Klasyfikator znajduje ją z dokładnością ponad 99%, a separacja w jego przestrzeni cech jest tak silna (Cohen's $d' = 4.59$), że klasy ledwie się stykają. Co więcej, można po tej granicy *przechodzić*. Gładkim ruchem w przestrzeni latentnej SD-VAE da się przekształcić obraz AI w bardziej naturalny i odwrotnie. Granica jest mierzalna, kierunek sterowalny, ruch wzdłuż niego ciągły.

Co z tym zrobić, to już inne pytanie.

Kodak Brownie z 1900 roku, czyli pierwszy aparat dostępny dla każdego, wywołał obawy, że fotografia przestaje być rzemiosłem. Aparaty cyfrowe pod koniec lat 90. wzbudziły niechęć profesjonalistów, którym brakowało ziarna filmu i rytuału ciemni. Każda z tych zmian niosła obawę, że *coś prawdziwego zostaje utracone*. Każda też po kilku dekadach wytworzyła nową estetykę, którą uznano za naturalną.

Obrazy, które dziś nazywamy "prawdziwymi", same w sobie są już mocno zapośredniczone. Filtry Photoshopa, korekta kolorów, sztuczne oświetlenie, bokeh modelowany matematycznie przez smartfony. To wszystko jest częścią codzienności. Granica między *naturalnym* a *wygenerowanym* jest mniej wyraźna, niż się wydaje na pierwszy rzut oka.

Pierwsza intuicja jest taka, że **obrazy generowane przez AI powinny zachowywać swój charakter sztuczności**. Niech będą rozpoznawalne. To uczciwa odpowiedź na obecny moment, gdy generatory rozwijają się szybciej niż nasza zdolność weryfikacji źródła. Historia uczy ostrożności wobec takich intuicji, ale tempo dzisiejszych zmian jest inne niż przy Brownie czy pierwszych aparatach cyfrowych. Tamte rewolucje rozkładały się na dekady. Ta dzieje się w ciągu kilku lat.

Pytanie warte uwagi nie brzmi *czy obrazy AI są fałszywe*. Brzmi: **co długotrwałe oglądanie obrazów wygenerowanych robi z naszą percepcją i emocjami**.

Mózg ma osobny obszar do rozpoznawania twarzy. Setki tysięcy lat ewolucji dostroiły go do subtelności: emocji, intencji, autentyczności. Stała ekspozycja na obrazy, które są *prawie* prawdziwe, ale nie do końca, może generować trudny do nazwania niepokój. Może rozregulowywać kalibrację tego, co uznajemy za autentyczne. Może zacierać granicę między *widziałem* a *wyobraziłem sobie*.

### Następne kroki

**Nowsze generatory.** Dataset Defactify obejmuje generatory z lat 2022–2024 (SD 2.1, SD XL, SD 3, DALL-E 3, Midjourney 6). Nowe modele (FLUX, SD 3.5, Sora) produkują obrazy z innymi sygnaturami artefaktów. Warto sprawdzić, czy klasyfikator generalizuje, czy wymaga retreningu.

**Większe i bardziej zróżnicowane zbiory danych.** Defactify zawiera głównie obrazy typu "portret/scena". Warto rozszerzyć o krajobrazy, architekturę, makrofotografię, obrazy medyczne, czyli kategorie, w których artefakty AI mogą wyglądać inaczej.

**Poza twarzami.** Obecna analiza Grad-CAM koncentrowała się na skórze, włosach, oczach. Generatory zostawiają ślady również w teksturach tkanin, geometrii cieni, refleksach na powierzchniach. Warto sprawdzić, czy klasyfikator wykrywa je tak samo skutecznie poza domeną portretu.

**Szumy matrycy i analiza częstotliwości.** Prawdziwe aparaty zostawiają charakterystyczny szum sensora (PRNU, *Photo Response Non-Uniformity*), unikalny dla każdego egzemplarza. Analiza FFT obrazów może ujawnić, czy klasyfikator korzysta z tych niskopoziomowych cech, czy ze struktur semantycznych. To bezpośrednio odpowiada na pytanie *co konkretnie odróżnia AI od fotografii*.

**Dostrajanie per-obraz.** Optymalizacja wektora `z` używa globalnych hiperparametrów (`lr=0.05`, `λ=0.001`) dla wszystkich obrazów. Indywidualne dostrojenie tych parametrów dla każdego obrazu, na przykład większa kotwica regularyzacyjna dla obrazów już blisko granicy decyzji, może poprawić zarówno SSIM, jak i transfer rate na inne detektory.

## Referencje

## Convolutional networks (CNN architecture)

- **LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. (1998).** *Gradient-based learning applied to document recognition.* Proceedings of the IEEE.
  The original LeNet paper. Foundational reading on convolution, pooling, and the gradient learning loop.

- **Krizhevsky, A., Sutskever, I., Hinton, G. E. (2012).** *ImageNet Classification with Deep Convolutional Neural Networks.* NeurIPS 2012. [paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
  AlexNet. The paper that made CNNs the standard for image recognition.

- **Tan, M., Le, Q. V. (2019).** *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
  The architecture used as the backbone in this project. Describes compound scaling of depth, width, and resolution.

- **Goodfellow, I., Bengio, Y., Courville, A. (2016).** *Deep Learning.* MIT Press. [available free online](https://www.deeplearningbook.org/)
  Textbook with a thorough chapter on convolutional networks. The most accessible single source for understanding CNN building blocks.

## Transfer learning

- **Yosinski, J., Clune, J., Bengio, Y., Lipson, H. (2014).** *How transferable are features in deep neural networks?* NeurIPS 2014. [arXiv:1411.1792](https://arxiv.org/abs/1411.1792)
  The key paper on which features transfer well and which require retraining. Provides the empirical basis for the partial-unfreezing decision in the Model section.

- **Deng, J. et al. (2009).** *ImageNet: A Large-Scale Hierarchical Image Database.* CVPR 2009.
  Describes the dataset whose weights the backbone is pretrained on. Important context for understanding what transfers and what does not.

## CNN debugging (Grad-CAM, Saliency)

- **Zhou, B. et al. (2016).** *Learning Deep Features for Discriminative Localization.* CVPR 2016. [arXiv:1512.04150](https://arxiv.org/abs/1512.04150)
  Original CAM. The predecessor to Grad-CAM, requires a specific architecture (GAP before the classifier).

- **Selvaraju, R. R. et al. (2017).** *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
  The technique used in this project. Generalizes CAM to any CNN architecture by using gradients instead of weights.

- **Chattopadhay, A. et al. (2018).** *Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks.* WACV 2018. [arXiv:1710.11063](https://arxiv.org/abs/1710.11063)
  Refinement of Grad-CAM. Better at locating multiple instances or distributed evidence, which is exactly the case in AI-generated image detection.

- **Simonyan, K., Vedaldi, A., Zisserman, A. (2013).** *Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps.* [arXiv:1312.6034](https://arxiv.org/abs/1312.6034)
  The original Saliency Map paper. Older than Grad-CAM by four years, but still the conceptual foundation.

- **Smilkov, D. et al. (2017).** *SmoothGrad: removing noise by adding noise.* [arXiv:1706.03825](https://arxiv.org/abs/1706.03825)
  Saliency Map cleaned up by averaging over noisy copies of the input. The version used in the project.

- **Adebayo, J. et al. (2018).** *Sanity Checks for Saliency Maps.* NeurIPS 2018. [arXiv:1810.03292](https://arxiv.org/abs/1810.03292)
  Critical paper. Shows that some saliency methods produce similar-looking maps even when the model is broken. Important caveat for anyone interpreting these maps.

## Activation maximization

- **Erhan, D., Bengio, Y., Courville, A., Vincent, P. (2009).** *Visualizing Higher-Layer Features of a Deep Network.* University of Montreal Technical Report.
  Where activation maximization started. The idea of optimizing the input to maximize a neuron's response.

- **Olah, C., Mordvintsev, A., Schubert, L. (2017).** *Feature Visualization.* Distill. [distill.pub/2017/feature-visualization](https://distill.pub/2017/feature-visualization/)
  The most accessible single source on the topic. Interactive figures, clean explanations of regularization, multi-scale optimization, and why naive maximization fails. Recommended starting point.

- **Mordvintsev, A., Olah, C., Tyka, M. (2015).** *Inceptionism: Going Deeper into Neural Networks.* Google Research Blog. [blog post](https://blog.research.google/2015/06/inceptionism-going-deeper-into-neural.html)
  DeepDream. The work that made activation maximization widely known. Same family of techniques as the latent optimization in this project.

## Latent space and VAE

- **Kingma, D. P., Welling, M. (2013).** *Auto-Encoding Variational Bayes.* [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
  The original VAE paper. Derives the ELBO loss and the reparameterization trick.

- **Doersch, C. (2016).** *Tutorial on Variational Autoencoders.* [arXiv:1606.05908](https://arxiv.org/abs/1606.05908)
  The clearest introduction to VAEs available. Read this before the original Kingma & Welling if you want intuition first and equations later.

- **Higgins, I. et al. (2017).** *beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.* ICLR 2017. [paper](https://openreview.net/forum?id=Sy2fzU9gl)
  Shows how to encourage disentangled latent directions, that is, latents where individual coordinates correspond to interpretable features. Relevant for understanding why latent directions can be meaningful.

- **Radford, A., Metz, L., Chintala, S. (2015).** *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.* [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
  DCGAN. Famous for the "king - man + woman = queen" style latent-vector arithmetic. The intuition behind the AI-ness axis search.

- **Rombach, R. et al. (2022).** *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
  The Stable Diffusion paper. Describes the SD-VAE used in this project (the encoder/decoder, not the diffusion process itself).

## Accessible tutorials and learning resources

Materials at a more introductory level than the papers above. Useful for picking up intuition before diving into the original sources, or for sharing the project with non-specialist readers.

### Neural network foundations

- **3Blue1Brown.** *Neural Networks playlist (4 videos).* [youtube.com/playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
  The most visually striking introduction to backpropagation and gradient descent. Aimed at viewers with high-school-level math.

- **Welch Labs.** *Neural Networks Demystified (7-part series).* [welchlabs.com](http://www.welchlabs.com/blog/2015/1/16/neural-networks-demystified-part-1-data-and-architecture) · [GitHub notebooks](https://github.com/stephencwelch/Neural-Networks-Demystified)
  Builds a working neural network from scratch in Python, derives every equation along the way. Short videos paired with companion iPython notebooks.

- **Nielsen, M. (2015).** *Neural Networks and Deep Learning.* [neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com/)
  Free online book. The canonical intuitive introduction. Covers everything up to convolutional networks, with worked code examples.

- **TensorFlow Playground.** [playground.tensorflow.org](https://playground.tensorflow.org/)
  Train a small neural network in the browser and see what each neuron learns. Probably the best 10-minute investment for building intuition.

### Practical deep learning and transfer learning

- **fast.ai.** *Practical Deep Learning for Coders.* [course.fast.ai](https://course.fast.ai/)
  Top-down course. Starts with a working classifier in the first half-hour and explains the theory afterwards. Strong emphasis on transfer learning, the exact workflow used in this project.

- **Karpathy, A.** *Neural Networks: Zero to Hero.* [karpathy.ai/zero-to-hero](https://karpathy.ai/zero-to-hero.html)
  YouTube series that builds up from a single neuron to a small transformer, all in code. Slower-paced than 3Blue1Brown, but lands the implementation details.

### Visualization and interpretability

- **Olah, C.** *colah's blog.* [colah.github.io](https://colah.github.io/)
  Multiple posts directly relevant to this project: *Understanding Convolutions*, *Conv Nets: A Modular Perspective*, *Visualizing Representations*, *Neural Networks, Manifolds, and Topology*. The last one is particularly useful for the manifold hypothesis in the Latent Space section.

- **Distill.pub.** [distill.pub](https://distill.pub/)
  Online journal of interactive articles on ML interpretability. Beyond the Feature Visualization piece in the academic references, see also *The Building Blocks of Interpretability* and *Activation Atlases*.

- **Alammar, J.** *Visualizing Machine Learning, One Concept at a Time.* [jalammar.github.io](https://jalammar.github.io/)
  Heavily illustrated explanations, mostly of NLP models but applicable in style and approach. A model of how to communicate ML concepts visually.

### Generative models, latent space, VAE

- **Kogan, G.** *Machine Learning for Artists (ml4a).* [ml4a.net](https://ml4a.net/)
  Designed for non-technical readers, but mathematically honest. Strong chapters on autoencoders, latent space walks, and generative models. The closest match in spirit to what this project tries to do.

- **Weng, L. (2018).** *From Autoencoder to Beta-VAE.* [lilianweng.github.io](https://lilianweng.github.io/posts/2018-08-12-vae/)
  Single-post tutorial covering autoencoders, VAEs, β-VAE, and VQ-VAE. Bridges the gap between a first-pass intro and Kingma & Welling's original paper.

## Practical tools and libraries

- **tf-keras-vis** (Kubota, K.). [github.com/keisen/tf-keras-vis](https://github.com/keisen/tf-keras-vis)
  TensorFlow/Keras library covering Grad-CAM, Grad-CAM++, Saliency, SmoothGrad, Activation Maximization. Used in the CNN visualization exercise.

- **pytorch-grad-cam** (Gildenblat, J.). [github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
  PyTorch equivalent. Broad coverage of CAM-family methods.

- **Captum** (Meta AI). [captum.ai](https://captum.ai/)
  PyTorch interpretation library. Covers saliency methods, integrated gradients, attribution techniques. More extensive than pytorch-grad-cam.

- **diffusers** (HuggingFace). [github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)
  Provides ready-to-use SD-VAE (`stabilityai/sd-vae-ft-mse`) and other generative-model components. The library used in the latent-editing pipeline.

- **TensorFlow / Keras documentation.** [tensorflow.org/tutorials](https://www.tensorflow.org/tutorials)
  Practical tutorials on transfer learning, fine-tuning, and Keras applications including EfficientNetB0.



# AI vs Real: detecting and editing AI-generated images

## Inspirations

AI-generated images can be stunning. The shapes line up, the colors are in place, the composition makes sense. And yet a portrait from a diffusion model often produces a sense of unease. The skin looks too smooth, the light falls at an odd angle, the contrast feels unnatural. It is hard to point to a specific cause. The image is "suspicious", but the reason stays out of reach.

This is no accident. The human brain has a dedicated region for face recognition, known as the *fusiform face area*. Over hundreds of thousands of years, evolution tuned this system to catch subtle cues: emotion, intent, authenticity. Even small departures from the **distribution of natural faces** trigger an alarm.

Hence the idea for this project, deliberately old-school. A convolutional neural network (CNN) trained on the task "camera vs. AI". For a CNN, an image is not a set of concepts ("face", "smile") but **statistical patterns of pixels**: textures, gradients, micro-noise. Exactly the things the human eye does not consciously register, but which still shape our perception. There are also visualization techniques that let us see which parts of the image the network looks at when making a decision.

As the first results came in, new questions emerged. Eventually an idea inspired by latent space took shape: generate a sequence of images going from AI to real. Analogous to a latent-space transition from a smiling face to a sad one.

<img  alt="image" src="https://github.com/user-attachments/assets/4ed65dd8-9b2f-4f34-8eb7-6d38602b89cf" />

## Process

### Dataset

**Source.** The `Rajarshi-Roy-research/Defactify_Image_Dataset` available on HuggingFace. About 47,000 images in six categories:

| Category | Count | Origin |
| --- | --- | --- |
| Real | ~7,000 | Photographs from ImageNet and COCO |
| Stable Diffusion 2.1 | ~7,000 | Open-source diffusion model |
| Stable Diffusion XL | ~7,000 | Next generation of SD |
| Stable Diffusion 3 | ~7,000 | Latest generation of SD |
| DALL-E 3 | ~7,000 | OpenAI commercial model |
| Midjourney 6 | ~7,000 | Midjourney commercial model |

Five AI generators against one category of real photographs. This produces a strong class imbalance (roughly 6:1 in favor of AI), compensated during training by weighting the loss function. The Real class carries six times the weight of AI.

**Labeling.** Each image has two labels: `Label_A` (binary, Real=0, AI=1) and `Label_B` (six classes, identifies the specific generator). The classifier uses only `Label_A`, but `Label_B` is essential for the per-generator analysis that follows.

### Model: transfer learning from EfficientNetB0

**Goal.** Train a binary classifier that maps an image to a probability $\text{prob}_{\text{AI}} \in [0, 1]$. The architecture has to be powerful enough for the task, but transparent enough that its internals can be analyzed with interpretation techniques.

**Starting point.** EfficientNetB0 pretrained on ImageNet. A deliberate choice: around 5 million parameters (a relatively small model), well-known architecture, available weights, a good balance between expressive power and training cost.

**Architecture.** On top of the EfficientNetB0 backbone (which ends with a 1280-channel 7×7 feature map), a simple classifier is built:

```
EfficientNetB0 → GlobalAveragePooling2D → Dropout(0.4) →
Dense(256, ReLU) → Dropout(0.3) → Dense(1, sigmoid)
```

**Key design decision: partial unfreezing of the backbone.** Standard transfer learning freezes the entire backbone and trains only the head. In this task such an approach fails for a fundamental reason:

> ImageNet teaches the network to recognize *objects* (cat, car, bridge). AI-generated detection requires recognizing *generator artifacts*: subtle texture patterns, frequency statistics, compression signatures. Features of this kind are simply not encoded in weights trained on ImageNet.

The fix: only the first ~100 layers (out of about 236) are frozen, the remaining ~136 are trained together with the head. Low-level edge and texture detectors stay untouched (they are universal), while higher layers adapt to the specifics of AI vs Real.

## CNN debugging

What does the network look at when making a decision? Two techniques are available: Grad-CAM and Saliency Map. They help us understand which parts of an image influence the AI vs Real decision. This may be exactly what reveals the features of AI-generated images.

### CAM (Class Activation Mapping)

CAM answers the question: **where in the image** did the model find the relevant features? The network already has everything it needs for this. The feature maps from the last convolutional layer say *where*: each one is a grid of activations in image space. The classifier weights say *what* matters for a given class. By default these two pieces of information drift apart, because the GlobalAveragePooling layer averages each map to a single number and loses the "where". CAM reverses the order: it first multiplies the feature maps by the classifier weights, and only then averages. The result is a spatial map, which is scaled to the image size as a heatmap.



### Saliency Map

Saliency answers the question: **which pixels** weighed most heavily on the prediction? The mechanism is the same as in network training: backpropagation, that is, gradient computation. Only the endpoint is different. In training, the gradient stops at the weights to update them. In Saliency, it keeps flowing all the way to the input pixels, where it is captured as a relevance map. Bright pixels are those whose small change would shift the output the most. Dark ones are those the model is insensitive to.

<img alt="01_xai_panel" src="https://github.com/user-attachments/assets/22430000-c27f-40b9-8aca-eaf22524c06c" />

## Latent Space

Once we know which regions and pixels matter, we can try modifying AI images to make them look more real. The inspiration came from VAE algorithms and the ability to morph images through latent space. The canonical example: a generated face going from smiling to sad. The goal here was the analogous example, only going from an AI-generated face to a real one.

Latent space is the **internal space of a generative model** (e.g. VAE, GAN) from which images can be produced. Every point in it corresponds to some plausible image. It is compressed, smooth, and continuous. Nearby points decode to similar images, and a smooth transition between two points yields a smooth transition between two images.

To understand *why* latent space makes sense at all, an analogy helps. A crumpled sheet of paper tossed into a room physically lives in three-dimensional space. But internally it is **two-dimensional**. Two coordinates are enough to identify any point on its surface. The sheet is a *manifold*: a structure with low internal dimension, embedded in a space of higher dimension.

The manifold hypothesis says that **images behave the same way**. A 224×224 image is formally 150,528 numbers (one dimension for every pixel of every channel). But "real" images, photographs of faces, landscapes, animals, occupy only a thin, low-dimensional sheet within that enormous space. Most combinations of 150,528 numbers are simply noise.

**Editing an image is movement along the manifold.** If we push an image in *any* direction in pixel space, we will almost certainly fall off the manifold of natural images and end up with noise. To avoid this, we need a model that *knows the geometry of this manifold*. That is exactly what the latent space of a generative model gives us. Movement in latent space is movement along the manifold, so every intermediate point decodes to a plausible image. This is precisely the property that makes the smooth transitions in this project possible.

## Idea: classifier as compass, SD-VAE as manifold

**The classifier provides the axis (direction). SD-VAE provides the manifold (the space of natural images). Gradient descent ties them into a coherent motion.**

```
        ┌──────────────────┐
        │  SD-VAE latent   │  ← this is where we modify z
        │     (4096D)      │
        └────────┬─────────┘
                 │ decode (SD-VAE)
                 ▼
        ┌──────────────────┐
        │   pixel image    │
        │   (256×256×3)    │
        └────────┬─────────┘
                 │ classify (EfficientNet)
                 ▼
        ┌──────────────────┐
        │     prob_AI      │  ← we want this to drop
        └────────┬─────────┘
                 │ gradient (autograd)
                 ▼
            ∂prob_AI/∂z         ← derivative w.r.t. SD-VAE latent
```

The classifier knows *which way* to go for the image to stop looking AI. SD-VAE knows *how to stay on the manifold*, so that intermediate images are still images and not noise. Without SD-VAE, a gradient flowing straight into the pixels would produce a classic *adversarial example*: invisible noise that fools the classifier without any visual change. With SD-VAE, every move in the latent yields a plausible image.

## What we optimize: the vector z

The entire loop moves a **single vector `z`** through the 4096-dimensional latent space of SD-VAE. That is the whole of it. The weights of SD-VAE (~84M parameters) and of the classifier (~5M parameters) are frozen from start to finish.

| Component | Status |
| --- | --- |
| SD-VAE (encoder + decoder) | frozen |
| Classifier (EfficientNetB0 + head) | frozen |
| **Vector `z` (4,096 numbers)** | **optimized** |

**Starting point.** The original image goes through the SD-VAE encoder once. We get `z_0`: the 4096 numbers that describe this particular image in the latent. Then we copy: `z = z_0.clone()`. From now on, `z` is the set of numbers we will actively modify, while `z_0` stays as an anchor.

**Loop of 80 iterations.** In each iteration:

1. Take the current `z` (already modified by previous steps).
2. Decode it into an image (SD-VAE).
3. The classifier rates this *new* image and outputs `prob_AI`.
4. Compute the loss: how high `prob_AI` is, plus a penalty for moving away from `z_0`.
5. Autograd tells us in which direction to push the 4096 numbers in `z` so that the loss decreases.
6. Update `z`: each of the 4096 numbers moves by a small step.

**What happens to `z` along the way.** At the start, `z` is an AI image in the latent. After the first iteration, the 4096 numbers shift slightly. The decoded image is *almost* identical, but `prob_AI` has dropped a little. After the second iteration, they shift again. For Stable Diffusion and Midjourney images, **2 to 3 iterations** are enough for `prob_AI` to fall below 0.1. The classifier changes its mind from "AI" to "Real". For DALL-E even 80 iterations are not enough; its images sit too far from the decision boundary.

**After the loop.** We have a new vector `z_final`, shifted relative to `z_0` but still close to it (thanks to the regularization anchor). The weights of both models are exactly the same as at the start. The classifier has not learned anything. We are the ones who moved through the space.

**alpha_0.00**

<img width="768" height="768" alt="alpha_0 22_aiprob_0022" src="https://github.com/user-attachments/assets/da03f6ac-8489-41cb-9142-da57cc5869f2" />

**alpha_0.11**

<img width="768" height="768" alt="alpha_0 00_aiprob_0783" src="https://github.com/user-attachments/assets/880d357e-7cb7-46fa-b142-d71d77cf1b73" />

**alpha_0.22**

<img width="768" height="768" alt="alpha_0 22_aiprob_0022" src="https://github.com/user-attachments/assets/422d66bf-33e1-4e2d-8886-f788fea0f1fe" />

**alpha_0.33**

<img width="768" height="768" alt="alpha_0 33_aiprob_0006" src="https://github.com/user-attachments/assets/25780525-3f56-4aeb-9da8-537a78e7a860" />

**alpha_0.44**

<img width="768" height="768" alt="alpha_0 44_aiprob_0002" src="https://github.com/user-attachments/assets/b186ba2e-3cf9-4acf-94e1-d6f29d94cf89" />

**alpha_0.56**

<img width="768" height="768" alt="alpha_0 56_aiprob_0001" src="https://github.com/user-attachments/assets/fdff7051-d6da-407c-8881-e497ca850684" />

**alpha_0.67**

<img width="768" height="768" alt="alpha_0 67_aiprob_0000" src="https://github.com/user-attachments/assets/45bcd93e-4e78-464c-80a6-6dfa44a14694" />

**alpha_0.78**

<img width="768" height="768" alt="alpha_0 78_aiprob_0000" src="https://github.com/user-attachments/assets/34d81644-07d3-4a61-a98a-46b5adb02638" />

**alpha_0.89**

<img width="768" height="768" alt="alpha_0 89_aiprob_0000" src="https://github.com/user-attachments/assets/ddb8610c-1808-469b-877c-b784d64f331d" />


**alpha_1.00**

<img width="768" height="768" alt="alpha_1 00_aiprob_0000" src="https://github.com/user-attachments/assets/592f2dec-7352-4cd1-97ff-12d8f38e1cc5" />

## Why this is not training

Classical network training teaches *the weights of the model*. It searches for a universal function that returns a sensible prediction for any image. It optimizes millions of parameters. It takes hours.

Here we are not training any weights. We are looking for **a single point in latent space for one specific image**. We optimize 4096 numbers. It takes a few seconds.

By analogy: training a network is *teaching the driver*, who then handles any situation. Our approach is *navigating with an existing car*. We have SD-VAE (the image modeler) and the classifier (the authenticity judge). Both are already trained. We are not teaching them. We are asking them: *we start at point `z_0`, the judge rates it as AI. Which way should we go for it to rate lower?*

Three practical consequences follow. **Convergence is fast**: 2 to 3 iterations instead of thousands of training steps. **The effect is local**: `z_final` found for image A does not help with image B; every image requires its own optimization. **There is no classical overfitting**, because we are not memorizing the dataset, only navigating an existing space. A different problem appears instead: *adversarial leakage*. The optimization may find a point that fools *our specific* classifier without producing a real semantic change. This is why the transfer rate to an independent detector is only 36%.

## Findings

Analysis of the generated transitions and the Grad-CAM maps revealed that the classifier together with SD-VAE consistently modifies a handful of specific image features:

1. **Skin color** — the tone and how it is distributed across the face
2. **Wrinkles** — the presence and geometry of small expression lines
3. **Noise** — micro-texture of the background and surfaces
4. **Light** — direction and softness of the lighting
5. **Sharpness** — soft focus in places where it does not belong
6. **Hair** — texture, gloss, individual strands

These are the areas where AI generators leave the most traces, and the ones the classifier draws on when making its decision.

## Conclusion

**A boundary between AI images and photographs exists.** The classifier finds it with accuracy above 99%, and the separation in its feature space is so strong (Cohen's $d' = 4.59$) that the classes barely touch. What is more, this boundary can be *crossed*. A smooth motion in SD-VAE latent space can turn an AI image into something more natural, and vice versa. The boundary is measurable, the direction is steerable, the motion along it is continuous.

What to do with that is a different question.

The Kodak Brownie of 1900, the first camera available to anyone, sparked fears that photography was no longer a craft. Digital cameras in the late 1990s met with resistance from professionals who missed film grain and the ritual of the darkroom. Each of these shifts carried the worry that *something true is being lost*. Each, after a few decades, produced a new aesthetic that came to be regarded as natural.

The images we call "real" today are themselves already heavily mediated. Photoshop filters, color grading, artificial lighting, bokeh modeled mathematically by smartphones. All of this is part of daily life. The line between *natural* and *generated* is fuzzier than it appears at first glance.

The first intuition is that **AI-generated images should retain their artificial character**. Let them be recognizable. That is an honest answer for the present moment, when generators evolve faster than our ability to verify a source. History teaches caution toward intuitions of this kind, but the pace of today's changes is unlike that of the Brownie or the first digital cameras. Those revolutions unfolded over decades. This one is happening in a few years.

The question worth asking is not *whether AI images are fake*. It is: **what does prolonged exposure to generated images do to our perception and our emotions?**

The brain has a dedicated region for recognizing faces. Hundreds of thousands of years of evolution tuned it to subtleties: emotion, intent, authenticity. Steady exposure to images that are *almost* real, but not quite, may produce a hard-to-name unease. It may unsettle our calibration of what we treat as authentic. It may blur the line between *I saw* and *I imagined*.

### Next steps

**Newer generators.** The Defactify dataset covers generators from 2022 to 2024 (SD 2.1, SD XL, SD 3, DALL-E 3, Midjourney 6). Newer models (FLUX, SD 3.5, Sora) produce images with different artifact signatures. Worth checking whether the classifier generalizes, or needs retraining.

**Larger and more diverse datasets.** Defactify contains mostly "portrait/scene" images. Worth extending to landscapes, architecture, macro photography, medical images: categories in which AI artifacts may look different.

**Beyond faces.** The current Grad-CAM analysis focused on skin, hair, and eyes. Generators also leave traces in fabric textures, the geometry of shadows, reflections on surfaces. Worth checking whether the classifier picks them up just as effectively outside the portrait domain.

**Sensor noise and frequency analysis.** Real cameras leave a characteristic sensor noise (PRNU, *Photo Response Non-Uniformity*), unique to each individual unit. FFT analysis of images can reveal whether the classifier draws on these low-level cues or on semantic structures. This addresses directly the question *what exactly separates AI from a photograph*.

**Per-image tuning.** The optimization of `z` uses global hyperparameters (`lr=0.05`, `λ=0.001`) for all images. Tuning these per image, for example a stronger regularization anchor for images already close to the decision boundary, may improve both SSIM and transfer rate to other detectors.

## References


References are grouped by the topics covered in the README. Each entry has a brief note explaining what the work brings to the table, so you can pick by need rather than by chronology.

## Convolutional networks (CNN architecture)

- **LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. (1998).** *Gradient-based learning applied to document recognition.* Proceedings of the IEEE.
  The original LeNet paper. Foundational reading on convolution, pooling, and the gradient learning loop.

- **Krizhevsky, A., Sutskever, I., Hinton, G. E. (2012).** *ImageNet Classification with Deep Convolutional Neural Networks.* NeurIPS 2012. [paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
  AlexNet. The paper that made CNNs the standard for image recognition.

- **Tan, M., Le, Q. V. (2019).** *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
  The architecture used as the backbone in this project. Describes compound scaling of depth, width, and resolution.

- **Goodfellow, I., Bengio, Y., Courville, A. (2016).** *Deep Learning.* MIT Press. [available free online](https://www.deeplearningbook.org/)
  Textbook with a thorough chapter on convolutional networks. The most accessible single source for understanding CNN building blocks.

## Transfer learning

- **Yosinski, J., Clune, J., Bengio, Y., Lipson, H. (2014).** *How transferable are features in deep neural networks?* NeurIPS 2014. [arXiv:1411.1792](https://arxiv.org/abs/1411.1792)
  The key paper on which features transfer well and which require retraining. Provides the empirical basis for the partial-unfreezing decision in the Model section.

- **Deng, J. et al. (2009).** *ImageNet: A Large-Scale Hierarchical Image Database.* CVPR 2009.
  Describes the dataset whose weights the backbone is pretrained on. Important context for understanding what transfers and what does not.

## CNN debugging (Grad-CAM, Saliency)

- **Zhou, B. et al. (2016).** *Learning Deep Features for Discriminative Localization.* CVPR 2016. [arXiv:1512.04150](https://arxiv.org/abs/1512.04150)
  Original CAM. The predecessor to Grad-CAM, requires a specific architecture (GAP before the classifier).

- **Selvaraju, R. R. et al. (2017).** *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
  The technique used in this project. Generalizes CAM to any CNN architecture by using gradients instead of weights.

- **Chattopadhay, A. et al. (2018).** *Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks.* WACV 2018. [arXiv:1710.11063](https://arxiv.org/abs/1710.11063)
  Refinement of Grad-CAM. Better at locating multiple instances or distributed evidence, which is exactly the case in AI-generated image detection.

- **Simonyan, K., Vedaldi, A., Zisserman, A. (2013).** *Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps.* [arXiv:1312.6034](https://arxiv.org/abs/1312.6034)
  The original Saliency Map paper. Older than Grad-CAM by four years, but still the conceptual foundation.

- **Smilkov, D. et al. (2017).** *SmoothGrad: removing noise by adding noise.* [arXiv:1706.03825](https://arxiv.org/abs/1706.03825)
  Saliency Map cleaned up by averaging over noisy copies of the input. The version used in the project.

- **Adebayo, J. et al. (2018).** *Sanity Checks for Saliency Maps.* NeurIPS 2018. [arXiv:1810.03292](https://arxiv.org/abs/1810.03292)
  Critical paper. Shows that some saliency methods produce similar-looking maps even when the model is broken. Important caveat for anyone interpreting these maps.

## Activation maximization

- **Erhan, D., Bengio, Y., Courville, A., Vincent, P. (2009).** *Visualizing Higher-Layer Features of a Deep Network.* University of Montreal Technical Report.
  Where activation maximization started. The idea of optimizing the input to maximize a neuron's response.

- **Olah, C., Mordvintsev, A., Schubert, L. (2017).** *Feature Visualization.* Distill. [distill.pub/2017/feature-visualization](https://distill.pub/2017/feature-visualization/)
  The most accessible single source on the topic. Interactive figures, clean explanations of regularization, multi-scale optimization, and why naive maximization fails. Recommended starting point.

- **Mordvintsev, A., Olah, C., Tyka, M. (2015).** *Inceptionism: Going Deeper into Neural Networks.* Google Research Blog. [blog post](https://blog.research.google/2015/06/inceptionism-going-deeper-into-neural.html)
  DeepDream. The work that made activation maximization widely known. Same family of techniques as the latent optimization in this project.

## Latent space and VAE

- **Kingma, D. P., Welling, M. (2013).** *Auto-Encoding Variational Bayes.* [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
  The original VAE paper. Derives the ELBO loss and the reparameterization trick.

- **Doersch, C. (2016).** *Tutorial on Variational Autoencoders.* [arXiv:1606.05908](https://arxiv.org/abs/1606.05908)
  The clearest introduction to VAEs available. Read this before the original Kingma & Welling if you want intuition first and equations later.

- **Higgins, I. et al. (2017).** *beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.* ICLR 2017. [paper](https://openreview.net/forum?id=Sy2fzU9gl)
  Shows how to encourage disentangled latent directions, that is, latents where individual coordinates correspond to interpretable features. Relevant for understanding why latent directions can be meaningful.

- **Radford, A., Metz, L., Chintala, S. (2015).** *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.* [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
  DCGAN. Famous for the "king - man + woman = queen" style latent-vector arithmetic. The intuition behind the AI-ness axis search.

- **Rombach, R. et al. (2022).** *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
  The Stable Diffusion paper. Describes the SD-VAE used in this project (the encoder/decoder, not the diffusion process itself).

## Accessible tutorials and learning resources

Materials at a more introductory level than the papers above. Useful for picking up intuition before diving into the original sources, or for sharing the project with non-specialist readers.

### Neural network foundations

- **3Blue1Brown.** *Neural Networks playlist (4 videos).* [youtube.com/playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
  The most visually striking introduction to backpropagation and gradient descent. Aimed at viewers with high-school-level math.

- **Welch Labs.** *Neural Networks Demystified (7-part series).* [welchlabs.com](http://www.welchlabs.com/blog/2015/1/16/neural-networks-demystified-part-1-data-and-architecture) · [GitHub notebooks](https://github.com/stephencwelch/Neural-Networks-Demystified)
  Builds a working neural network from scratch in Python, derives every equation along the way. Short videos paired with companion iPython notebooks.

- **Nielsen, M. (2015).** *Neural Networks and Deep Learning.* [neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com/)
  Free online book. The canonical intuitive introduction. Covers everything up to convolutional networks, with worked code examples.

- **TensorFlow Playground.** [playground.tensorflow.org](https://playground.tensorflow.org/)
  Train a small neural network in the browser and see what each neuron learns. Probably the best 10-minute investment for building intuition.

### Practical deep learning and transfer learning

- **fast.ai.** *Practical Deep Learning for Coders.* [course.fast.ai](https://course.fast.ai/)
  Top-down course. Starts with a working classifier in the first half-hour and explains the theory afterwards. Strong emphasis on transfer learning, the exact workflow used in this project.

- **Karpathy, A.** *Neural Networks: Zero to Hero.* [karpathy.ai/zero-to-hero](https://karpathy.ai/zero-to-hero.html)
  YouTube series that builds up from a single neuron to a small transformer, all in code. Slower-paced than 3Blue1Brown, but lands the implementation details.

### Visualization and interpretability

- **Olah, C.** *colah's blog.* [colah.github.io](https://colah.github.io/)
  Multiple posts directly relevant to this project: *Understanding Convolutions*, *Conv Nets: A Modular Perspective*, *Visualizing Representations*, *Neural Networks, Manifolds, and Topology*. The last one is particularly useful for the manifold hypothesis in the Latent Space section.

- **Distill.pub.** [distill.pub](https://distill.pub/)
  Online journal of interactive articles on ML interpretability. Beyond the Feature Visualization piece in the academic references, see also *The Building Blocks of Interpretability* and *Activation Atlases*.

- **Alammar, J.** *Visualizing Machine Learning, One Concept at a Time.* [jalammar.github.io](https://jalammar.github.io/)
  Heavily illustrated explanations, mostly of NLP models but applicable in style and approach. A model of how to communicate ML concepts visually.

### Generative models, latent space, VAE

- **Kogan, G.** *Machine Learning for Artists (ml4a).* [ml4a.net](https://ml4a.net/)
  Designed for non-technical readers, but mathematically honest. Strong chapters on autoencoders, latent space walks, and generative models. The closest match in spirit to what this project tries to do.

- **Weng, L. (2018).** *From Autoencoder to Beta-VAE.* [lilianweng.github.io](https://lilianweng.github.io/posts/2018-08-12-vae/)
  Single-post tutorial covering autoencoders, VAEs, β-VAE, and VQ-VAE. Bridges the gap between a first-pass intro and Kingma & Welling's original paper.

## Practical tools and libraries

- **tf-keras-vis** (Kubota, K.). [github.com/keisen/tf-keras-vis](https://github.com/keisen/tf-keras-vis)
  TensorFlow/Keras library covering Grad-CAM, Grad-CAM++, Saliency, SmoothGrad, Activation Maximization. Used in the CNN visualization exercise.

- **pytorch-grad-cam** (Gildenblat, J.). [github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
  PyTorch equivalent. Broad coverage of CAM-family methods.

- **Captum** (Meta AI). [captum.ai](https://captum.ai/)
  PyTorch interpretation library. Covers saliency methods, integrated gradients, attribution techniques. More extensive than pytorch-grad-cam.

- **diffusers** (HuggingFace). [github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)
  Provides ready-to-use SD-VAE (`stabilityai/sd-vae-ft-mse`) and other generative-model components. The library used in the latent-editing pipeline.

- **TensorFlow / Keras documentation.** [tensorflow.org/tutorials](https://www.tensorflow.org/tutorials)
  Practical tutorials on transfer learning, fine-tuning, and Keras applications including EfficientNetB0.

