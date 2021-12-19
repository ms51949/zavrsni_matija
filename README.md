# zavrsni_matija
Geografski ponderirana neuronska mreža


GWR-Geographically weighted regression

    -GWR.py - ovdje se nalazi kod za GWR
            - data - podaci koje koristim u regresiji(toy4.csv-podaci koji su dobiveni pomoću jednadžbe u članku(slika 11)) 
            - X_train- 2 stupca neovisnih varijabli(podaci za treniranje)
            - Y_train- 1 stupac ovisne varijable(tražene vrijednosti)
            - LOCATION_train - lokacije varijabli
            - def Wi-l-lokacija
                    -locations-sve lokacije iz LOCATION_train
                    -bandwitdh- to je mjera koja određuje u kojoj mjeri okolne lokacije imaju utjecaja na trenutnu lokaciju

                    -vraća dijagonalnu matricu koja na dijagobnali imama vrijednosti koje gove kolko neka lokacija prima utjecaja od ostalih lokacija
            -def fit-X- skup podataka za treniranje
                    -y- tražene vrijednosti
                    -locations
                    -bandwitdh
                    
                    -pom1 i pom2 su samo pomočne varijable za računanje bete
                    -beta-je matrica dimenzija(1,3) koju čini vrijednost
                    za svaku od 3 funkcije na toj lokaciji (vrijednosti funkcija uz 1, x1, i x2)

                    -vrćamo betu

            -fitData-vraća matricu koja sadrži za svaku lokaciju u LOCATION_train
                    njezinu betu

    - ja sam prošao kroz svaki toy i dobio predicdion za vrijednosti funkcija koje se nalaze uz (1, x1, x2),
    a spremljene su pod nazivom npr. za toy4 "toy4Coef.txt",pa sam zakomentirao liniju za spremanje rezultata,
    prikaz rezultata je moguć je pomoću ShowCoef.py


    -ShowCoef.py - kod za prikaz dobivenih rezultata GWR.py


GWANN- Geographically weighted artificial neural network

        - GWANN.py  - kod za neuronsku mrežu
                    - na početku uzimam potrebne podatke kao i u GWR
                    -inicijaliziram loss funkciju sa bandwidthom i neuronsku mrežu
                    - funkcija create_mini_batches stvara mini batchove, a funkcija gradient vraća error i gradient n askupupu dobivenin podataka i trenutne neuronske mreže
                    - u GWANN sam stavio tri optimizacijska algoritma adam, nestrov i mini-batch
                    - najuspješniji se pokazao adam pa njega koristim kod treniranja mreže
                    - pokretanjem koda inicijalizira se nova ne naučena neuronska mreža te se pokretanjem određene funkcije(adam, nestrov ili  mini batches)pokreče njezino
                    treniranje te nakon što su gotove sve iteracije se spremaju težine koje spajaju neurone s vanjskim neuronima(5 neurona + 1 bias),te spremim prediction za y
                    
        - LossFun.py - funkcija za loss funkciju
                    - u njoj radim geografsko ponderiranje, kada ju inicijaliziram stvorim matricu gdjeimam vrijednost koja govori koliko svaka lokacija utječe jedna na drugu
                    (n lokacija =>(n, n) matrica )
        -showNeurons.py- pokretanjem koda prikazuju se neuroni
        -data file- u fileu se nalazi kod za stvaranje i prikaz koeficijenata za stvaranje podataka koji se koriste u istraživanju
                    
