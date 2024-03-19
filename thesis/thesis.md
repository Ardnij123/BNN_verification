# Verification of binarized neural network using ASP

Formální věci...

## Úvod

- Co řešíme, proč to řešíme
	+ NP-úplné problémy
		* zde bychom mohli verifikovat ex-post
	+ problémy, u kterých neznáme řešení
		* případně verifikovat řešení může být samo exponenciálně náročné
	+ kritické problémy
		* při špatném výsledku nás to může stát hodně životů/peněz
- aktuální stav verifikace BNN a neuronek obecně (SAT/SMT)
- práce související s tématem (state of the art)
- Existuje nějaký framework?
- jaké jsou aktuální limity pro velikosti verifikovaných sítí
- možná rozdělit na 2 kapitoly?

## Použité nástroje

- obecně o ASP
	+ solver pro NP-úplné problémy
- Clingo
	+ rozdělení celého frameworku
	+ základy použití, kódování problému

## Můj přínos

- definice problému
- proč je moje řešení korektní
- kódování sítě, výpočtu na ní
- kódování vstupu, výstupu
- využití agregátních fcí - mohlo by vést ke zlepšení?
	+ agg funkce jako speciální případ OBDD

## Evaluation

- ohodnocení modelu
	+ je rozdíl mezi hledání, které vrátí málo výsledků vůči hodně výsledkům
	+ time to first / time to all
	+ clingo je (asi) optimalizované na hledání jednoho řešení, tady je chceme všechny
	+ -> Je rozumné hledat od nejvíce svázaného zadání a postupně rozšiřovat?
- Jaký je rozumný limit pro velikost verifikované sítě
	+ je problém spíše s širokými/hlubokými sítěmi
- jak si vede vůči původnímu zdroji, dalším implementacím
- TODO: asi to vrazit na Metacentrum či tak něco? - můj ntb by pravděpodobně zkresloval výsledky

## Diskuse

- má ASP pro další verifikaci význam?
- je možnost, že by přinesl alespoň relevantní benchmark pro další verifikační algoritmy (např. nad obecnými NN?)
	+ není velká možnost, že by porazil algoritmy přizpůsobené přímo pro daný problém, ale je tu možnost že by byl fajn dokud nebudeme mít něco silnějšího než ASP pro problém