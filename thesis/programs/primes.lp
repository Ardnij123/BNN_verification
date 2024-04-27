prime(X) :- not comp(X), X = 2..20.
comp(X * Y) :- X = 2..20, Y = 2..20, X * Y <= 20.
