% This encoding is independent of representation
{ on(0, 1..N) } :- layer(0, N).

:-     on(0, N), not input(N), inpfix(N).
:- not on(0, N),     input(N), inpfix(N).
