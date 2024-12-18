% This encoding is independent of representation
on(0, N) :- input(N), inpfix(N).
{ on(0, K) } :- not inpfix(K), layer(0, N), K = 1..N.
