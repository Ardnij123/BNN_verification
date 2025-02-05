% This encoding is independent of representation
on(0, N) :- input(N), inpfix(N).
-on(0, N) :- not input(N), inpfix(N).
1 { on(0, K); -on(0, K) } 1 :- not inpfix(K), layer(0, N), K = 1..N.
