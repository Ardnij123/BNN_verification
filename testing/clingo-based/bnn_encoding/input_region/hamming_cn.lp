% This encoding is independent of representation
1 { on(0, K); -on(0, K) } 1 :- layer(0, N), K=1..N.

:- #count{ N :  on(0, N), not input(N);
           N : -on(0, N),     input(N) } > R,
    hammdist(R).
