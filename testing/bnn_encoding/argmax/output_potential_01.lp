% !binarised 01
% This encoding has to be used with (0, 1)-binarised inputs
potential(D+1, N, S+B) :-
    S = #sum{ W,M : on(D, M), weight(D+1, M, N, W) },
    bias(D+1, N, B), output_layer(D+1).

1 { output(1..N) } 1 :- output_layer(D+1), layer(D+1, N).

:- output(N),
    potential(D+1, N, S), potential(D+1, M, T),
    output_layer(D+1),
    N != M, S < T.
:- output(N),
    potential(D+1, N, S), potential(D+1, M, T),
    outpre(N, P), outpre(M, Q),
    output_layer(D+1),
    N != M, S = T, P > Q.
