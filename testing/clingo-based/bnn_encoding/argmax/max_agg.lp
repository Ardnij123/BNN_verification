% !binarised pm1
% This encoding has to be used with (+1, -1)-binarised inputs
potential(D+1, N, S+B) :-
    S = #sum{
         W,M :     on(D, M), weight(D, M, N, W);
        -W,M : not on(D, M), weight(D, M, N, W) },
    bias(D, N, B), output_layer(D+1).

output(N) :-
    (Sum, Precedence, Node) = #max{
        (S, -P, N) : potential(D+1, N, S), outpre(N, P) },
    output_layer(D+1).
