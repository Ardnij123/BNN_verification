% !binarised pm1
% This encoding has to be used with (+1, -1)-binarised inputs
potential(D+1, N, S+B) :-
    S = #sum{
         W,M :     on(D, M), weight(D+1, M, N, W);
        -W,M : not on(D, M), weight(D+1, M, N, W) },
    bias(D+1, N, B), output_layer(D+1).

output(Node) :-
    (Sum, Precedence, Node) = #max{
        (S, -P, N) : potential(D+1, N, S), outpre(N, P) },
    output_layer(D+1).
