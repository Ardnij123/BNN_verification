% !binarised pm1
% This encoding has to be used with (+1, -1)-binarised inputs
on(L, N) :-
    -B <= #sum{
         W,M :     on(L-1, M), weight(L, M, N, W);
        -W,M : not on(L-1, M), weight(L, M, N, W) },
    bias(L, N, B), not output_layer(L).
