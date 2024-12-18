% !binarised 01
% This encoding has to be used with (0, 1)-binarised inputs
on(L, N) :-
    -B <= #sum{ W,M : on(L-1, M), weight(L, M, N, W) },
    bias(L, N, B), not output_layer(L).
