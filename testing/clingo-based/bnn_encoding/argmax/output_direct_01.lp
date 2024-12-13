% !binarised 01
% This encoding has to be used with (0, 1)-binarised inputs
1 { output(1..N) } 1 :- output_layer(D+1), layer(D+1, N).

:- output(N),
    C - B > #sum{  W,O, 1 : on(D, O), weight(D+1, O, N, W) ;
                  -W,O,-1 : on(D, O), weight(D+1, O, M, W) },
    bias(D+1, N, B), bias(D+1, M, C),
    output_layer(D+1),
    N != M.
:- output(N),
    C - B >= #sum{  W,O, 1 : on(D, O), weight(D+1, O, N, W) ;
                  -W,O,-1 : on(D, O), weight(D+1, O, M, W) },
    bias(D+1, N, B), bias(D+1, M, C),
    outpre(N, P), outpre(M, Q),
    output_layer(D+1),
    N != M, P > Q.
