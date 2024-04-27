% Layer sizes
% layer=0 - input layer
% layer(layer, size)

% Which layer is for output
% output_layer(layer)
output_layer(L) :- L = #max{ X : layer(X, _) }, layer(L, _).

% Weights
% weight(from_layer, from_node, to_node, weight)

% Biases
% bias(layer, node, bias)

% On
% on(layer, node)
% input layer
{on(0, 0..K-1)} :- layer(0, K).
%*
% Equivalent to both in speed and value
% on(0, I) :- not off(0, I), layer(0, K), I = 0..K-1.
% off(0, I) :- not on(0, I), layer(0, K), I = 0..K-1.
*%

% hidden layers
on(L, N) :-
    #sum{ W,I : on(L-1, I), weight(L-1, I, N, W) } >= B,
    bias(L-1, N, B), not output_layer(L).

%* Equivalent to (slower)
on(L, N) :-
    Pos = #count{ I : on(L-1, I), weight(L-1, I, N, 1) },
    Neg = #count{ I : on(L-1, I), weight(L-1, I, N, -1) },
    Pos - Neg >= B,
    bias(L-1, N, B), not output_layer(L).
*%

% Output precedence
% highest order -> best candidate for output
% outpre(output, order)

% Outnode
% Sum of weights and bias
% outnode(node, sum)
% outnode(N, S + B) :-
%     S = #sum{ W,I : on(L-1, I), weight(L-1, I, N, W) },
%     output_layer(L), bias(L-1, N, B).

%* Equivalent to (same speed; slower)
outnode(N, S) :-
    S = #sum{ W,I : on(L-1, I), weight(L-1, I, N, W); B,-1 },
    output_layer(L), bias(L-1, N, B).

outnode(N, B + Pos - Neg) :-
    Pos = #count{ I : on(L-1, I), weight(L-1, I, N, 1)},
    Neg = #count{ I : on(L-1, I), weight(L-1, I, N, -1)},
    output_layer(L), bias(L-1, N, B).
*%

% Output
% output(node)
1 {output(0..N-1)} 1 :- layer(L, N), output_layer(L).

% the sum of another is higher
:- output(N), O = 1..M-1, output_layer(L), layer(L, M), O != N,
    #sum{ W, I, N : on(L-1, I), weight(L-1, I, N, W);
         -W, I, O : on(L-1, I), weight(L-1, I, O, W);
          BN, -1; -BO, -2} < 0,
    bias(L-1, N, BN), bias(L-1, O, BO).
% the sum of another is same and order is higher
:- output(N), O = 1..M-1, output_layer(L), layer(L, M), O != N,
    #sum{ W, I, N : on(L-1, I), weight(L-1, I, N, W);
         -W, I, O : on(L-1, I), weight(L-1, I, O, W);
          BN, -1; -BO, -2} = 0,
    bias(L-1, N, BN), bias(L-1, O, BO),
    outpre(N, PN), outpre(O, PO), PO > PN.
% sum and order of another is the same and node.no is lower
:- output(N), O = 1..M-1, output_layer(L), layer(L, M), O != N,
    #sum{ W, I, N : on(L-1, I), weight(L-1, I, N, W);
         -W, I, O : on(L-1, I), weight(L-1, I, O, W);
          BN, -1; -BO, -2} = 0,
    bias(L-1, N, BN), bias(L-1, O, BO),
    outpre(N, PN), outpre(O, PO), PO = PN,
    N > O.

% output(Node) :-
%     (Sum, Order, -Node) = #max{ (S, O, -N) : outnode(N, S), outpre(N, O) }.
#show output/1.
% #show on/2.


% Target
% :- not on(2, 3). % bit on index 3 in layer 2 is on

% Input vector
% input(node, value)

% each input has only 1 value
:- input(N, V1), input(N, V2), V1 != V2.

% Inputs that have to be equal
% inputOn(node)
% inputOff(node)

% input layer has form of inputOn/Off
:- inputOn(N), not on(0, N).
:- inputOff(N), on(0, N).

% Max hamming distance from input
% hamdist(distance)
% if not used, just do not insert it

% input space is at most hamdist from on
:- #count{ N : input(N, -1), on(0, N); N : input(N, 1), not on(0, N) } > H,
    hamdist(H).

% Show only inputs with output nonequal to that of input vector
inputOn(0, N) :- input(N, 1).
inputOn(L, N) :-
    #sum{ W,I : inputOn(L-1, I), weight(L-1, I, N, W) } >= B,
    bias(L-1, N, B), not output_layer(L).

inputOutnode(N, S) :-
    S = #sum{ W,I : inputOn(L-1, I), weight(L-1, I, N, W); B,-1 },
    output_layer(L), bias(L-1, N, B).

inputOutput(Node) :-
    (Sum, Order, -Node) = #max{ (S, O, -N) : inputOutnode(N, S), outpre(N, O) }.

:- output(Node), inputOutput(Node).
