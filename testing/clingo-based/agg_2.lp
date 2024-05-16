% THIS PROGRAM DOES USE ONLY CONSTRAINTS
% WHEN COMPUTING OUTPUT
% YIELDING MUCH SMALLER GROUND PROGRAM AND FASTER COMPUTE

% Layer sizes
% layer=0 - input layer
% layer(layer, size)

% Which layer is for output
% output_layer(layer)
output_layer(L) :- L = #max{ X : layer(X, _) }.

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
    #sum{ W,I : on(L-1, I), weight(L, I, N, W) } >= -B,
    bias(L, N, B), not output_layer(L).

%* Equivalent to (slower)
on(L, N) :-
    Pos = #count{ I : on(L-1, I), weight(L, I, N, 1) },
    Neg = #count{ I : on(L-1, I), weight(L, I, N, -1) },
    Pos - Neg >= B,
    bias(L, N, B), not output_layer(L).
*%

% Output precedence
% highest order -> best candidate for output
% outpre(output, order)

% Outnode
% Sum of weights and bias
% outnode(node, sum)
% outnode(N, S + B) :-
%     S = #sum{ W,I : on(L-1, I), weight(L, I, N, W) },
%     output_layer(L), bias(L, N, B).

%* Equivalent to (same speed; slower)
outnode(N, S) :-
    S = #sum{ W,I : on(L-1, I), weight(L, I, N, W); B,-1 },
    output_layer(L), bias(L, N, B).

outnode(N, B + Pos - Neg) :-
    Pos = #count{ I : on(L-1, I), weight(L, I, N, 1)},
    Neg = #count{ I : on(L-1, I), weight(L, I, N, -1)},
    output_layer(L), bias(L, N, B).
*%

% Output
% output(node)
1 {output(0..N-1)} 1 :- layer(L, N), output_layer(L).

% the sum of another is higher
:- output(N), O = 0..M-1, output_layer(L), layer(L, M), O != N,
    #sum{ W, I, this  : on(L-1, I), weight(L, I, N, W);
         -W, I, other : on(L-1, I), weight(L, I, O, W) } < BO - BN,
    bias(L, N, BN), bias(L, O, BO).
% the sum of another is same and order is higher
:- output(N), O = 0..M-1, output_layer(L), layer(L, M), O != N,
    #sum{ W, I, this  : on(L-1, I), weight(L, I, N, W);
         -W, I, other : on(L-1, I), weight(L, I, O, W) } = BO - BN,
    bias(L, N, BN), bias(L, O, BO),
    outpre(N, PN), outpre(O, PO), PO > PN.
% % sum and order of another is the same and node.no is lower
% :- output(N), O = 1..M-1, output_layer(L), layer(L, M), O != N,
%     #sum{ W, I, N : on(L, I), weight(L, I, N, W);
%          -W, I, O : on(L, I), weight(L, I, O, W);
%           BN, -1; -BO, -2} = 0,
%     bias(L, N, BN), bias(L, O, BO),
%     outpre(N, PN), outpre(O, PO), PO = PN,
%     N > O.

% output(Node) :-
%     (Sum, Order, -Node) = #max{ (S, O, -N) : outnode(N, S), outpre(N, O) }.
#show output/1.
% #show on/2.


% Target
% :- not on(2, 3). % bit on index 3 in layer 2 is on

% Input vector
% input(node, value)

% % each input has only 1 value
% :- input(N, V1), input(N, V2), V1 != V2.

% % Inputs that have to be equal
% % inputOn(node)
% % inputOff(node)

% % input layer has form of inputOn/Off
% :- inputOn(N), not on(0, N).
% :- inputOff(N), on(0, N).
:- inpfix(N), on(0, N), not input(N).
:- inpfix(N), not on(0, N), input(N).

% Max hamming distance from input
% hamdist(distance)
% if not used, just do not insert it

% input space is at most hamdist from on
:- #count{ N : not input(N), on(0, N); N : input(N), not on(0, N) } > H,
    hamdist(H).

% Show only inputs with output nonequal to that of input vector
inputOn(0, N) :- input(N).
inputOn(L, N) :-
    #sum{ W,I : inputOn(L-1, I), weight(L, I, N, W) } >= -B,
    bias(L, N, B), not output_layer(L).

% Output
% output(node)
1 {inputOutput(0..N-1)} 1 :- layer(L, N), output_layer(L).

% the sum of another is higher
:- inputOutput(N), O = 0..M-1, output_layer(L), layer(L, M), O != N,
    #sum{ W, I, this  : inputOn(L-1, I), weight(L, I, N, W);
         -W, I, other : inputOn(L-1, I), weight(L, I, O, W) } < BO - BN,
    bias(L, N, BN), bias(L, O, BO).
% the sum of another is same and order is higher
:- inputOutput(N), O = 0..M-1, output_layer(L), layer(L, M), O != N,
    #sum{ W, I, this  : inputOn(L-1, I), weight(L, I, N, W);
         -W, I, other : inputOn(L-1, I), weight(L, I, O, W) } = BO - BN,
    bias(L, N, BN), bias(L, O, BO),
    outpre(N, PN), outpre(O, PO), PO > PN.
#show inputOutput/1.
% #show on/2.
% #show inputOn/2.

:- output(Node), inputOutput(Node).
